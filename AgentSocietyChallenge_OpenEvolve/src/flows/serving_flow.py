import json
import os
import re
import sys
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
# NOTE: SimulationCrew is imported lazily inside trigger_crew_inference() to
# avoid the "super(type, obj)" class-identity error that occurs when
# simulation_crew.py is imported under two different sys.modules keys.
# Always importing it fresh (with stale-cache eviction) guarantees a single
# canonical class object per process.


def extract_json_from_output(raw_output: str) -> dict:
    """Extract and sanitize JSON from LLM raw output with regex fallback."""
    text = str(raw_output).strip()
    
    # Fix double curly braces {{ }} -> { }
    text = text.replace('{{', '{').replace('}}', '}')
    
    # Strategy 1: Try to find a JSON object containing "stars" and "review"
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find a JSON with "predicted_rating" and "generated_review"
    match = re.search(r'\{[^{}]*"predicted_rating"[^{}]*"generated_review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try to extract a star rating number from free text
    star_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|分|顆星)', text, re.IGNORECASE)
    rating = float(star_match.group(1)) if star_match else 4.0

    return {"stars": rating, "review": text}


class InferenceState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    predicted_rating: float = 0.0
    generated_review: str = ""

class AgentSocietyServingFlow(Flow[InferenceState]):
    def __init__(self, agents_config_path: str = None, *args, **kwargs):
        # super().__init__ MUST come first — CrewAI's Flow base class is
        # Pydantic-backed and resets the instance __dict__ during init.
        # Setting custom attributes beforehand causes them to disappear.
        super().__init__(*args, **kwargs)
        self.agents_config_path = agents_config_path

    @start()
    def init_request(self):
        # 初始化階段，紀錄收到的 user_id 和 item_id
        pass

    @listen(init_request)
    def trigger_crew_inference(self):
        # 定義傳遞到任務 {user_id} 與 {item_id} 變數的值
        inputs = {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id
        }

        # ── Lazy import with stale-cache eviction ─────────────────────────
        # Evict any previously cached module objects so that we always get
        # ONE canonical SimulationCrew class, regardless of how many times
        # this module or crewai_simulation_agent has been imported.
        # Without this, Python's zero-argument super() cell captures a
        # *different* class object than the one used to construct the instance,
        # producing: "super(type, obj): obj is not an instance or subtype"
        for _key in list(sys.modules.keys()):
            if 'simulation_crew' in _key:
                del sys.modules[_key]
        from src.crews.simulation_crew import SimulationCrew  # noqa: PLC0415

        # 啟動並執行 Crew AI 團隊
        # NOTE: SimulationCrew.__init__ reads OPENEVOLVE_AGENTS_YAML /
        # OPENEVOLVE_TASKS_YAML from env at instantiation time, so the env vars
        # set by openevolve_evaluator.py are automatically picked up here.
        crew_instance = SimulationCrew()

        # ── Override agents config via explicit arg (crewai_simulation_agent.py) ──
        # Only apply when an explicit path is provided AND the file still exists
        # (temp files created by OpenEvolve may be cleaned up before this runs).
        if self.agents_config_path and os.path.exists(self.agents_config_path):
            import yaml
            with open(self.agents_config_path, "r", encoding="utf-8") as f:
                crew_instance.agents_config = yaml.safe_load(f)
        elif self.agents_config_path:
            print(f"[ServingFlow] Warning: agents_config_path not found: {self.agents_config_path!r}, using env/default.")

        result = crew_instance.crew().kickoff(inputs=inputs)
        
        # 使用多層 Regex 容錯解析 LLM 的回傳結果
        try:
            if result.pydantic:
                data = result.pydantic.model_dump()
            else:
                data = extract_json_from_output(result.raw)

            self.state.predicted_rating = float(data.get('stars', data.get('predicted_rating', 4.0)))
            self.state.generated_review = str(data.get('review', data.get('generated_review', 'Good.')))
        except Exception:
            # 最終備援：把整段 raw output 當 review 用
            self.state.predicted_rating = 4.0
            self.state.generated_review = str(result.raw)

        return self.state.model_dump()
