import json
import re
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from src.crews.simulation_crew import SimulationCrew


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

    @start()
    def init_request(self):
        return self.state.model_dump()

    @listen(init_request)
    def trigger_crew_inference(self):
        inputs = {
            "user_id": self.state.user_id,
            "item_id": self.state.item_id
        }

        result = None
        try:
            result = SimulationCrew().crew().kickoff(inputs=inputs)
            data = None
            pydantic_obj = getattr(result, "pydantic", None)

            if pydantic_obj is not None and not callable(pydantic_obj):
                data = pydantic_obj.model_dump()

            if data is None:
                raw = getattr(result, "raw", str(result))
                data = extract_json_from_output(raw)

            if not isinstance(data, dict):
                data = extract_json_from_output(str(data))

            self.state.predicted_rating = float(
                data.get("stars", data.get("predicted_rating", 4.0))
            )
            self.state.generated_review = str(
                data.get("review", data.get("generated_review", "Good."))
            )

        except Exception as e:
            raw = getattr(result, "raw", str(result)) if result else str(e)
            print("Parse error:", e)
            print("Raw result:", raw)

            self.state.predicted_rating = 4.0
            self.state.generated_review = raw

        return self.state.model_dump()