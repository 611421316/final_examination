import json
import re
from typing import Any, Dict

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

try:
    from src.crews.simulation_crew import SimulationCrew, normalize_final_output
except Exception:
    from simulation_crew import SimulationCrew, normalize_final_output

try:
    from src.tools.exact_lookup_tools import build_prediction_context
except Exception:
    from exact_lookup_tools import build_prediction_context


def extract_json_from_output(raw_output: Any) -> Dict[str, Any]:
    """Extract final JSON from CrewAI output with fallback repair."""
    if raw_output is None:
        return {"stars": 4.0, "review": "Good."}

    if isinstance(raw_output, dict):
        return raw_output

    if hasattr(raw_output, "pydantic") and raw_output.pydantic:
        try:
            return raw_output.pydantic.model_dump()
        except Exception:
            pass

    if hasattr(raw_output, "raw"):
        raw_output = raw_output.raw

    text = str(raw_output).strip()
    text = text.replace("```json", "").replace("```", "").strip()
    text = text.replace("{{", "{").replace("}}", "}")

    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False

        for idx in range(start, len(text)):
            ch = text[idx]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

                if depth == 0:
                    try:
                        parsed = json.loads(text[start : idx + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break

    star_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:stars?|分|顆星)", text, re.IGNORECASE)
    rating = float(star_match.group(1)) if star_match else 4.0

    return {
        "stars": rating,
        "review": text if text else "Good.",
    }


def call_build_prediction_context(user_id: str, item_id: str) -> Dict[str, Any]:
    """
    Call the deterministic Python lookup/tool before CrewAI runs.

    This prevents the retriever LLM from hallucinating fake context such as
    "Cafe Bliss" and ensures config/tasks.yaml can safely use {prediction_context}.
    """
    try:
        raw_context = build_prediction_context.run(
            user_id=user_id,
            item_id=item_id,
        )
    except TypeError:
        try:
            raw_context = build_prediction_context.run({
                "user_id": user_id,
                "item_id": item_id,
            })
        except AttributeError:
            raw_context = build_prediction_context(user_id=user_id, item_id=item_id)
    except AttributeError:
        raw_context = build_prediction_context(user_id=user_id, item_id=item_id)

    if isinstance(raw_context, str):
        try:
            return json.loads(raw_context)
        except json.JSONDecodeError:
            start = raw_context.find("{")
            end = raw_context.rfind("}")
            if start >= 0 and end > start:
                return json.loads(raw_context[start : end + 1])
            raise

    if isinstance(raw_context, dict):
        return raw_context

    return json.loads(str(raw_context))


class InferenceState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    predicted_rating: float = 0.0
    generated_review: str = ""


class AgentSocietyServingFlow(Flow[InferenceState]):
    def __init__(self, agents_config_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agents_config_path = agents_config_path

    @start()
    def init_request(self):
        return {
            "user_id": self.state.user_id,
            "item_id": self.state.item_id,
        }

    @listen(init_request)
    def trigger_crew_inference(self):
        prediction_context = call_build_prediction_context(
            user_id=self.state.user_id,
            item_id=self.state.item_id,
        )

        predicted_stars = float(prediction_context.get("predicted_stars", 3.8))

        inputs = {
            "user_id": self.state.user_id,
            "item_id": self.state.item_id,
            "prediction_context": json.dumps(prediction_context, ensure_ascii=False),
            "predicted_stars": predicted_stars,
        }

        print("[FLOW DEBUG] prediction_context injected:", True)
        print("[FLOW DEBUG] predicted_stars:", predicted_stars)
        print("[FLOW DEBUG] case:", prediction_context.get("case"))
        print("[FLOW DEBUG] rating_weight_trace:", prediction_context.get("rating_weight_trace"))

        crew_instance = SimulationCrew()

        if self.agents_config_path:
            import yaml

            with open(self.agents_config_path, "r", encoding="utf-8") as f:
                crew_instance.agents_config = yaml.safe_load(f)

        result = crew_instance.crew().kickoff(inputs=inputs)

        try:
            data = normalize_final_output(result)
        except Exception:
            data = extract_json_from_output(result)

        # Deterministic safety: final stars must equal Python predicted_stars.
        # Even if the final LLM returns only "5" or invalid JSON, keep the official
        # rating controlled by build_prediction_context.
        self.state.predicted_rating = predicted_stars
        self.state.generated_review = str(data.get("review", "Good."))

        return self.state.model_dump()
