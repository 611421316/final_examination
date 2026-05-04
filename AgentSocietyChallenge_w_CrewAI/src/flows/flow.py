from crewai.flow.flow import Flow, start, listen, router
from src.crews.crew_sequential import SequentialCrew
import json
from pathlib import Path

class ReviewPredictionFlow(Flow):

    @start()
    def prepare_input(self):
        test_json_path = Path("data/test_review_subset.json")
        try:
                if not test_json_path.exists():
                    raise FileNotFoundError(f"File not found: {test_json_path}")

                with test_json_path.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()

                if not first_line:
                    raise ValueError("The test file is empty.")

                test_case = json.loads(first_line)

                required_keys = ["user_id", "item_id"]
                for key in required_keys:
                    if key not in test_case:
                        raise KeyError(f"Missing required key: {key}")

                inputs = {
                    "user_id": test_case["user_id"],
                    "item_id": test_case["item_id"],
                }

                self.state["inputs"] = inputs
                return inputs

        except Exception as e:
            print(f"Please make sure your input data is correct: {e}")
            self.state["inputs"] = None
            return None


    @router(prepare_input)
    def route_by_sentiment(self, data):
        if data is not None:
            return "run_crew_path"
        return "alert_path"


    @listen("alert_path")
    def handle_alert(self):
        print("🚨 Data is None, Exiting Flow")
        return {
            "status": "failed",
            "message": "Invalid input data"
        }


    @listen("run_crew_path")
    def run_sequential_crew(self):
        inputs = self.state.get("inputs")

        if not isinstance(inputs, dict):
            return {
                "status": "failed",
                "message": f"inputs must be dict, got {type(inputs).__name__}"
            }

        result = SequentialCrew().crew().kickoff(inputs=inputs)
        return result

def kickoff():
    print("Run flow")
    flow = ReviewPredictionFlow()
    result = flow.kickoff()
    print(result)
    return result

if __name__ == "__main__":
        kickoff()