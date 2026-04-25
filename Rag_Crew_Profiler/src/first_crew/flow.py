from crewai.flow.flow import Flow, start, listen
from first_crew.crew import FirstCrew
import json


class ReviewPredictionFlow(Flow):

    @start()
    def prepare_input(self):
         # === Step 1: Understand the Target & Dataset ===
        # Read the first entry from the test set as a demo
        test_json_path = "data/test_review_subset.json"
        with open(test_json_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            test_case = json.loads(first_line)

        inputs = {
            'user_id': test_case['user_id'],
            'item_id': test_case['item_id']
        }
        print(f"Starting Prediction for User: {inputs['user_id']} | Item: {inputs['item_id']}")
        return inputs

    @listen(prepare_input)
    def run_sequential_crew(self, inputs):
        result = FirstCrew().sequential_crew().kickoff(inputs=inputs)
        return result


def kickoff():
    print("Run flow")
    flow = ReviewPredictionFlow()
    result = flow.kickoff()
    return result