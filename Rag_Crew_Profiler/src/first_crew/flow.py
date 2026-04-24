from crewai.flow.flow import Flow, start, listen
from first_crew.crew import FirstCrew


class ReviewPredictionFlow(Flow):

    @start()
    def prepare_input(self):
        return {
            "user_id": "_BcWyKQL16ndpBdggh2kNA",
            "item_id": "uBDXcXlLR9IuRV1N2m0SPQ",
            "query": "Predict user preference and optimize review explanation"
        }

    @listen(prepare_input)
    def run_sequential_crew(self, inputs):
        result = FirstCrew().sequential_crew().kickoff(inputs=inputs)
        return result


def kickoff():
    print("Run flow")
    flow = ReviewPredictionFlow()
    result = flow.kickoff()
    return result