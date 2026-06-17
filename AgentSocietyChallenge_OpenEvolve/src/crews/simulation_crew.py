import json
from pathlib import Path
from typing import Any, Dict

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

try:
    from src.tools.exact_lookup_tools import build_prediction_context
except Exception:
    try:
        from exact_lookup_tools import build_prediction_context
    except Exception as exc:
        build_prediction_context = None
        _IMPORT_ERROR = exc
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        agents_path = parent / "config" / "agents.yaml"
        tasks_path = parent / "config" / "tasks.yaml"

        if agents_path.exists() and tasks_path.exists():
            return parent

    cwd = Path.cwd()
    if (cwd / "config" / "agents.yaml").exists() and (cwd / "config" / "tasks.yaml").exists():
        return cwd

    return cwd


_PROJECT_ROOT = find_project_root()


def extract_json_object(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    if start < 0:
        return "{}"

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
                return text[start : idx + 1]

    return text[start:]


def safe_json_loads(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "json_dict") and value.json_dict:
        return value.json_dict

    if hasattr(value, "pydantic") and value.pydantic:
        try:
            return value.pydantic.model_dump()
        except Exception:
            pass

    if hasattr(value, "raw"):
        value = value.raw

    text = str(value).strip()
    text = text.replace("{{", "{").replace("}}", "}")
    json_text = extract_json_object(text)

    try:
        parsed = json.loads(json_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def normalize_final_output(result: Any) -> Dict[str, Any]:
    parsed = safe_json_loads(result)

    stars = parsed.get("stars", parsed.get("predicted_stars", 4.0))
    try:
        stars = float(stars)
    except Exception:
        stars = 4.0

    stars = max(1.0, min(5.0, stars))

    review = parsed.get("review", parsed.get("generated_review", ""))
    if not isinstance(review, str):
        review = str(review)

    review = review.strip()
    if not review:
        review = (
            "Overall, this was a solid experience. The place had enough positives "
            "to make it worth a visit, even if there were a few small things that "
            "kept it from being perfect."
        )

    return {"stars": stars, "review": review}


@CrewBase
class SimulationCrew:
    agents_config = str(_PROJECT_ROOT / "config" / "agents.yaml")
    tasks_config = str(_PROJECT_ROOT / "config" / "tasks.yaml")

    @agent
    def exact_yelp_data_retriever(self) -> Agent:
        if build_prediction_context is None:
            raise ImportError(
                "Cannot import build_prediction_context from exact_lookup_tools. "
                f"Original error: {_IMPORT_ERROR}"
            )

        return Agent(
            config=self.agents_config["exact_yelp_data_retriever"],
            tools=[build_prediction_context],
            allow_delegation=False,
            max_iter=2,
            verbose=True,
        )

    @agent
    def yelp_case_detection_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["yelp_case_detection_analyst"],
            tools=[],
            allow_delegation=False,
            max_iter=1,
            verbose=True,
        )

    @agent
    def yelp_user_behavior_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["yelp_user_behavior_analyst"],
            tools=[],
            allow_delegation=False,
            max_iter=1,
            verbose=True,
        )

    @agent
    def yelp_item_review_context_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["yelp_item_review_context_analyst"],
            tools=[],
            allow_delegation=False,
            max_iter=1,
            verbose=True,
        )

    @agent
    def deterministic_yelp_review_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config["deterministic_yelp_review_simulator"],
            tools=[],
            allow_delegation=False,
            max_iter=1,
            verbose=True,
        )

    @task
    def retrieve_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_data_task"],
            agent=self.exact_yelp_data_retriever(),
        )

    @task
    def detect_case_task(self) -> Task:
        return Task(
            config=self.tasks_config["detect_case_task"],
            agent=self.yelp_case_detection_analyst(),
            context=[self.retrieve_data_task()],
        )

    @task
    def analyze_user_behavior_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_user_behavior_task"],
            agent=self.yelp_user_behavior_analyst(),
            context=[
                self.retrieve_data_task(),
                self.detect_case_task(),
            ],
        )

    @task
    def analyze_item_review_context_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_item_review_context_task"],
            agent=self.yelp_item_review_context_analyst(),
            context=[
                self.retrieve_data_task(),
                self.detect_case_task(),
                self.analyze_user_behavior_task(),
            ],
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config["predict_review_task"],
            agent=self.deterministic_yelp_review_simulator(),
            context=[
                self.retrieve_data_task(),
                self.detect_case_task(),
                self.analyze_user_behavior_task(),
                self.analyze_item_review_context_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.exact_yelp_data_retriever(),
                self.yelp_case_detection_analyst(),
                self.yelp_user_behavior_analyst(),
                self.yelp_item_review_context_analyst(),
                self.deterministic_yelp_review_simulator(),
            ],
            tasks=[
                self.retrieve_data_task(),
                self.detect_case_task(),
                self.analyze_user_behavior_task(),
                self.analyze_item_review_context_task(),
                self.predict_review_task(),
            ],
            process=Process.sequential,
            verbose=True,
        )


def run_simulation(user_id: str, item_id: str) -> Dict[str, Any]:
    result = SimulationCrew().crew().kickoff(
        inputs={
            "user_id": user_id,
            "item_id": item_id,
        }
    )
    return normalize_final_output(result)


if __name__ == "__main__":
    output = run_simulation(
        user_id="sample_user_id",
        item_id="sample_item_id",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))