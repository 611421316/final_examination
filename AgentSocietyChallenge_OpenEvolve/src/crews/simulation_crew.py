import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# === LLM Provider Selection ===
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
print("Here is provider: ", llm_provider)
if llm_provider == "nvidia":
    default_llm = LLM(
        model=f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'meta/llama-3.1-8b-instruct')}",
        api_key=os.getenv("NVIDIA_API_KEY", ""),
        base_url=os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
    )
else:
    default_llm = LLM(model="ollama/phi3")

@CrewBase
class SimulationCrew():
    """Yelp Recommendation Simulation Crew (Single Agent)"""
    agents_config = str(_PROJECT_ROOT / 'config' / 'agents.yaml')
    tasks_config  = str(_PROJECT_ROOT / 'config' / 'tasks.yaml')

    @agent
    def user_analyst(self):
        return Agent(
            config=self.agents_config["user_analyst"],
            verbose=True,
            llm=default_llm
        )

    @agent
    def item_analyst(self):
        return Agent(
            config=self.agents_config["item_analyst"],
            verbose=True,
            llm=default_llm
        )

    @agent
    def prediction_modeler(self):
        return Agent(
            config=self.agents_config["prediction_modeler"],
            verbose=True,
            llm=default_llm
        )

    @task
    def analyze_user_task(self):
        cfg = dict(self.tasks_config["analyze_user_task"])
        cfg.pop("agent", None)

        return Task(
            config=cfg,
            agent=self.user_analyst()
        )


    @task
    def analyze_item_task(self):
        cfg = dict(self.tasks_config["analyze_item_task"])
        cfg.pop("agent", None)

        return Task(
            config=cfg,
            agent=self.item_analyst()
        )


    @task
    def predict_review_task(self):
        cfg = dict(self.tasks_config["predict_review_task"])
        cfg.pop("agent", None)

        return Task(
            config=cfg,
            agent=self.prediction_modeler(),
            context=[
                self.analyze_user_task(),
                self.analyze_item_task()
            ],
            output_file="report.json"
        )

    @crew
    def crew(self):

        return Crew(
            agents=[
                self.user_analyst(),
                self.item_analyst(),
                self.prediction_modeler()
            ],

            tasks=[
                self.analyze_user_task(),
                self.analyze_item_task(),
                self.predict_review_task()
            ],

            process=Process.sequential,
            verbose=True
        )
