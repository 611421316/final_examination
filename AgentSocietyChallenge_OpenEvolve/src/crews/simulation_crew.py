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
    def behavior_reasoning_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['behavior_reasoning_agent'],
            verbose=True,
            llm=default_llm,
            max_rpm=15
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'],
            output_file='report.json'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.behavior_reasoning_agent(),
            ],
            tasks=[
                self.predict_review_task(),
            ],
            process=Process.sequential,
            verbose=True
        )
