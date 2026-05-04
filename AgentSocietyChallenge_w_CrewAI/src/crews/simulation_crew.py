import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

# 根據目錄結構加載自訂的工具層
from src.tools.interaction_tool_wrapper import get_interaction_tool

# 明確指定模型，避免落到預設 gpt-4.1-mini（在 NVIDIA endpoint 會 404）
default_llm = LLM(
    model=f"openai/{os.getenv('NVIDIA_MODEL_NAME', 'meta/llama-3.1-8b-instruct')}",
    api_key=os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("NVIDIA_API_BASE")
    or os.getenv("OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1"),
)

@CrewBase
class SimulationCrew():
    """Simulation Crew for generating user review simulation"""
    
    # 指向剛才撰寫好的 YAML 配置檔
    agents_config = '../../config/agents_simulator.yaml'
    tasks_config = '../../config/tasks_simulator.yaml'

    @agent
    def data_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['data_retriever'],
            verbose=False,
            tools=[get_interaction_tool()], # 綁定我們的注入式 Tool wrapper
            llm=default_llm
        )

    @agent
    def psychological_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['psychological_analyst'],
            verbose=False,
            llm=default_llm
        )

    @agent
    def behavior_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config['behavior_simulator'],
            verbose=False,
            llm=default_llm
        )

    @task
    def retrieve_data_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_data_task']
        )

    @task
    def analyze_preference_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_preference_task']
        )

    @task
    def simulate_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['simulate_review_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
