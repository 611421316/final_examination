import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# 根據目錄結構加載自訂的工具層
from src.tools.interaction_tool_wrapper import get_interaction_tool

@CrewBase
class SimulationCrew():
    """Simulation Crew for generating user review simulation"""
    
    # 指向剛才撰寫好的 YAML 配置檔
    agents_config = '../../config/agents.yaml'
    tasks_config = '../../config/tasks.yaml'

    @agent
    def data_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['data_retriever'],
            verbose=False,
            tools=[get_interaction_tool()] # 綁定我們的注入式 Tool wrapper
        )

    @agent
    def psychological_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['psychological_analyst'],
            verbose=False
        )

    @agent
    def behavior_simulator(self) -> Agent:
        return Agent(
            config=self.agents_config['behavior_simulator'],
            verbose=False
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
