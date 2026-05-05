import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# 根據目錄結構加載自訂的工具層
from src.tools.interaction_tool_wrapper import get_interaction_tool

@CrewBase
class SimulationCrew():
    """Simulation Crew for generating user review simulation"""
    
    # 指向剛才撰寫好的 YAML 配置檔
    agents_config = '../../config/agents_simulator.yaml'
    tasks_config = '../../config/tasks_simulator.yaml'

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'],
            verbose=False,
            tools=[get_interaction_tool()] # 綁定我們的注入式 Tool wrapper
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'],
            verbose=False,
            tools=[get_interaction_tool()] # 綁定我們的注入式 Tool wrapper
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'],
            verbose=False
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task']
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task']
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )