import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Resolve project root (two levels up from src/crews/crew.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from crewai import Agent, Crew, Process, Task, LLM

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

from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from typing import List
import os

from langchain_community.embeddings import HuggingFaceEmbeddings

with open('docs/eda_knowledge.md', 'r', encoding='utf-8') as f:
    eda_content = f.read()

eda_knowledge = StringKnowledgeSource(
    content=eda_content,
    metadata={"source": "EDA for RAG"}
)


# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

rag_config = {
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-small-en-v1.5"
        }
    },
    "vectordb": {
        "provider": "chromadb",
        "config": {
            "dir": "./data/my_chroma"
        }
    }
}


# === Step 3: Configure Tools (Custom File Tools to Bypass Embedchain/OpenAI API) ===
import re
import json

def _extract_id(query: str) -> str:
    match = re.search(r'([a-zA-Z0-9_-]{22})', query)
    return match.group(1) if match else query.split()[-1]

@tool("search_user_profile_data")
def user_rag_tool(search_query: str) -> str:
    """
    Searches the user profile database. Input MUST be a natural language query.
    Extracts the 22-character ID and looks it up in data/filtered_user.json.
    """
    target_id = _extract_id(search_query)
    try:
        with open('data/filtered_user.json', 'r', encoding='utf-8') as f:
            for line in f:
                if target_id in line:
                    return line
    except FileNotFoundError:
        pass
    return f"User {target_id} not found."

@tool("search_restaurant_feature_data")
def item_rag_tool(search_query: str) -> str:
    """
    Searches the restaurant database. Input MUST be a natural language query.
    Extracts the 22-character ID and looks it up in data/filtered_item.json.
    """
    target_id = _extract_id(search_query)
    try:
        with open('data/filtered_item.json', 'r', encoding='utf-8') as f:
            for line in f:
                if target_id in line:
                    return line
    except FileNotFoundError:
        pass
    return f"Item {target_id} not found."

@tool("search_historical_reviews_data")
def review_rag_tool(search_query: str) -> str:
    """
    Searches historical review texts. Input MUST be a natural language query.
    Extracts the 22-character ID and finds up to 5 reviews in data/test_review.json.
    """
    target_id = _extract_id(search_query)
    results = []
    try:
        with open('data/test_review.json', 'r', encoding='utf-8') as f:
            for line in f:
                if target_id in line:
                    results.append(line.strip())
                    if len(results) >= 5:
                        break
    except FileNotFoundError:
        pass
    return "\n".join(results) if results else f"Reviews for {target_id} not found."

# === Step 2: Inject Global Background Knowledge (CrewAI Knowledge) ===
with open('docs/Yelp Data Translation.md', 'r', encoding='utf-8') as f:
    schema_content = f.read()

schema_knowledge = StringKnowledgeSource(
    content=schema_content,
    metadata={"source": "Yelp Schema Definition"}
)

@CrewBase
class SequentialCrew():
    """Yelp Recommendation Crew"""
    agents_config = str(_PROJECT_ROOT / 'config' / 'agents.yaml')
    tasks_config  = str(_PROJECT_ROOT / 'config' / 'tasks_sequential.yaml')
    agents: List[BaseAgent]
    tasks: List[Task]

    # === Step 6: System Assembly & Tool Binding ===
    # Mount specific RAG Tools onto specific Agents
    @agent
    def internet_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['internet_researcher'],
            llm=default_llm,
            verbose=True
        )

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=default_llm,
            verbose=True
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=default_llm,
            verbose=True
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            llm=default_llm,
            verbose=True
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'], # type: ignore[index]
        )

    @task
    def internet_researcher_task(self) -> Task:
        return Task(
            config=self.tasks_config['internet_research_task'],
            agent=self.internet_researcher(),
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task'], # type: ignore[index]
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'], # type: ignore[index]
            output_file='report.json'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.user_analyst(),
                self.item_analyst(),
                self.internet_researcher(),
                self.prediction_modeler(),
            ],
            tasks=[
                self.analyze_user_task(),
                self.analyze_item_task(),
                self.internet_researcher_task(),
                self.predict_review_task(),
        ],
            process=Process.sequential,
            knowledge_sources=[schema_knowledge, eda_knowledge],
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            },
            verbose=True
        )
