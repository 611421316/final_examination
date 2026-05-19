from chromadb.config import Settings
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = _PROJECT_ROOT / "data" / "my_chroma"
os.environ["CREWAI_STORAGE_DIR"] = str(CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)
from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import JSONSearchTool, SerperDevTool

serper_tool = SerperDevTool(
    name="search_internet",
    description=(
        "Search the internet for general restaurant review trends, Yelp rating behavior, "
        "customer satisfaction factors, and public background information."
    )
)
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

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NVIDIA_API_BASE_ROOT = "https://integrate.api.nvidia.com"
NVIDIA_API_BASE_V1 = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL_NAME = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Keep OPENAI_API_KEY set so Pydantic validation in crewai_tools doesn't crash.
# Do NOT set OPENAI_API_BASE — that would redirect Embedchain's local
# sentence-transformer embedding calls to Nvidia (which returns 404).
# The default_llm object already carries the Nvidia base_url for actual LLM calls.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# === Step 3: Configure RAG Tools (CrewAI RAG Tools) ===
def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai.utilities.paths import db_storage_path
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = "data/my_chroma/chroma.sqlite3"
    print("Check db_file")
    
    if os.path.exists(db_file):
        print("db_file exists")
        try:
            # Check native sqlite3 for existing collection to heavily avoid 100% JSON text synchronous chunking bottleneck
            # and avoid ChromaDB singleton initialization conflicts with CrewAI's internal Settings
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM collections WHERE name = ?", (collection_name,))
            if cursor.fetchone() is not None:
                collection_exists = True
            conn.close()
        except Exception:
            pass
    print("[DEBUG] FINAL COLLECTION CHECK")
    print(f"Collection Exists: {collection_exists}")
    if collection_exists:
        print(f"Tool {collection_name} exists")
        print(f"[DEBUG] Using collection: {collection_name}")
        print(f"[DEBUG] JSON source: {json_path}")
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        print(f"Tool {collection_name} created")
        print(f"[DEBUG] Using collection: {collection_name}")
        print(f"[DEBUG] JSON source: {json_path}")
    
    tool.name = name
    tool.description = description
    return tool

user_rag_tool = create_rag_tool(
    json_path='data/filtered_user.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_3',
    config=rag_config,
    name="search_user_profile_data",
    description=(
    "Search user profile information and statistics using semantic similarity. "
    "This tool can retrieve a user's name, review_count, average_stars, yelping_since, "
    "elite, friends, fans, and compliment metrics. "

    "Input MUST be a natural language search_query string. For specific users, ALWAYS include the exact 'user_id' inside the query. "

    "Example queries:\n"
    "- 'Find profile statistics and average_stars where user_id is _BcWyKQL16ndpBdggh2kNA'\n"
    "- 'What is the review_count and elite status where user_id is XgE3E2Sm-nhtTS_9PjtJsQ?'\n"
    
    "Do NOT pass raw user_id or JSON objects directly as the query."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/filtered_item.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_3',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
    "Search general business and item information using item_id or natural language queries. "
    "This tool can retrieve business categories, stars, city, state, hours, review_count, and attributes. "

    "For specific businesses, ALWAYS include the exact 'item_id' inside the search_query. "

    "Example queries:\n"
    "- 'Find business information where item_id is uBDXcXlLR9IuRV1N2m0SPQ'\n"
    "- 'What are the categories, hours, and stars where item_id is -JIeZE7f926mnRNcdnYk6Q?'\n"
    "- 'Find highly rated sushi restaurants or auto repair shops'\n"
    
    "NEVER pass raw JSON objects."
)

)

review_rag_tool = create_rag_tool(
    json_path='data/train_review.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_3',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
    "Search historical review data and texts using semantic similarity. "
    "This tool retrieves detailed review information including the text, stars, date, "
    "useful, funny, and cool metrics for specific users or items/businesses. "

    "Input MUST be a natural language search_query string. "
    "To find reviews for a specific user or business, ALWAYS include the exact 'user_id' or 'item_id' inside the query. "

    "Example queries:\n"
    "- 'Find past reviews where user_id is _BcWyKQL16ndpBdggh2kNA about food quality and service'\n"
    "- 'What do the 5-star reviews say where item_id is uBDXcXlLR9IuRV1N2m0SPQ?'\n"
    "- 'Search for negative text mentioning bad service where item_id is 9zlIJ7Q5W4AENjpGgaNSsQ'\n"
    
    "Do NOT pass raw user_id, item_id, or JSON objects directly as the query."
)

)
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
            tools=[serper_tool],
            verbose=True,
            llm=default_llm,
            max_rpm=5
        )

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            verbose=True,
            llm=default_llm,
            max_rpm=5
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            verbose=True,
            llm=default_llm,
            max_rpm=5
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            verbose=True,
            llm=default_llm,
            max_rpm=5
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool]
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
            config=self.tasks_config['analyze_item_task'], 
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
            knowledge_sources=[schema_knowledge],
            embedder=rag_config["embedding_model"],
            verbose=True
        )
