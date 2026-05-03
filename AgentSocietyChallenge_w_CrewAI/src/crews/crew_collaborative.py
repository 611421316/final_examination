import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
load_dotenv()

# Resolve project root (two levels up from src/crews/crew.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import JSONSearchTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
import os

from langchain_community.embeddings import HuggingFaceEmbeddings

with open('docs/eda_knowledge.md', 'r', encoding='utf-8') as f:
    eda_content = f.read()

eda_knowledge = StringKnowledgeSource(
    content=eda_content,
    metadata={"source": "EDA for RAG"}
)

# Workaround for early CrewAI-Tools versions that enforce OpenAI Key validation via Pydantic
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "NA")

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

rag_config = {
    "embedder": {
        "provider": "sentence-transformers",
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


# === Step 3: Configure RAG Tools (CrewAI RAG Tools) ===
def create_rag_tool(json_path: str, collection_name: str, config: dict, name: str, description: str) -> JSONSearchTool:
    from crewai_tools.tools.json_search_tool.json_search_tool import FixedJSONSearchToolSchema
    import sqlite3
    import os
    
    collection_exists = False
    db_file = os.path.join(os.getcwd(), "data", "my_chroma", "chroma.sqlite3")
    
    if os.path.exists(db_file):
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
    if collection_exists:
        tool = JSONSearchTool(collection_name=collection_name, config=config)
        # CRITICAL: Force the Pydantic schema to hide json_path from the Agent, 
        # so it doesn't trigger validation errors or pass the path and trigger the 3-hour hash loop!
        tool.args_schema = FixedJSONSearchToolSchema
    else:
        tool = JSONSearchTool(json_path=json_path, collection_name=collection_name, config=config)
        
    tool.name = name
    tool.description = description
    return tool

user_rag_tool = create_rag_tool(
    json_path='data/filtered_user.json',
    collection_name='benchmark_true_fresh_index_Filtered_User_1',
    config=rag_config,
    name="search_user_profile_data",
    description=(
        "Searches the user profile database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the review habits and average stars for user _BcWyKQL16?'. "
        "Do NOT pass raw user_id or JSON objects directly."
    )
)

item_rag_tool = create_rag_tool(
    json_path='data/filtered_item.json',
    collection_name='benchmark_true_fresh_index_Filtered_Item_1',
    config=rag_config,
    name="search_restaurant_feature_data",
    description=(
        "Searches the restaurant/business database using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'What are the categories, location, and star rating for business abc123?'. "
        "Do NOT pass raw item_id or JSON objects directly."
    )
)

review_rag_tool = create_rag_tool(
    json_path='data/test_review.json',
    collection_name='benchmark_true_fresh_index_Filtered_Review_1',
    config=rag_config,
    name="search_historical_reviews_data",
    description=(
        "Searches historical review texts using semantic similarity. "
        "Input MUST be a natural language search_query string, e.g. "
        "'Find past reviews written by user _BcWyKQL16 about food quality and service'. "
        "Do NOT pass raw user_id, item_id, or JSON objects directly."
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
class CollaborativeCrew():
    """Yelp Recommendation Crew"""
    agents_config = str(_PROJECT_ROOT / 'config' / 'agents.yaml')
    tasks_config  = str(_PROJECT_ROOT / 'config' / 'tasks_collaborative.yaml')

    # === Step 6: System Assembly & Tool Binding ===
    # Mount specific RAG Tools onto specific Agents
    @agent
    def internet_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['internet_researcher'],
            verbose=True,
            llm=default_llm
        )

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            verbose=True,
            llm=default_llm
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            verbose=True,
            llm=default_llm
        )

    @agent
    def reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['reviewer'], # type: ignore[index]
            verbose=True,
            llm=default_llm
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            verbose=True,
            llm=default_llm
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'], # type: ignore[index]
        )

    @task
    def internet_research_task(self) -> Task:
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
        )

    @task
    def review_prediction_task(self) -> Task:
        return Task(
            config=self.tasks_config['review_prediction_task'], # type: ignore[index]
        )

    @task
    def final_prediction_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_prediction_task'], # type: ignore[index]
            output_file='report.json'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.internet_researcher(),
                self.user_analyst(),
                self.item_analyst(),
                self.prediction_modeler(),
                self.reviewer(),
            ],
            tasks=[
                self.internet_research_task(),
                self.analyze_user_task(),
                self.analyze_item_task(),
                self.predict_review_task(),
                self.review_prediction_task(),
                self.final_prediction_task(),
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

