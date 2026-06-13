from chromadb.config import Settings
import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = _PROJECT_ROOT / "lmdb_cache" / "my_chroma"
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
from src.tools.exact_lookup_tools import lookup_user_by_id, lookup_item_by_id, lookup_reviews_by_user_and_item, none_tool, lowercase_none_tool


import os

from langchain_community.embeddings import HuggingFaceEmbeddings

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
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
    # Use actual path where CrewAI stores ChromaDB (macOS: ~/Library/Application Support/<AppName>)
    db_file = str(Path(db_storage_path()) / "chroma.sqlite3")
    print(f"Check db_file: {db_file}")
    
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
    json_path='dummy_dataset/user.json',
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
    json_path='dummy_dataset/item.json',
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
    json_path='dummy_dataset/review.json',
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
class SimulationCrew():
    """Yelp Recommendation Crew"""
    # Support OpenEvolve overrides via environment variables:
    #   OPENEVOLVE_AGENTS_YAML → override agents config path
    #   OPENEVOLVE_TASKS_YAML  → override tasks config path
    #   OPENEVOLVE_CREW_JSON   → override crew structure (process, order, max_rpm)
    #
    # NOTE: These are set in __init__ (not as class attributes) so that env vars
    # are read at instantiation time, not at import/class-definition time.
    # This is critical for OpenEvolve which sets env vars before each evaluate() call.
    agents_config = str(_PROJECT_ROOT / 'config' / 'agents.yaml')  # default; overridden in __init__
    tasks_config  = str(_PROJECT_ROOT / 'config' / 'tasks.yaml')   # default; overridden in __init__
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, *args, **kwargs):
        # Read env vars HERE (at instantiation), not at class definition time.
        # This ensures OpenEvolve's env var changes take effect for each evaluate() call.
        agents_env = os.environ.get("OPENEVOLVE_AGENTS_YAML", "")
        tasks_env  = os.environ.get("OPENEVOLVE_TASKS_YAML", "")

        if agents_env and os.path.exists(agents_env):
            self.__class__.agents_config = agents_env
        else:
            if agents_env:
                print(f"[SimulationCrew] Warning: OPENEVOLVE_AGENTS_YAML path not found: {agents_env!r}, using default.")
            self.__class__.agents_config = str(_PROJECT_ROOT / 'config' / 'agents.yaml')

        if tasks_env and os.path.exists(tasks_env):
            self.__class__.tasks_config = tasks_env
        else:
            if tasks_env:
                print(f"[SimulationCrew] Warning: OPENEVOLVE_TASKS_YAML path not found: {tasks_env!r}, using default.")
            self.__class__.tasks_config = str(_PROJECT_ROOT / 'config' / 'tasks.yaml')

        # Use explicit super(SimulationCrew, self) instead of zero-argument super().
        # Reason: @CrewBase uses _CrewBaseType.__call__ which creates a BRAND NEW class
        # via CrewBaseMeta(name, bases, dict). The original __classcell__ is consumed
        # during the first (pre-decoration) class creation, so __class__ in the __init__
        # closure still refers to the OLD pre-decoration class object.
        # zero-arg super() checks isinstance(self, __class__) which fails because self
        # is an instance of the NEW decorated class.
        # super(SimulationCrew, self) looks up the global NAME at call-time → decorated class → OK.
        super(SimulationCrew, self).__init__(*args, **kwargs)

    def _crew_cfg(self) -> dict:
        """Load OPENEVOLVE_CREW_JSON if set, else return empty dict (use defaults)."""
        crew_json = os.environ.get("OPENEVOLVE_CREW_JSON", "")
        if crew_json and os.path.exists(crew_json):
            try:
                with open(crew_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[SimulationCrew] Warning: could not load OPENEVOLVE_CREW_JSON: {e}")
        return {}

    def _agent_setting(self, cfg: dict, agent_name: str, key: str, default):
        """Get per-agent setting from crew config, falling back to default."""
        return (
            cfg.get("agent_settings", {})
               .get(agent_name, {})
               .get(key, default)
        )

    # === Step 6: System Assembly & Tool Binding ===
    # Mount specific RAG Tools onto specific Agents
    @agent
    def internet_researcher(self) -> Agent:
        cfg = self._crew_cfg()
        return Agent(
            config=self.agents_config['internet_researcher'],
            tools=[serper_tool],
            verbose=self._agent_setting(cfg, 'internet_researcher', 'verbose', True),
            llm=default_llm,
            max_rpm=self._agent_setting(cfg, 'internet_researcher', 'max_rpm', 15)
        )

    @agent
    def user_analyst(self) -> Agent:
        cfg = self._crew_cfg()
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[lookup_user_by_id, user_rag_tool, none_tool, lowercase_none_tool],
            verbose=self._agent_setting(cfg, 'user_analyst', 'verbose', True),
            llm=default_llm,
            max_rpm=self._agent_setting(cfg, 'user_analyst', 'max_rpm', 10)
        )

    @agent
    def item_analyst(self) -> Agent:
        cfg = self._crew_cfg()
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[lookup_item_by_id, item_rag_tool, none_tool, lowercase_none_tool],
            verbose=self._agent_setting(cfg, 'item_analyst', 'verbose', True),
            llm=default_llm,
            max_rpm=self._agent_setting(cfg, 'item_analyst', 'max_rpm', 10)
        )

    @agent
    def review_analyst(self) -> Agent:
        cfg = self._crew_cfg()
        return Agent(
            config=self.agents_config['review_analyst'], # type: ignore[index]
            tools=[lookup_reviews_by_user_and_item, review_rag_tool, none_tool, lowercase_none_tool],
            verbose=self._agent_setting(cfg, 'review_analyst', 'verbose', True),
            llm=default_llm,
            max_rpm=self._agent_setting(cfg, 'review_analyst', 'max_rpm', 10)
        )

    @agent
    def prediction_modeler(self) -> Agent:
        cfg = self._crew_cfg()
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            verbose=self._agent_setting(cfg, 'prediction_modeler', 'verbose', True),
            llm=default_llm,
            max_rpm=self._agent_setting(cfg, 'prediction_modeler', 'max_rpm', 10)
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'],  # type: ignore[index]
            cache=False,
        )

    @task
    def internet_researcher_task(self) -> Task:
        return Task(
            config=self.tasks_config['internet_research_task'],
            agent=self.internet_researcher(),
            cache=False,
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task'],
            cache=False,
        )

    @task
    def analyze_reviews_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_reviews_task'],
            cache=False,
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'],  # type: ignore[index]
            output_file='report.json',
            cache=False,
        )

    @crew
    def crew(self) -> Crew:
        cfg = self._crew_cfg()

        # ── Agent list (order evolvable via OPENEVOLVE_CREW_JSON) ──────────
        _agent_map = {
            "user_analyst":       self.user_analyst(),
            "item_analyst":       self.item_analyst(),
            "review_analyst":     self.review_analyst(),
            "internet_researcher": self.internet_researcher(),
            "prediction_modeler": self.prediction_modeler(),
        }
        agents_order = cfg.get("agents_order", list(_agent_map.keys()))
        # Only include agents that exist in the map; ignore unknown names
        agents_list = [_agent_map[a] for a in agents_order if a in _agent_map]
        # Append any missing agents at the end (safety)
        for name, obj in _agent_map.items():
            if name not in agents_order:
                agents_list.append(obj)

        # ── Task list (order evolvable via OPENEVOLVE_CREW_JSON) ───────────
        _task_map = {
            "analyze_user_task":       self.analyze_user_task(),
            "analyze_item_task":       self.analyze_item_task(),
            "internet_researcher_task": self.internet_researcher_task(),
            "analyze_reviews_task":    self.analyze_reviews_task(),
            "predict_review_task":     self.predict_review_task(),
        }
        tasks_order = cfg.get("tasks_order", list(_task_map.keys()))
        tasks_list = [_task_map[t] for t in tasks_order if t in _task_map]
        for name, obj in _task_map.items():
            if name not in tasks_order:
                tasks_list.append(obj)

        # ── Process strategy ───────────────────────────────────────────────
        process_str = cfg.get("process", "sequential")
        process = Process.hierarchical if process_str == "hierarchical" else Process.sequential

        return Crew(
            agents=agents_list,
            tasks=tasks_list,
            process=process,
            knowledge_sources=[schema_knowledge],
            embedder=rag_config["embedding_model"],
            memory=False,
            cache=False,
            verbose=True
        )
