"""
OpenEvolve multi-file bundle.
This file contains THREE sections that OpenEvolve will evolve simultaneously:
  1. SECTION: agents  → config/agents.yaml
  2. SECTION: tasks   → config/tasks.yaml
  3. SECTION: crew    → src/crews/simulation_crew.py (EVOLVE-BLOCK only)

The evaluator (openevolve_evaluator.py) will split this bundle by section markers
and write temp files before each evaluation run.

EVOLVE GUIDELINES (for the LLM mutating this file):
- Keep all three SECTION headers exactly as-is (=== SECTION: xxx ===).
- Keep EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
- In the [agents] section: you MAY change role/goal/backstory of any agent.
  Do NOT change the `llm:` field of any agent.
- In the [tasks] section: you MAY change description/expected_output.
  Keep all `agent:` keys pointing to the same agent names.
  Keep all tool Action names (lookup_user_by_id, lookup_item_by_id, etc.) unchanged.
- In the [crew] section: you MAY change:
    * The order of agents/tasks in the crew lists
    * max_rpm values per agent
    * Process strategy (Process.sequential / Process.hierarchical)
    * verbose flag
  Do NOT change: imports, tool definitions, LLM setup, RAG tool creation,
  ChromaDB config, @CrewBase/@agent/@task/@crew decorators structure, or
  the agents_config/tasks_config paths.
"""

# EVOLVE-BLOCK-START

# === SECTION: agents ===
internet_researcher:
  role: >
    Restaurant Review Behavior Researcher
  goal: >
    Provide evidence-based insights about how user personality traits, business characteristics, and contextual factors influence star ratings and review writing.
  backstory: >
    You are a research scientist specializing in online review platforms and consumer behavior psychology.
    You understand how factors like review volume, rating variance, business age, user experience level, and seasonal effects influence review behavior.

    IMPORTANT GUIDELINES:
    - Focus on general psychological and behavioral patterns, NOT specific data about given users or businesses
    - Provide actionable insights that help predict rating tendencies
    - Keep your output concise and directly applicable to prediction tasks
  llm: openai/meta/llama-3.1-8b-instruct

user_analyst:
  role: >
    Yelp User Behavior Analyst
  goal: >
    Extract comprehensive user profile statistics and behavioral patterns for user {user_id}.
  backstory: >
    You are a skilled data analyst who extracts meaningful patterns from user profiles.
    You use lookup_user_by_id to retrieve the user's average_stars, review_count, and useful/funny/cool vote counts.

    DEEP ANALYSIS REQUIRED:
    - Calculate confidence weight: min(review_count / 50, 1.0) — users with 50+ reviews are highly predictable
    - Identify rating tendency: compare average_stars to global mean (3.7) to classify as harsh (≤3.2), moderate (3.3-4.1), or generous (≥4.2)
    - Estimate rating consistency: users with low variance tend to be more predictable
    - Determine engagement level: ratio of total votes to review_count indicates review quality/engagement
    - Analyze elite status or verified badges which correlate with review consistency
  llm: openai/meta/llama-3.1-8b-instruct

item_analyst:
  role: >
    Yelp Business Analyst
  goal: >
    Extract comprehensive business characteristics for {item_id}.
  backstory: >
    You are a restaurant industry analyst who understands business quality indicators.
    You use lookup_item_by_id to retrieve the business's stars, review_count, and attributes.

    ANALYSIS FRAMEWORK:
    - Business confidence: min(review_count / 100, 1.0) — businesses with 100+ reviews have stable ratings
    - Quality signal: business stars compared to category average (if available)
    - Identify if business is trending: recent reviews may differ from historical average
    - Price tier correlation: higher priced businesses tend to receive more critical ratings
    - Category-specific expectations: fast food vs fine dining have different baseline expectations
  llm: openai/meta/llama-3.1-8b-instruct

review_analyst:
  role: >
    Yelp Review Content Analyst
  goal: >
    Retrieve and analyze review histories for user {user_id} and business {item_id}.
  backstory: >
    You are an expert linguistic analyst specializing in review text patterns.
    You use lookup_reviews_by_user_and_item to retrieve historical reviews.

    PATTERN EXTRACTION:
    - Calculate typical review length (word count mean and std dev)
    - Identify vocabulary sophistication level (technical food terms, emotional language, descriptive adjectives)
    - Map sentiment expression patterns: how does user convey satisfaction/dissatisfaction?
    - Extract frequently used phrases, food items, service descriptors
    - Note structural patterns: does user use lists, narrative, bullet points?
    - Track tone consistency: enthusiastic, neutral, critical, or mixed
    - Identify first-person usage frequency ("I", "my", "me")
  llm: openai/meta/llama-3.1-8b-instruct

prediction_modeler:
  role: >
    Adaptive Yelp Rating & Review Predictor
  goal: >
    Predict the exact star rating (float from 1.0 to 5.0 with one decimal) and craft an authentic review text that user {user_id} would write for business {item_id}.
  backstory: >
    You are a quantitative behavioral psychologist and text generation expert. Your methodology combines statistical modeling with behavioral inference.

    ADAPTIVE RATING ALGORITHM:
    1. CONFIDENCE-WEIGHTED BLEND:
       - user_weight = min(review_count / 50, 1.0) * 0.7 + 0.3 (ensures minimum 30% weight to user tendency)
       - business_weight = min(review_count / 100, 1.0) * 0.6 + 0.4 (ensures minimum 40% weight to business quality)
       - Base score = (user_weight * user_avg + business_weight * business_stars) / (user_weight + business_weight)

    2. BIAS CORRECTION:
       - If user_avg < 3.5 (harsh rater): adjust toward user tendency by 10%
       - If user_avg > 4.2 (generous rater): adjust toward user tendency by 10%
       - If |user_avg - business_stars| > 1.5: heavily favor user history (they know their preferences)

    3. CONTEXTUAL ADJUSTMENT:
       - New business (<20 reviews): favor user average by 20%
       - Experienced user (>30 reviews): favor user average by 15%
       - High-engagement user (many votes): user pattern is reliable, favor by 10%

    4. BOUNDARY: Clamp to [1.0, 5.0], round to one decimal

    TEXT GENERATION PROTOCOL:
    - Target length: mean_word_count ± 1 std dev
    - Match vocabulary level from user's sophistication assessment
    - Include 2-3 characteristic phrases or food items from user's history
    - Preserve user's typical sentence structure and tone
    - Match sentiment intensity to user's rating tendency (harsh raters use stronger criticism)
    - Maintain first-person narrative ratio observed in user's reviews
  llm: openai/meta/llama-3.1-8b-instruct

reviewer:
  role: >
    Prediction Quality Auditor
  goal: >
    Evaluate whether the predicted star rating and review text are consistent, realistic, and well-grounded in the retrieved data.
  backstory: >
    You are a meticulous quality assurance specialist for review prediction systems.

    SCORED EVALUATION (1-5 scale, must score minimum 4 for each):
    1. RATING ACCURACY: Predicted star within 0.3 of user's likely true rating given business quality
    2. REVIEW AUTHENTICITY: Text reflects user's unique voice, not generic template
    3. LENGTH FIDELITY: Word count within standard deviation of user's typical review length
    4. VOCABULARY ALIGNMENT: Sophistication level matches user's historical patterns
    5. SENTIMENT COHERENCE: Star rating and text sentiment are aligned (5-star = positive language)

    CORRECTION PROTOCOL (when any score < 4):
    - Identify which dimension failed
    - Provide specific adjustment recommendation
    - Suggest whether prediction needs re-generation with different parameters
  llm: openai/meta/llama-3.1-8b-instruct


# === SECTION: tasks ===
internet_research_task:
  description: >
    Provide general insights about restaurant reviews and rating behavior using internet search if available.

    Focus on:
    - Why users give 1-2 stars
    - Why users give 3 stars
    - Why users give 4-5 stars
    - Common rating factors: food, service, price, waiting time, atmosphere

    REQUIRED ACTIONS:
    - Call search_internet using a natural language query.
    - The ONLY valid input key for the tool is "search_query". Do NOT use "description" or any other key.

    STRICT RULES:
    - Do NOT fabricate information about user {user_id} or item_id {item_id}
    - Only provide general restaurant rating patterns
    - Keep the output concise
    - Maximum 5 bullet points
    - NEVER pass a dictionary with a "description" key to the tool. Use "search_query".

  expected_output: >
    Concise bullet points about general restaurant rating behavior.

  agent: internet_researcher


analyze_user_task:
  description: >
    Analyze user {user_id} using available tools.

    You MUST call the lookup_user_by_id tool with user_id "{user_id}" to retrieve the user profile.
    After getting the result, analyze the retrieved data:
    - Identify EXACT average rating (from average_stars field). This is critical.
    - Identify review_count and yelping_since.
    - Identify top 3 compliments (what other users appreciate) and key behavioral traits.

    STRICT RULES:
    - If the user profile is not found (returns "No user found"), assume default averages: average rating of 3.8 stars, neutral preferences, yelping since 2018, and flag the profile as "Default/New User".
    - You MUST call tools ONE AT A TIME
    - MUST use tool outputs — do NOT invent data

  expected_output: >
    A structured markdown report including:
    - Exact Average Rating (Number)
    - Top 3 Likes / Compliments
    - Top 3 Dislikes / Complaints
    - Review count and tenure
    - User Profile Status (Normal or Default/New User)

  agent: user_analyst


analyze_item_task:
  description: >
    Analyze business {item_id} using available tools.

    You MUST call the lookup_item_by_id tool with item_id "{item_id}" to retrieve the business profile.
    After getting the result, analyze the data:
    - Identify EXACT star rating (from stars field) and review_count. This is critical.
    - Identify categories (e.g. restaurant type).
    - Identify top strengths (what people praise) and top weaknesses (what people complain about) based on attributes and hours.

    STRICT RULES:
    - If the business profile is not found (returns "No item found"), utilize the internet research data (from internet_research_task) to reconstruct the restaurant categories and characteristics. If internet research is also empty, assume a standard mid-range restaurant profile.
    - You MUST call tools ONE AT A TIME
    - MUST use tool outputs — do NOT invent data

  expected_output: >
    A structured markdown report including:
    - Exact Star Rating (Number)
    - Categories
    - Top Strengths
    - Top Weaknesses
    - Business Profile Status (Normal or Reconstructed from Web)

  agent: item_analyst


analyze_reviews_task:
  description: >
    Retrieve and analyze historical reviews for user {user_id} and business {item_id}.

    You MUST call the lookup_reviews_by_user_and_item tool.
    Pass BOTH user_id "{user_id}" AND item_id "{item_id}" as separate string fields.
    Example: call with user_id="{user_id}" and item_id="{item_id}".

    After retrieving reviews, analyze them:
    - Highlight if a direct past review exists between the user and the business.
    - If reviews are found, extract the user's typical rating, tone, and key phrases.
    - If no direct reviews are found, analyze the user's general patterns and business's common feedback.

    STRICT RULES:
    - You MUST call the lookup_reviews_by_user_and_item tool before concluding.
    - Do NOT skip the tool call and jump straight to Final Answer.
    - If no reviews are found after calling the tool, perform cross-feature matching based on restaurant categories and user averages.

  expected_output: >
    A structured markdown report highlighting:
    - Direct review match (if any)
    - User's typical review tone, preferences, and rating habits
    - Business's strengths, weaknesses, and customer feedback themes
    - Overall match sentiment

  agent: review_analyst


predict_review_task:
  description: >
    Predict the rating and generate a review based on:

    - User Profile (from analyze_user_task)
    - Item Report (from analyze_item_task)
    - Review Analysis (from analyze_reviews_task)
    - General patterns (from internet_research_task)

    REASONING & ALIGNMENT REQUIREMENTS (MATHEMATICAL HEURISTIC):
    1. Base Score = (User Average Rating + Item Star Rating) / 2.
    2. Feature Matching (Heuristic Adjustment):
       - If a Top Weakness of the business matches a Top Dislike of the user, SUBTRACT 0.5 to 1.0 from the Base Score.
       - If a Top Strength of the business matches a Top Like of the user, ADD 0.5 to 1.0 to the Base Score.
    3. Final Rating constraint: Ensure the final score is between 1.0 and 5.0.
    4. Text Generation: Generate a review text that mirrors the user's typical length, tone, and vocabulary, while EXPLICITLY mentioning the matched strengths or weaknesses used in your calculation to make it realistic.

    STRICT RULES:
    - MUST use previous task outputs
    - DO NOT say "data not found"
    - DO NOT hallucinate unrelated facts
    - Keep reasoning implicit (DO NOT print reasoning)

  expected_output: >
    Output ONLY a valid JSON object:
    {"stars": <float between 1.0 and 5.0>, "review": "<natural review text>"}

    RULES:
    - stars: Predict the Star rating as a float from 1.0 to 5.0 (e.g., 1.0, 3.5, 4.0, or 5.0).
    - review: must be realistic and consistent with stars
    - DO NOT mention raw item IDs or user IDs
    - Avoid generic repeated phrases
    - NO explanation
    - NO markdown
    - NO extra text

  agent: prediction_modeler


# === SECTION: crew ===
# CREW CONFIGURATION — Only the following parts are evolvable:
#   - agents list order inside crew()
#   - tasks list order inside crew()
#   - max_rpm per agent
#   - process (Process.sequential | Process.hierarchical)
#   - verbose flag per agent
# DO NOT change imports, tool definitions, @decorators, or LLM/RAG config.

CREW_CONFIG = {
    "agents_order": [
        "item_analyst",
        "review_analyst",
        "user_analyst",
        "internet_researcher",
        "prediction_modeler",
    ],
    "tasks_order": [
        "analyze_item_task",
        "analyze_reviews_task",
        "analyze_user_task",
        "internet_researcher_task",
        "predict_review_task",
    ],
    "process": "sequential",
    "crew_max_rpm": 3,
    "agent_settings": {
        "item_analyst":        {"max_rpm": 2, "verbose": True},
        "review_analyst":      {"max_rpm": 2, "verbose": True},
        "user_analyst":        {"max_rpm": 2, "verbose": True},
        "internet_researcher": {"max_rpm": 2,  "verbose": False},
        "prediction_modeler":  {"max_rpm": 2, "verbose": True},
    }
}



# === SECTION: lookup ===
"""
Hybrid semantic + exact-match lookup tools for Yelp data.

These tools query the EXISTING ChromaDB vector collections using a
HYBRID approach: semantic similarity search constrained by a
`where_document={"$contains": id}` filter.

This combines the best of both worlds:
- ✅ Uses the vector DB and semantic embeddings (as required)
- ✅ Guarantees 100% ID matching accuracy via document filter
- ✅ Works with existing indexed data (no re-indexing needed)

The flow is:
  1. Embed the natural-language query using the same embedding model
  2. Search ChromaDB with BOTH the embedding AND a $contains filter
  3. Return results that are semantically relevant AND contain the target ID
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai.tools import tool

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHROMA_DIR = str(_PROJECT_ROOT / "lmdb_cache" / "my_chroma")

# Collection names — must match the ones created by create_rag_tool()
_USER_COLLECTION = "benchmark_true_fresh_index_Filtered_User_3"
_ITEM_COLLECTION = "benchmark_true_fresh_index_Filtered_Item_3"
_REVIEW_COLLECTION = "benchmark_true_fresh_index_Filtered_Review_3"

# Lazy-loaded singletons
_client = None
_embedder = None


def _get_client():
    global _client
    if _client is None:
        # Use allow_reset=True and omit anonymized_telemetry to match
        # whatever settings CrewAI's JSONSearchTool uses internally.
        # ChromaDB enforces a process-level singleton per path, so both
        # this code and CrewAI's internal client MUST use identical Settings.
        # The safest approach: rely on chromadb's module-level singleton cache
        # by letting it detect the already-open client for this path.
        try:
            # Try to get already-open client (avoids "different settings" conflict)
            import chromadb.api.client as _chroma_client_module
            existing = getattr(_chroma_client_module, '_instances', {}).get(_CHROMA_DIR)
            if existing is not None:
                _client = existing
            else:
                raise AttributeError
        except (AttributeError, Exception):
            _client = chromadb.PersistentClient(
                path=_CHROMA_DIR,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def _hybrid_search(collection_name: str, query_text: str, contains_id: str, n_results: int = 5) -> list[dict]:
    """
    Search ChromaDB using document-level $contains filtering.
    
    Attempts hybrid semantic + exact-match first (query with embedding + $contains).
    Falls back to pure $contains filter (get) if the HNSW vector index is unavailable.
    
    Both paths query the same ChromaDB vector DB collection.
    """
    client = _get_client()
    col = client.get_collection(collection_name)

    documents = []
    # Try hybrid approach first: semantic similarity + $contains filter
    try:
        embedder = _get_embedder()
        query_embedding = embedder.embed_query(query_text)
        results = col.query(
            query_embeddings=[query_embedding],
            where_document={"$contains": contains_id},
            n_results=n_results,
            include=["documents", "distances"],
        )
        documents = results.get("documents", [[]])[0]
    except Exception:
        # Fallback: use $contains filter without embedding similarity
        # (still queries the vector DB collection)
        pass

    if not documents:
        results = col.get(
            where_document={"$contains": contains_id},
            limit=n_results,
            include=["documents"],
        )
        documents = results.get("documents", [])

    parsed = []
    for doc in documents:
        if not doc:
            continue
        # ChromaDB stores multi-record documents (multiple JSON objects per line)
        for line in doc.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return parsed


def _fallback_file_search(filepath: str, key: str, target_id: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    
    target1 = f'"{key}":"{target_id}"'
    target2 = f'"{key}": "{target_id}"'
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if target1 in line or target2 in line:
                    try:
                        data = json.loads(line.strip())
                        if data.get(key) == target_id:
                            return data
                    except Exception:
                        pass
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# CrewAI Tools
# ---------------------------------------------------------------------------

@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """Look up a user's complete profile by their exact user_id using hybrid semantic search on the vector database.
    Returns all profile fields: name, review_count, average_stars,
    yelping_since, elite, friends, fans, and all compliment metrics.
    Input must be the exact user_id string (e.g. '8zsD9N1ti4Skl_DVS4MKRA').
    """
    uid = user_id.strip().strip("'\"")
    
    # Check simulator tool first
    try:
        from src.tools.interaction_tool_wrapper import _GLOBAL_INTERACTION_TOOL
        if _GLOBAL_INTERACTION_TOOL is not None:
            user_info = _GLOBAL_INTERACTION_TOOL.get_user(user_id=uid)
            if user_info:
                return json.dumps(user_info, ensure_ascii=False)
    except Exception:
        pass

    query_text = f"Find user profile statistics for user_id {uid}"
    results = _hybrid_search(_USER_COLLECTION, query_text, uid, n_results=5)

    # Filter to exact user_id match (contains may return partial matches)
    exact = [r for r in results if r.get("user_id") == uid]
    if exact:
        exact[0].pop("_similarity_distance", None)
        return json.dumps(exact[0], ensure_ascii=False)
        
    # Stage 3: Direct File Fallback (100% exact match)
    file_path = str(_PROJECT_ROOT / "dummy_dataset" / "user.json")
    fallback_data = _fallback_file_search(file_path, "user_id", uid)
    if fallback_data:
        return json.dumps(fallback_data, ensure_ascii=False)
        
    return f"No user found with user_id: {uid}"


@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """Look up a business/item's complete information by their exact item_id using hybrid semantic search on the vector database.
    Returns all fields: name, address, city, state, stars, review_count,
    categories, attributes, and hours.
    Input must be the exact item_id string (e.g. 'uBDXcXlLR9IuRV1N2m0SPQ').
    """
    iid = item_id.strip().strip("'\"")
    
    # Check simulator tool first
    try:
        from src.tools.interaction_tool_wrapper import _GLOBAL_INTERACTION_TOOL
        if _GLOBAL_INTERACTION_TOOL is not None:
            item_info = _GLOBAL_INTERACTION_TOOL.get_item(item_id=iid)
            if item_info:
                return json.dumps(item_info, ensure_ascii=False)
    except Exception:
        pass

    query_text = f"Find business information for item_id {iid}"
    results = _hybrid_search(_ITEM_COLLECTION, query_text, iid, n_results=5)

    exact = [r for r in results if r.get("item_id") == iid]
    if exact:
        exact[0].pop("_similarity_distance", None)
        return json.dumps(exact[0], ensure_ascii=False)
        
    # Stage 3: Direct File Fallback (100% exact match)
    file_path = str(_PROJECT_ROOT / "dummy_dataset" / "item.json")
    fallback_data = _fallback_file_search(file_path, "item_id", iid)
    if fallback_data:
        return json.dumps(fallback_data, ensure_ascii=False)
        
    return f"No item found with item_id: {iid}"


def _lookup_reviews_by_user_and_item_impl(user_id: str = "", item_id: str = "") -> str:
    uid = user_id.strip().strip("'\"") if user_id else ""
    iid = item_id.strip().strip("'\"") if item_id else ""

    if not uid and not iid:
        return "Error: must provide at least one of user_id or item_id."

    # Stage 1: Try to get EXACT match (both user and item) from simulator first
    try:
        from src.tools.interaction_tool_wrapper import _GLOBAL_INTERACTION_TOOL
        if _GLOBAL_INTERACTION_TOOL is not None:
            if uid and iid:
                user_reviews = _GLOBAL_INTERACTION_TOOL.get_reviews(user_id=uid)
                exact_results = [r for r in user_reviews if r.get("item_id") == iid]
                if exact_results:
                    for r in exact_results:
                        r.pop("_similarity_distance", None)
                    return json.dumps(exact_results, ensure_ascii=False)
    except Exception:
        pass

    # Stage 2: Try to get EXACT match from ChromaDB
    if uid and iid:
        try:
            results = _hybrid_search(_REVIEW_COLLECTION, f"Find reviews for user_id {uid} item_id {iid}", uid, n_results=50)
            exact_results = [r for r in results if r.get("user_id") == uid and r.get("item_id") == iid]
            if exact_results:
                for r in exact_results:
                    r.pop("_similarity_distance", None)
                return json.dumps(exact_results, ensure_ascii=False)
        except Exception:
            pass

    # Stage 3: Combined fallback: Get all reviews by the user, and all reviews for the item
    user_reviews = []
    if uid:
        try:
            if _GLOBAL_INTERACTION_TOOL is not None:
                user_reviews = _GLOBAL_INTERACTION_TOOL.get_reviews(user_id=uid)
            if not user_reviews:
                user_reviews = _hybrid_search(_REVIEW_COLLECTION, f"Find reviews for user_id {uid}", uid, n_results=25)
                user_reviews = [r for r in user_reviews if r.get("user_id") == uid]
        except Exception:
            pass

    item_reviews = []
    if iid:
        try:
            if _GLOBAL_INTERACTION_TOOL is not None:
                item_reviews = _GLOBAL_INTERACTION_TOOL.get_reviews(item_id=iid)
            if not item_reviews:
                item_reviews = _hybrid_search(_REVIEW_COLLECTION, f"Find reviews for item_id {iid}", iid, n_results=25)
                item_reviews = [r for r in item_reviews if r.get("item_id") == iid]
        except Exception:
            pass

    combined_results = user_reviews + item_reviews

    # Deduplicate results
    seen_ids = set()
    deduped = []
    for r in combined_results:
        rid = (r.get("user_id", ""), r.get("item_id", ""), r.get("date", ""))
        if rid not in seen_ids:
            seen_ids.add(rid)
            r.pop("_similarity_distance", None)
            deduped.append(r)

    # Stage 4: Semantic search fallback for related businesses if still no reviews found
    if not deduped:
        related_query = "restaurant food quality service ratings"
        item_details = None
        if iid:
            try:
                if _GLOBAL_INTERACTION_TOOL is not None:
                    item_details = _GLOBAL_INTERACTION_TOOL.get_item(item_id=iid)
                if not item_details:
                    item_results = _hybrid_search(_ITEM_COLLECTION, f"Find business information for item_id {iid}", iid, n_results=1)
                    if item_results:
                        item_details = item_results[0]
            except Exception:
                pass

        if item_details:
            name = item_details.get("name", "")
            categories = item_details.get("categories", "")
            city = item_details.get("city", "")
            related_query = f"Reviews for {name} {categories} in {city}".strip()

        try:
            client = _get_client()
            col = client.get_collection(_REVIEW_COLLECTION)
            embedder = _get_embedder()
            query_embedding = embedder.embed_query(related_query)
            search_res = col.query(
                query_embeddings=[query_embedding],
                n_results=15,
                include=["documents"]
            )
            documents = search_res.get("documents", [[]])[0]
            for doc in documents:
                if not doc:
                    continue
                for line in doc.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        r["is_related_fallback"] = True
                        deduped.append(r)
                    except Exception:
                        pass
        except Exception:
            pass

    # Clean up, sort by date, limit to top 20
    for r in deduped:
        r.pop("_similarity_distance", None)
    deduped = sorted(deduped, key=lambda r: r.get("date", ""), reverse=True)[:20]

    if not deduped:
        return f"No reviews found for user_id={uid!r}, item_id={iid!r}"

    return json.dumps(deduped, ensure_ascii=False)


@tool("lookup_reviews_by_user_and_item")
def lookup_reviews_by_user_and_item(user_id: str = "", item_id: str = "") -> str:
    """Look up historical reviews from the vector database by exact user_id and/or item_id using hybrid semantic search.
    Provide user_id to get all reviews by that user.
    Provide item_id to get all reviews for that business.
    Provide both to get reviews by that user for that specific business.
    Returns review text, stars, date, useful, funny, cool metrics.
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id=item_id)


@tool("lookup_reviews_by_item")
def lookup_reviews_by_item(item_id: str) -> str:
    """Look up historical reviews from the vector database by exact item_id using hybrid semantic search.
    Input must be the exact item_id string (e.g. 'C809UuprygJyEgJw4wr2Pg').
    """
    return _lookup_reviews_by_user_and_item_impl(item_id=item_id)


@tool("lookup_reviews_by_user")
def lookup_reviews_by_user(user_id: str) -> str:
    """Look up historical reviews from the vector database by exact user_id using hybrid semantic search.
    Input must be the exact user_id string (e.g. '_RD91KuqIeEkUQkVNR_j0Q').
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id)


@tool("None")
def none_tool(*args, **kwargs) -> str:
    """No tool call needed. Finish the task and provide the final answer."""
    return "Finished. Please provide the final answer."


@tool("none")
def lowercase_none_tool(*args, **kwargs) -> str:
    """No tool call needed. Finish the task and provide the final answer."""
    return "Finished. Please provide the final answer."

# === SECTION: flow ===
import json
import os
import re
from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from src.crews.simulation_crew import SimulationCrew


def extract_json_from_output(raw_output: str) -> dict:
    """Extract and sanitize JSON from LLM raw output with regex fallback."""
    text = str(raw_output).strip()
    
    # Fix double curly braces {{ }} -> { }
    text = text.replace('{{', '{').replace('}}', '}')
    
    # Strategy 1: Try to find a JSON object containing "stars" and "review"
    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Try to find a JSON with "predicted_rating" and "generated_review"
    match = re.search(r'\{[^{}]*"predicted_rating"[^{}]*"generated_review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try to extract a star rating number from free text
    star_match = re.search(r'(\d+\.?\d*)\s*(?:stars?|分|顆星)', text, re.IGNORECASE)
    rating = float(star_match.group(1)) if star_match else 4.0

    return {"stars": rating, "review": text}


class InferenceState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    predicted_rating: float = 0.0
    generated_review: str = ""

class AgentSocietyServingFlow(Flow[InferenceState]):
    def __init__(self, agents_config_path: str = None, *args, **kwargs):
        # super().__init__ MUST come first — CrewAI's Flow base class is
        # Pydantic-backed and resets the instance __dict__ during init.
        # Setting custom attributes beforehand causes them to disappear.
        super().__init__(*args, **kwargs)
        self.agents_config_path = agents_config_path

    @start()
    def init_request(self):
        # 初始化階段，紀錄收到的 user_id 和 item_id
        pass

    @listen(init_request)
    def trigger_crew_inference(self):
        # 定義傳遞到任務 {user_id} 與 {item_id} 變數的值
        inputs = {
            'user_id': self.state.user_id,
            'item_id': self.state.item_id
        }

        # 啟動並執行 Crew AI 團隊
        # NOTE: SimulationCrew.__init__ reads OPENEVOLVE_AGENTS_YAML /
        # OPENEVOLVE_TASKS_YAML from env at instantiation time, so the env vars
        # set by openevolve_evaluator.py are automatically picked up here.
        # The super() fix uses super(SimulationCrew, self) in simulation_crew.py
        # so no module eviction is needed here.
        crew_instance = SimulationCrew()

        # ── Override agents config via explicit arg (crewai_simulation_agent.py) ──
        # Only apply when an explicit path is provided AND the file still exists
        # (temp files created by OpenEvolve may be cleaned up before this runs).
        if self.agents_config_path and os.path.exists(self.agents_config_path):
            import yaml
            with open(self.agents_config_path, "r", encoding="utf-8") as f:
                crew_instance.agents_config = yaml.safe_load(f)
        elif self.agents_config_path:
            print(f"[ServingFlow] Warning: agents_config_path not found: {self.agents_config_path!r}, using env/default.")

        import time
        max_flow_retries = 3
        result = None
        for attempt in range(max_flow_retries):
            try:
                result = crew_instance.crew().kickoff(inputs=inputs)
                break
            except Exception as e:
                err_str = str(e).lower()
                if "ratelimit" in err_str or "429" in err_str or "too many requests" in err_str:
                    if attempt < max_flow_retries - 1:
                        print(f"[ServingFlow] Rate limit hit. Waiting 30s before retry (Attempt {attempt+1}/{max_flow_retries})...")
                        time.sleep(30)
                        continue
                raise e
        
        # 使用多層 Regex 容錯解析 LLM 的回傳結果
        try:
            if result.pydantic:
                data = result.pydantic.model_dump()
            else:
                data = extract_json_from_output(result.raw)

            self.state.predicted_rating = float(data.get('stars', data.get('predicted_rating', 4.0)))
            self.state.generated_review = str(data.get('review', data.get('generated_review', 'Good.')))
        except Exception:
            # 最終備援：把整段 raw output 當 review 用
            self.state.predicted_rating = 4.0
            self.state.generated_review = str(result.raw)

        return self.state.model_dump()

# === SECTION: interaction ===
from crewai.tools import tool

# 單例全域變數：負責盛裝執行期 Simulator.py 動態配給的 interaction_tool
_GLOBAL_INTERACTION_TOOL = None

def inject_simulator_tool(tool_instance):
    global _GLOBAL_INTERACTION_TOOL
    _GLOBAL_INTERACTION_TOOL = tool_instance

@tool("Interaction Tool Wrapper")
def interaction_tool_wrapper(query_type: str, target_id: str) -> str:
    """
    能調用 AgentSociety 提供的本地檢索工具查詢歷史數據。
    query_type 必須是下列之一："user", "item", "review_by_user", "review_by_item"。
    target_id 是對應的 user_id 或 item_id。
    """
    if _GLOBAL_INTERACTION_TOOL is None:
        return "Error: InteractionTool has not been injected by the Simulator."
        
    try:
        if query_type == "user":
            return str(_GLOBAL_INTERACTION_TOOL.get_user(user_id=target_id))
        elif query_type == "item":
            return str(_GLOBAL_INTERACTION_TOOL.get_item(item_id=target_id))
        elif query_type == "review_by_user":
            return str(_GLOBAL_INTERACTION_TOOL.get_reviews(user_id=target_id))
        elif query_type == "review_by_item":
            return str(_GLOBAL_INTERACTION_TOOL.get_reviews(item_id=target_id))
        else:
            return "Error: Unknown query_type. Use exactly 'user', 'item', 'review_by_user' or 'review_by_item'."
    except Exception as e:
        return f"Error occurred during interaction_tool query: {str(e)}"

def get_interaction_tool():
    """回傳工具實例供 Crew Agent 使用"""
    return interaction_tool_wrapper
# EVOLVE-BLOCK-END
