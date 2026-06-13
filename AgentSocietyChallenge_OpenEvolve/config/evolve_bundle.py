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
        "user_analyst",
        "item_analyst",
        "review_analyst",
        "internet_researcher",
        "prediction_modeler",
    ],
    "tasks_order": [
        "analyze_user_task",
        "analyze_item_task",
        "internet_researcher_task",
        "analyze_reviews_task",
        "predict_review_task",
    ],
    "process": "sequential",
    "agent_settings": {
        "internet_researcher": {"max_rpm": 15, "verbose": True},
        "user_analyst":        {"max_rpm": 10, "verbose": True},
        "item_analyst":        {"max_rpm": 10, "verbose": True},
        "review_analyst":      {"max_rpm": 10, "verbose": True},
        "prediction_modeler":  {"max_rpm": 10, "verbose": True},
    }
}

# EVOLVE-BLOCK-END
