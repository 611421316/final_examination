# Environment Setup and Local Model Integration Guide (Milestone 1)

Welcome to the first step of this semester's **AgentSociety Challenge**.

Before we begin building a Multi-Agent system capable of reading and analyzing Yelp reviews, we need to properly configure our development environment and underlying models. During the implementation process, you might encounter issues such as package version conflicts, validation errors, or strict request limits (Rate Limits) from free APIs.

To help you overcome these common initial hurdles, I have prepared this **"Local, Free, and Unlimited" startup guide**. By following these setup steps, you can successfully avoid the most frequent environment configuration errors, allowing you to focus your time on optimizing your Prompts and system architecture.

---

## Step 1: Environment Setup with uv

For this course, I strongly recommend using `uv` to manage packages instead of the traditional `pip install` and `requirements.txt`. `uv` is a fast and popular package management tool widely used in the industry today.

I have prepared the `pyproject.toml` and `uv.lock` files, which have their underlying dependencies perfectly aligned. (You can find these in the `docs/hw_supports/` folder. Please copy both files to your project's root directory).

**🛠️ Execution Steps:**
1. Ensure `uv` is installed on your computer (Mac users can use the command `brew install uv`).
2. Open your terminal, navigate to the project root directory, and enter the following command:
   ```bash
   uv sync
   ```
> [!TIP]
> This command will automatically build a clean, isolated virtual environment and download the stable, tested dependency packages such as `crewai`, `langchain-community`, and `sentence-transformers`, ensuring everyone's execution environment remains consistent.

---

## Step 2: Install the Local Language Model Ollama (Solving the 429 Rate Limit)

When developing inference-driven Agents, the system frequently exchanges information with the language model within short timeframes. If you use free cloud APIs (like Gemini Flash or Groq), it is very easy to exhaust your free request quota within a minute, causing the program to crash with a `429 Rate Limit Error` or `RESOURCE_EXHAUSTED` error.

**To solve this problem, we will switch to a language model that runs entirely locally.**

**🛠️ Execution Steps:**
1. Go to the [Ollama Official Website](https://ollama.com/download) to download and install the application.
2. After launching Ollama, open your terminal and enter the following command to download Microsoft's `Phi-3` model:
   ```bash
   ollama run phi3
   ```
*(Wait for the download to complete. Once the `>>>` prompt appears, you can press `Ctrl+D` to exit. This model will remain on standby in the background server from now on.)*

---

## Step 3: Configure `.env` Environment Variables

CrewAI's underlying packages require proper environment variables to operate smoothly. Please create (or modify) a `.env` file in the root directory of your project with the following content:

```env
# Instruct all Agents to use the newly downloaded local Ollama model
MODEL=ollama/phi3

# [IMPORTANT] Bypass the early Pydantic mechanism in CrewAI-Tools that forcefully checks for an OpenAI Key
# Adding this line prevents the Agent from mistakenly sending requests to OpenAI when reading Knowledge, which would cause a 401 Unauthorized error.
OPENAI_API_KEY=NA

# If you have registered a free Serper account for web searching, you can place your key here (Optional)
SERPER_API_KEY=YourSerperKey
```

---

## Step 4: Correct Configuration and Binding of RAG Retrieval Tools

This system needs to extract intelligence from the Yelp JSON datasets, so we will be using `JSONSearchTool`. Below are two common configuration errors that can cause the system to crash. Please be sure to write your code according to the example provided below:

1. **Strict Package Parameter Validation**: In the latest version of the tools library, passing LangChain objects directly will cause validation failures. Please use a Dictionary to explicitly assign the local Sentence-Transformer for building inference chunks.
2. **Tool Naming Conflicts**: When assigning RAG tools to different Agents, if the tool names (`.name`) are identical, the Function Calling mechanism will fail to differentiate them, triggering a `400 INVALID_ARGUMENT` error.

**🛠️ `crew.py` Configuration Example:**

Below is a top-level implementation reference for `crew.py` that integrates the troubleshooting principles mentioned above. Please ensure your code includes these critical initialization steps:

```python
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai_tools import JSONSearchTool

# 1. Bypass the underlying issue in CrewAI-Tools that forcefully checks for an OpenAI Key
os.environ["OPENAI_API_KEY"] = "NA"

# 2. Configure the global Embedding Model (Used for background retrieval such as CrewAI Knowledge)
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-en-v1.5'
)

# 3. Dedicated configuration file for RAG Tools (Dictionary format)
# Since we set MODEL=ollama/phi3 in .env, there is no need to specify the LLM Provider here; the tool will automatically fallback to using Ollama.
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer", # Specify the use of purely local CPU for automatic chunk generation
        "config": {
            "model_name": "BAAI/bge-small-en-v1.5"
        }
    }
}

# 4. [IMPORTANT] Ensure an independent name (.name) and description (.description) is set for each retrieval tool
user_rag_tool = JSONSearchTool(json_path='data/user_subset.json', collection_name='v3_hf_user_data', config=rag_config)
user_rag_tool.name = "search_user_profile_data"
user_rag_tool.description = "Useful to retrieve a specific user's giving habits, average stars, and review counts."

item_rag_tool = JSONSearchTool(json_path='data/item_subset.json', collection_name='v3_hf_item_data', config=rag_config)
item_rag_tool.name = "search_restaurant_feature_data"
item_rag_tool.description = "Useful to retrieve a specific restaurant's location, categories, attributes, and overall stars."

review_rag_tool = JSONSearchTool(json_path='data/review_subset.json', collection_name='v3_hf_review_data', config=rag_config)
review_rag_tool.name = "search_historical_reviews_data"
review_rag_tool.description = "Useful to retrieve the actual text content of past reviews for users or restaurants."
```

---

After completing the four setup steps above, your foundational environment is successfully constructed.

Please execute the following in your terminal:
👉 **`uv run first_crew`**

During the initial execution, the program will take a few minutes to encode the Yelp datasets and build the local vector database. Once this phase is complete, you will be able to observe the collaboration and decision-making process of the 3 AI Agents on screen, ultimately generating high-quality recommendation prediction reports for you.

Wishing everyone success in their first experiment!
