# 🚀 RAG Vector Index Setup Guide

> **Goal:** Skip the multi-hour vector indexing process and directly use pre-built ChromaDB indexes to run the CrewAI multi-agent system.

---

## 📋 Prerequisites

| Item | Requirement |
|------|-------------|
| Python | 3.12+ |
| Package Manager | [Astral `uv`](https://docs.astral.sh/uv/) |
| OS | macOS / Linux / Windows |
| API Key | Nvidia Build API Key (free to register) |

---

## Step 1: Clone the Project and Install Dependencies

```bash
git clone <repo-url>
cd Rag_Crew_Profiler
uv sync
```

> ⚠️ **Do NOT use `pip install`!** This project exclusively uses `uv` for environment management.

---

## Step 2: Download and Install the Pre-built Index

### 2.1 Obtain the Archive

Download `chroma_index.tar.gz` (~2.3 GB) from the Google Drive link provided by your instructor.

### 2.2 Extract to the Correct Path

CrewAI uses the `appdirs` package to manage storage paths. **The target path differs by operating system:**

**macOS:**
```bash
mkdir -p ~/Library/Application\ Support/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/Library/Application\ Support/Rag_Crew_Profiler/
```

**Linux:**
```bash
mkdir -p ~/.local/share/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/.local/share/Rag_Crew_Profiler/
```

**Windows (PowerShell):**
```powershell
$dest = "$env:LOCALAPPDATA\Rag_Crew_Profiler"
New-Item -ItemType Directory -Force -Path $dest
tar -xzf chroma_index.tar.gz -C $dest
```

### 2.3 Verify

Confirm that the `chroma.sqlite3` file (~4.6 GB) exists at the target path.

**macOS:**
```bash
ls -lh ~/Library/Application\ Support/Rag_Crew_Profiler/chroma.sqlite3
```

**Linux:**
```bash
ls -lh ~/.local/share/Rag_Crew_Profiler/chroma.sqlite3
```

**Windows (PowerShell):**
```powershell
ls "$env:LOCALAPPDATA\Rag_Crew_Profiler\chroma.sqlite3"
```

---

## Step 3: Configure Environment Variables

Create a `.env` file with the following entries and fill in your API keys:

```dotenv
# === LLM Selection (ollama or nvidia) ===
LLM_PROVIDER=nvidia

# -- Local Ollama Settings --
# MODEL=ollama/phi3

# -- NVIDIA API Settings --
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1

GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
COHERE_API_KEY=your_cohere_api_key
```

> 💡 `minimaxai/minimax-m2.7` is currently a free endpoint on Nvidia Build — no payment required.
> 
> Visit [https://build.nvidia.com/](https://build.nvidia.com/) to register and obtain your API Key.

---

## Step 4: Verify the Index Loads Correctly

Run the retrieval benchmark script to confirm the pre-built index is accessible:

```bash
uv run python src/first_crew/benchmark_rag.py
```

**Expected output (normal):**
```text
=== Starting Local RAG Benchmarking (Cached Indexes) ===

Querying Filtered User RAG Tool...
User Tool Retrieval Time: 0.32 seconds

Querying Filtered Item RAG Tool...
Item Tool Retrieval Time: 0.08 seconds

Querying Filtered Review RAG Tool...
Review Tool Retrieval Time: 0.29 seconds

=== Benchmarking Complete ===
```

> ✅ Each query should complete within **1 second** (excluding model initialization time). If any query takes longer than 10 seconds, go back to Step 2 and verify that `chroma.sqlite3` is placed in the correct path.

---

## Step 5: Run the CrewAI Multi-Agent Prediction

```bash
uv run first_crew
```

The system will execute three Agents sequentially:

| Order | Agent | Task |
|-------|-------|------|
| 1 | Yelp User Profiler | Analyze the target user's review habits and preferences |
| 2 | Yelp Restaurant Analyst | Analyze the target restaurant's features and reputation |
| 3 | Review Prediction Expert | Synthesize the above to predict star rating and review text |

**Final results** are written to `report.json`:

```json
{
  "stars": 4.0,
  "review": "The predicted review text..."
}
```

---

## 🔧 Troubleshooting

### Q1: `401 Incorrect API key` error
**Cause:** The `NVIDIA_API_KEY` in `.env` is incorrect or expired.  
**Fix:** Go to [Nvidia Build](https://build.nvidia.com/) and regenerate your API Key.

### Q2: `An instance of Chroma already exists with different settings` error
**Cause:** ChromaDB was initialized multiple times within the same Python process.  
**Fix:** Make sure you are not running multiple scripts simultaneously. Restart your terminal and try again.

### Q3: `benchmark_rag.py` passes, but Agents report "data not found" during `uv run first_crew`
**Cause:** The Agent is using an incorrect tool invocation format.  
**Fix:** Verify that `config/tasks.yaml` contains `search_query` usage examples (already included in this project).

### Q4: I want to build my own index from scratch instead of using the pre-built one
**Command:**
```bash
uv run python src/first_crew/benchmark_indexing.py
```
> ⚠️ This will vectorize the full 211MB JSON dataset. On a CPU-only machine, this takes **1–4 hours**.

---

## 📁 Index Collection Reference

| Collection Name | Source Data | Description |
|-----------------|-----------|-------------|
| `benchmark_true_fresh_index_Filtered_User_1` | `data/filtered_user.json` | User profiles (preferences, average stars, etc.) |
| `benchmark_true_fresh_index_Filtered_Item_1` | `data/filtered_item.json` | Restaurant/business features (categories, location, etc.) |
| `benchmark_true_fresh_index_Filtered_Review_1` | `data/test_review.json` | Full historical review texts |

**Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional vectors, runs on CPU)

> 🔴 **Important:** If you switch to a different embedding model, you MUST rebuild the index! Different models produce vectors with different dimensions, and mixing them will cause completely incorrect retrieval results.

---

## 🏗️ System Architecture Overview

```
main.py                     ← Entry point: reads test data and launches the Crew
  └── crew.py               ← Core orchestration: defines Agent/Task/Tool bindings
        ├── create_rag_tool()   ← Smart cache detector (sqlite3 → instant load or full indexing)
        ├── config/agents.yaml  ← Agent roles, goals, and backstories
        └── config/tasks.yaml   ← Task descriptions and expected output formats
```

How `create_rag_tool()` works:
1. Uses `sqlite3` to directly read `chroma.sqlite3` and check if the target Collection exists
2. **Exists** → Skip indexing, instant load (< 1 second)
3. **Does not exist** → Automatically trigger full index build (takes hours)
