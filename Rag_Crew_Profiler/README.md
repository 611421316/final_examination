# 🔍 Rag Crew Profiler

A **CrewAI multi-agent system** that predicts Yelp review ratings and generates review text using RAG (Retrieval-Augmented Generation) over real Yelp datasets.

Built for the [AgentSociety Challenge](https://www.agentsocietychallenge.com/) — Track 1: Recommendation.

---

## ✨ Key Features

- **3-Agent Pipeline**: User Profiler → Restaurant Analyst → Review Prediction Expert
- **Smart Index Caching**: Detects existing ChromaDB collections via `sqlite3` to bypass CPU-intensive re-indexing (hours → < 1 second)
- **Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` (384-dim) — runs entirely on CPU, no GPU required
- **Free LLM Endpoint**: Configured for `minimaxai/minimax-m2.7` via Nvidia Build API (free tier)
- **YAML-First Architecture**: Agent roles, goals, and task descriptions are fully separated from Python code

---

## 🏗️ Architecture

```
main.py                         ← Entry point: reads test data, launches Crew, writes report.json
  └── crew.py                   ← Core orchestration: Agent/Task/Tool bindings
        ├── create_rag_tool()   ← Smart cache detector (sqlite3 check → instant load or full index)
        ├── config/agents.yaml  ← Agent roles, goals, and backstories
        └── config/tasks.yaml   ← Task descriptions and expected output formats
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **User Profiler** | Analyze user's review history and preferences | `search_user_profile_data`, `search_historical_reviews_data` |
| **Restaurant Analyst** | Analyze business features and reputation | `search_restaurant_feature_data`, `search_historical_reviews_data` |
| **Prediction Expert** | Synthesize profiles to predict stars & review text | *(uses context from previous agents)* |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Astral `uv`](https://docs.astral.sh/uv/)
- [Nvidia Build API Key](https://build.nvidia.com/) (free)

### 1. Clone & Install

```bash
git clone https://github.com/yuchieh/Rag_Crew_Profiler.git
cd Rag_Crew_Profiler
uv sync
```

> ⚠️ This project uses `uv` exclusively. Do **not** use `pip install`.

### 2. Configure Environment

Create a `.env` file in the project root:

```dotenv
LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-your-key-here
NVIDIA_MODEL_NAME=minimaxai/minimax-m2.7
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
```

### 3. Set Up Vector Index

**Option A — Use pre-built index** (recommended, < 1 min):

Download `chroma_index.tar.gz` from the provided Google Drive link and extract:

```bash
# macOS
mkdir -p ~/Library/Application\ Support/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/Library/Application\ Support/Rag_Crew_Profiler/

# Linux
mkdir -p ~/.local/share/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/.local/share/Rag_Crew_Profiler/
```

**Option B — Build from scratch** (1–4 hours on CPU):

```bash
uv run python src/first_crew/benchmark_indexing.py
```

### 4. Run

```bash
uv run first_crew
```

Results are written to `report.json`:

```json
{
  "stars": 4.0,
  "review": "Predicted review text..."
}
```

---

## 📁 Project Structure

```
Rag_Crew_Profiler/
├── src/first_crew/
│   ├── main.py                  # Entry point
│   ├── crew.py                  # Agent/Task/Tool orchestration + smart caching
│   ├── benchmark_indexing.py    # Full index build script
│   ├── benchmark_rag.py         # Retrieval speed benchmark
│   ├── config/
│   │   ├── agents.yaml          # Agent definitions
│   │   └── tasks.yaml           # Task definitions
│   └── tools/
│       └── custom_tool.py       # Custom tool definitions
├── data/                        # Yelp dataset (gitignored)
├── docs/
│   ├── RAG_Index_Student_Guide.md       # Student guide (中文)
│   ├── RAG_Index_Student_Guide_EN.md    # Student guide (English)
│   ├── Embedding_Index_Lessons_Learned.md  # Lessons learned (English)
│   └── Embedding_Index_踩坑總結.md        # Lessons learned (中文)
├── pyproject.toml
├── uv.lock
└── .env                         # API keys (gitignored)
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Full indexing time (CPU) | 1–4 hours |
| Cached retrieval time | < 0.3 sec / query |
| ChromaDB index size | ~4.6 GB |
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) |

---

## 📚 Documentation

- [Student Guide (English)](docs/RAG_Index_Student_Guide_EN.md) — Step-by-step setup for using pre-built indexes
- [Student Guide (中文)](docs/RAG_Index_Student_Guide.md)
- [Lessons Learned (English)](docs/Embedding_Index_Lessons_Learned.md) — Key pitfalls and solutions
- [踩坑總結 (中文)](docs/Embedding_Index_踩坑總結.md)

---

## 🛠️ Tech Stack

- **Agent Framework**: [CrewAI](https://crewai.com)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) via HuggingFace
- **LLM**: [MiniMax M2.7](https://build.nvidia.com/) via Nvidia Build API
- **Package Manager**: [Astral uv](https://docs.astral.sh/uv/)
