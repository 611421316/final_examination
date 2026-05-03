# 🚀 RAG 向量索引使用指南

> **目標：** 跳過耗時數小時的向量索引建立過程，直接使用預先建好的 ChromaDB 索引來運行 CrewAI 多智能體系統。

---

## 📋 前置條件

| 項目 | 需求 |
|------|------|
| Python | 3.12+ |
| 套件管理 | [Astral `uv`](https://docs.astral.sh/uv/) |
| 作業系統 | macOS / Linux / Windows |
| API Key | Nvidia Build API Key（免費申請） |

---

## Step 1：Clone 專案並安裝依賴

```bash
git clone <repo-url>
cd Rag_Crew_Profiler
uv sync
```

> ⚠️ **嚴禁使用 `pip install`！** 本專案全面採用 `uv` 管理環境。

---

## Step 2：下載並安裝預建索引

### 2.1 取得壓縮檔

從老師提供的 Google Drive 連結，下載 `chroma_index.tar.gz`（約 2.3 GB）。

### 2.2 解壓到正確路徑

CrewAI 使用 `appdirs` 套件管理儲存路徑，**不同作業系統的目標路徑不同**：

**macOS：**
```bash
mkdir -p ~/Library/Application\ Support/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/Library/Application\ Support/Rag_Crew_Profiler/
```

**Linux：**
```bash
mkdir -p ~/.local/share/Rag_Crew_Profiler/
tar -xzf chroma_index.tar.gz -C ~/.local/share/Rag_Crew_Profiler/
```

**Windows (PowerShell)：**
```powershell
$dest = "$env:LOCALAPPDATA\Rag_Crew_Profiler"
New-Item -ItemType Directory -Force -Path $dest
tar -xzf chroma_index.tar.gz -C $dest
```

### 2.3 驗證

確認目標路徑下出現 `chroma.sqlite3` 檔案（約 4.6 GB）。

**macOS：**
```bash
ls -lh ~/Library/Application\ Support/Rag_Crew_Profiler/chroma.sqlite3
```

**Linux：**
```bash
ls -lh ~/.local/share/Rag_Crew_Profiler/chroma.sqlite3
```

**Windows (PowerShell)：**
```powershell
ls "$env:LOCALAPPDATA\Rag_Crew_Profiler\chroma.sqlite3"
```

---

## Step 3：設置環境變數

建立 `.env` 檔案，納入以下項目並填入你的 API Key：

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

> 💡 `minimaxai/minimax-m2.7` 是目前 Nvidia Build 上的免費端點，無需付費即可使用。
> 
> 前往 [https://build.nvidia.com/](https://build.nvidia.com/) 註冊帳號並取得 API Key。

---

## Step 4：驗證索引是否正確載入

執行檢索效能測試腳本，確認預建索引可以被正常讀取：

```bash
uv run python src/first_crew/benchmark_rag.py
```

**預期輸出（正常情況）：**
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

> ✅ 每項查詢應在 **1 秒以內** 完成 (不含模型起始時間)。若任何一項超過10秒，請回到 Step 2 確認 `chroma.sqlite3` 是否放在正確路徑。

---

## Step 5：執行 CrewAI 多智能體預測

```bash
uv run first_crew
```

系統會依序執行三個 Agent：

| 順序 | Agent | 任務 |
|------|-------|------|
| 1 | Yelp User Profiler | 分析目標使用者的評論習慣與偏好 |
| 2 | Yelp Restaurant Analyst | 分析目標餐廳的特色與評價 |
| 3 | Review Prediction Expert | 綜合以上資訊，預測星星數與評論內容 |

**最終結果** 會寫入 `report.json`：

```json
{
  "stars": 4.0,
  "review": "預測的評論文字..."
}
```

---

## 🔧 常見問題排查

### Q1：出現 `401 Incorrect API key` 錯誤
**原因：** `.env` 中的 `NVIDIA_API_KEY` 不正確或已過期。  
**解法：** 前往 [Nvidia Build](https://build.nvidia.com/) 重新產生 API Key。

### Q2：出現 `An instance of Chroma already exists with different settings` 錯誤
**原因：** 同一 Python 行程中，ChromaDB 被重複初始化。  
**解法：** 確認沒有同時執行多個腳本。重新啟動終端機後再執行。

### Q3：`benchmark_rag.py` 測試成功，但 `uv run first_crew` 時 Agent 回報找不到資料
**原因：** Agent 的工具呼叫格式不正確。  
**解法：** 確認 `config/tasks.yaml` 中有包含 `search_query` 的使用範例（本專案已內建）。

### Q4：我想從頭建立自己的索引，不使用預建索引
**指令：**
```bash
uv run python src/first_crew/benchmark_indexing.py
```
> ⚠️ 這會對 211MB 的 JSON 資料集進行全量向量化，在純 CPU 環境下需要 **1-4 小時**。

---

## 📁 索引內容對照表

| Collection 名稱 | 來源資料 | 說明 |
|-----------------|---------|------|
| `benchmark_true_fresh_index_Filtered_User_1` | `data/filtered_user.json` | 使用者特徵（偏好、平均星數等） |
| `benchmark_true_fresh_index_Filtered_Item_1` | `data/filtered_item.json` | 餐廳/商家特徵（分類、地點等） |
| `benchmark_true_fresh_index_Filtered_Review_1` | `data/test_review.json` | 歷史評論全文 |

**Embedding 模型：** `BAAI/bge-small-en-v1.5`（384 維向量，本地 CPU 即可運行）

> 🔴 **重要：** 如果你更換 Embedding 模型，必須重新建立索引！不同模型產生的向量維度不同，混用會導致檢索結果全部錯誤。

---

## 🏗️ 系統架構速覽

```
main.py                     ← 程式入口，讀取測試資料並啟動 Crew
  └── crew.py               ← 核心組裝層，定義 Agent/Task/Tool 綁定
        ├── create_rag_tool()   ← 智能快取偵測器（sqlite3 → 秒級載入 or 全量索引）
        ├── config/agents.yaml  ← Agent 角色、目標、背景故事
        └── config/tasks.yaml   ← Task 描述與預期輸出格式
```

`create_rag_tool()` 的運作邏輯：
1. 用 `sqlite3` 直接讀取 `chroma.sqlite3`，檢查目標 Collection 是否存在
2. **已存在** → 跳過索引，秒級載入（< 1 秒）
3. **不存在** → 自動觸發全量索引建立（數小時）
