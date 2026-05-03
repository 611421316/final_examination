# 📝 CrewAI Embedding Index — 踩坑總結

> 總結我們在 CrewAI RAG 多智能體系統中，實作 ChromaDB 向量索引的持久化儲存與讀取時，遇到的關鍵發現與踩過的坑。

---

## 🔍 核心發現：效能瓶頸在哪裡？

| 階段 | 耗時 | 原因 |
|------|------|------|
| **索引建立**（首次） | 1–4 小時 | CPU 密集的文字切塊 + SHA256 雜湊 + 向量化寫入 ChromaDB |
| **索引重用**（天真快取） | 仍需 1+ 小時 | `JSONSearchTool(json_path=...)` 會**強制**重跑切塊 + 雜湊比對，即使 ChromaDB 已有資料 |
| **索引重用**（正確做法） | < 1 秒 | 拔掉 `json_path`，直接以 `collection_name` 掛載既有索引 |

---

## 🕳️ 踩過的四個坑

### 坑 1：`json_path` 是萬惡之源

只要宣告 `JSONSearchTool(json_path='...')`，CrewAI 底層就會**無條件**把整個 JSON 檔案讀入、切塊、算 SHA256，再逐一比對 ChromaDB。這段純 Python 同步邏輯即使不寫入任何資料，光是計算就需要數分鐘到數小時。

> ✅ **解法：** 確認 Collection 已存在後，省略 `json_path` 參數，並強制套用 `FixedJSONSearchToolSchema`，防止 LLM 將 `json_path` 作為參數傳回工具而意外觸發重新索引。

### 坑 2：ChromaDB Singleton 衝突

最初用 `chromadb.PersistentClient` 來檢查 Collection 是否存在，結果跟 CrewAI 內部的 ChromaDB 實例搶佔記憶體，噴出 `An instance of Chroma already exists with different settings`。

> ✅ **解法：** 改用原生 `sqlite3` 直接讀 `chroma.sqlite3` 的 `collections` 表，完全不驚動 ChromaDB 引擎。

### 坑 3：Tool Description 模糊導致 Agent 調用失敗

拔掉 `json_path` 後，Tool 的 Pydantic Schema 改變了，加上 Description 太模糊，LLM 直接傳 `{"user_id": "xxx"}` 而不是 `{"search_query": "..."}`，導致 Pydantic 驗證失敗。

> ✅ **解法：** 在 Tool Description 和 Task Description 中明確寫出 `search_query` 的使用範例和格式要求。

### 坑 4：Knowledge 知識庫偷用 OpenAI Embedding

RAG Tools 用了本地 HuggingFace Embedding，但 `StringKnowledgeSource` 的 Embedding 走的是 CrewAI 預設的 OpenAI `text-embedding-3-small`。由於 `OPENAI_API_KEY` 被設成 Nvidia Key，導致 401 認證失敗。

> ✅ **解法：** 在 `Crew()` 建構子加入 `embedder={"provider": "huggingface", "config": {"model": "BAAI/bge-small-en-v1.5"}}`。

---

## 🏗️ 最終架構：`create_rag_tool()` 智能快取偵測器

```python
def create_rag_tool(json_path, collection_name, config, name, description):
    # 1. 用 sqlite3 偷看 chroma.sqlite3 — 避免驚動 ChromaDB Singleton
    # 2. Collection 存在 → JSONSearchTool(collection_name=...) + FixedSchema
    # 3. Collection 不存在 → JSONSearchTool(json_path=...) 觸發全量建索引
```

---

## ⚡ 關鍵數據

| 指標 | 數值 |
|------|------|
| 原始資料集大小 | ~211 MB（3 個 JSON） |
| ChromaDB 索引大小 | ~4.6 GB（`chroma.sqlite3`） |
| 壓縮後大小 | ~2.3 GB（`tar.gz`） |
| 全量索引時間（CPU） | 1–4 小時 |
| 快取讀取時間 | < 0.3 秒/查詢 |
| Embedding 模型 | `BAAI/bge-small-en-v1.5`（384 維） |
| 儲存路徑（macOS） | `~/Library/Application Support/Rag_Crew_Profiler/` |
| 儲存路徑（Linux） | `~/.local/share/Rag_Crew_Profiler/` |
| 儲存路徑（Windows） | `%LOCALAPPDATA%\Rag_Crew_Profiler\` |
