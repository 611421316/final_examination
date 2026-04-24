# Milestone 1: Static Baseline System (CrewAI + RAG + Knowledge)

本指南專注於引導學生完成 **AgentSociety Challenge - Milestone 1**。目標是基於目前的 `src/first_crew/` 架構，結合 CrewAI 的 Knowledge 與 RAG Tools，打造一個能預測 `test_review_subset.json` 中使用者星等與評論內容的多智能體系統 (Multi-Agent System)。

---

## �️ 教學與實作步驟 (Step-by-Step Guide)

### Step 1: 了解目標與資料集 (Data & Objective)
*   **輸入 (Input):** 從 `data/test_review_subset.json` 中抽取出的一組 `{user_id}` 與 `{business_id}`。
*   **輸出 (Output):** AI Agents 綜合判斷後預測出的 `{stars}` (星等) 與 `{text}` (評論內容)。
*   **檢索池 (Retrieval Pool):** `data/user_subset.json`, `data/item_subset.json`, `data/review_subset.json`。

### Step 2: 注入全局背景知識 (CrewAI Knowledge)
**目的**：讓所有的 Agents 都能「先天」看懂 Yelp JSON 的欄位名稱定義，避免他們在讀取資料時對 `compliment_hot` 或 `useful` 等欄位產生解釋錯誤的幻覺 (Hallucination)。
*   **實作方法**：在 `src/first_crew/crew.py` 中，使用 `MDTextKnowledgeSource` 或 `StringKnowledgeSource` 將 `docs/Yelp Data Translation.md` 的內容讀入，並配置到 `@crew` 的 `knowledge` 參數中。

### Step 3: 配置主動檢索武器 (CrewAI RAG Tools)
**目的**：為 Agents 裝備能夠去搜尋龐大本機 JSON 檔案的工具。
*   **實作方法**：指導學生在 `src/first_crew/crew.py` 初始化三個 `JSONSearchTool`。
    *   `user_rag_tool = JSONSearchTool(json_path='data/user_subset.json')`
    *   `item_rag_tool = JSONSearchTool(json_path='data/item_subset.json')`
    *   `review_rag_tool = JSONSearchTool(json_path='data/review_subset.json')`

### Step 4: 嚴格分離的 Agent 定義 (`config/agents.yaml`)
遵循 YAML-First 原則，定義三個職責分明的 Agents：
1.  **`user_analyst` (使用者輪廓分析師):** 專注於分析特定 user 的歷史行為。
2.  **`item_analyst` (店家分析師):** 專注於分析特定 business 的設施與總體評價。
3.  **`prediction_modeler` (評論預測家):** 讀取前兩者的分析報告，進行最終預測。

### Step 5: 嚴格分離的 Task 定義 (`config/tasks.yaml`)
1.  **`analyze_user_task`**: 使用 `{user_id}` 尋找使用者的歷史紀錄，輸出其「給分習慣與口味偏好」。(指派給 `user_analyst`)
2.  **`analyze_item_task`**: 使用 `{business_id}` 尋找店家的特徵與別人的評論，輸出「餐廳優劣勢總結」。(指派給 `item_analyst`)
3.  **`predict_review_task`**: 綜合上述兩份報告，輸出一段 Markdown，內部包含預測的 `Star/Rating` 以及模擬寫出的 `Review Text`。(指派給 `prediction_modeler`)

### Step 6: 系統組裝與工具綁定 (`src/first_crew/crew.py`)
重點在於透過 Decorator 將 Step 3 的 RAG Tools 掛載給正確的 Agent：
*   在 `@agent def user_analyst(self):` 中加入 `tools=[user_rag_tool, review_rag_tool]`。
*   在 `@agent def item_analyst(self):` 中加入 `tools=[item_rag_tool, review_rag_tool]`。
*   在 `@crew` 中，將 Step 2 的 `knowledge` 變數加入 `Crew(agents=..., tasks=..., knowledge=knowledge_sources)` 進行全局綁定。

### Step 7: 測試執行腳本 (`src/first_crew/main.py`)
撰寫一段 Python 程式碼，讀取 `data/test_review_subset.json` 的第一筆測試資料。
擷取其 `user_id` 與 `business_id` 後，將其作為 `inputs` 參數丟入 `FirstCrew().crew().kickoff(inputs=inputs)`，並將模型輸出的星等/評論與 JSON 內的 Ground Truth（實際答案）進行比對，以此完成 Milestone 1！
