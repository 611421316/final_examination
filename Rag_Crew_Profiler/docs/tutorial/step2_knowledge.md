# Step 2: 注入全局背景知識 (CrewAI Knowledge)

**為什麼需要 Knowledge？**
Yelp 的 JSON 檔案中有許多專有的重點欄位（例如 `useful`, `funny`, `compliment_hot`）。如果直接讓大語言模型 (LLM) 去看這些 JSON，它可能會曲解欄位的意思，產生幻覺。

**實作方式：**
我們利用 CrewAI 提供的 `StringKnowledgeSource` 或 `MDTextKnowledgeSource`，把我們寫好的 `docs/Yelp Data Translation.md` 吃進來，變成全體 Agent 共用的「字典常識」。一旦配置到 `@crew(knowledge_sources=[...])` 中，所有 Agent 就能在潛意識裡理解資料集的定義。
