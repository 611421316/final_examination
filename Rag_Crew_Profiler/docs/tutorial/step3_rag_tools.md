# Step 3: 配置主動檢索武器 (CrewAI RAG Tools)

**為什麼需要 RAG？**
Yelp 資料子集動輒幾 MB，如果直接硬塞進 Agent 的 Prompt 會導致 Token 暴增甚至崩潰。

**實作方式：**
我們為 Agents 準備名為 `JSONSearchTool` 的武器（RAG Tool）。
- 給 `user_analyst` 可以查詢使用者庫 (`user_subset.json`) 與評論庫 (`review_subset.json`) 的權限。
- 給 `item_analyst` 查詢店家庫 (`item_subset.json`) 的權限。
這樣一來，他們就可以在執行任務時，「主動」下 Query 來檢索 JSON 樹中的關聯節點。
