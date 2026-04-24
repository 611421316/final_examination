# Step 1: 了解目標與資料集 (Data & Objective)

**目標分析：**
在 AgentSociety Challenge (Milestone 1) 中，我們需要建立一個 Static Baseline 系統，藉由擷取測試資料集的「指定使用者ID」與「指定餐廳ID」，讓 Agent 分析兩者的過去資料，最後預測出該使用者會給出的星等 (Stars) 與撰寫的評論 (Text)。

**資料集角色：**
- **測試入口** (`data/test_review_subset.json`): 這是我們的考卷，負責提供 `{user_id}` 和 `{item_id}`。
- **檢索記憶庫** (`user_subset.json`, `item_subset.json`, `review_subset.json`): 這是 Agent 可以主動查閱的圖書館，協助他們拼湊出「使用者畫像」和「餐廳特徵」。
