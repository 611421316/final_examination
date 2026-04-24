# Step 7: 測試執行腳本 (`main.py`)

為了驗證系統是否能順暢運作，我們修改了 `src/first_crew/main.py` 的進入點：
1. 它會直接去讀取 `data/test_review_subset.json`。
2. 抓取第一列（第一筆）測資的 `user_id` 和 `item_id`。
3. 把這兩個值作為 `inputs` 送進 CrewAI 的 `kickoff()` 方法中。

如果配置都正確，你會看到 Agent 在 Console 互相對話，最後產出 `report.json`。
