# Step 5: 嚴格分離的 Task 定義 (`tasks.yaml`)

對應的三個任務定義在 `config/tasks.yaml` 裡，透過描述去引導 Agent 的行動與輸出結果。

1. **`analyze_user_task`**: 要求代理利用 `{user_id}` 去翻找檢索庫，整理出一份完整的 Markdown 分析報告。
2. **`analyze_item_task`**: 要求代理利用 `{item_id}` 去翻找檢索庫，整理出店家的評價報告。
3. **`predict_review_task`**: 強制要求最終的輸出長相必須是一個 JSON Object，精準包含 `"stars"` 和 `"text"` 兩個欄位，這也是後續 OpenEvolve 要讀取來計算 MAE 分數的關鍵。
