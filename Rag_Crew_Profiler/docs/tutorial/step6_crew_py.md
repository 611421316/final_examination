# Step 6: 系統組裝與工具綁定 (`crew.py`)

在 `src/first_crew/crew.py` 當中，我們的重點是**配發裝備**：
- 透過 `@agent` 裝飾器，我們讀取 YAML，並在 Python 程式碼中利用 `tools=[user_rag_tool]` 將特定的 RAG 工具掛載給負責的代理。
- 透過 `@task` 裝飾器，我們將任務讀入，並可使用 `output_file='report.json'` 來保存預測家的最終成果。
- 在 `Crew()` 建構子中，將 `Step 2` 準備好的 `knowledge_sources` 放進去，這樣整個 Crew 團隊就會帶著這份背景知識去工作。
