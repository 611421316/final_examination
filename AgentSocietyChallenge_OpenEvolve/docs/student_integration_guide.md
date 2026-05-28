# 學生整合指南 — 把自己的 CrewAI Agent 接進 OpenEvolve 進化框架

> 這份文件教你如何把你在 **`AgentSocietyChallenge_w_CrewAI`**（教學版父 repo）裡設計的
> 多 Agent CrewAI 系統，**遷移**到 **`AgentSocietyChallenge_OpenEvolve`** 這個整合了
> OpenEvolve 進化最佳化的版本，並啟動 LLM 自動演化你的 Agent prompt。

---

## 0. 兩個 Repo 的定位

| Repo | 角色 | 你會做什麼 |
|------|------|-----------|
| [`AgentSocietyChallenge_w_CrewAI`](https://github.com/yuchieh/AgentSocietyChallenge_w_CrewAI) | **設計階段沙盒** | 在這裡設計、迭代、測試你的 Multi-Agent Crew |
| [`AgentSocietyChallenge_OpenEvolve`](https://github.com/yuchieh/AgentSocietyChallenge_OpenEvolve) | **進化階段框架** | 把設計好的 Crew 搬進來，讓 LLM 自動演化最佳 prompt |

兩者**底層完全相同**（同一個 `websocietysimulator/` 競賽框架、同一個 CrewAI 整合層），差別只在於後者多了一個 OpenEvolve evaluator 包在外面。

---

## 1. 整體三層架構

```
┌─────────────────────────────────────────────────────────────┐
│ OpenEvolve（外層 — 只在進化版有）                            │
│   - 用 LLM 突變 agents.yaml 的內容                          │
│   - 跑 evaluator → 收 fitness → 決定下一輪突變方向          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ 把突變後的 YAML 交給下層
┌─────────────────────────────────────────────────────────────┐
│ 你的 CrewAI 系統（中層 — 兩個 repo 都有）                    │
│   - config/agents.yaml      ← 你設計的 Agent 角色            │
│   - config/tasks.yaml       ← 你設計的 Task 指令             │
│   - src/crews/...           ← 你組裝的 Crew 邏輯             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ 透過 wrapper 查資料、回傳預測
┌─────────────────────────────────────────────────────────────┐
│ 競賽 Simulator（底層 — 兩個 repo 都有，不可修改）             │
│   - websocietysimulator/                                    │
│   - InteractionTool: get_user / get_item / get_reviews      │
│   - SimulationEvaluator: 算 stars MAE + review 相似度        │
└─────────────────────────────────────────────────────────────┘
```

**好消息：你的 CrewAI 系統不用任何改動就能搬過來。** 你在父 repo 裡跑得通的東西，搬到進化版之後依然跑得通。差別只是「多了一層可以演化你的 prompts」。

> ✅ **資料路徑也完全一致**：兩個 repo 都使用 `dummy_dataset/`、`dummy_tasks/`、`dummy_groundtruth/`、`review.json` 等相同的目錄結構與檔名，所以你在父 repo 寫的任何 `data_dir` / `task_dir` 路徑都能直接搬過來不用改。

---

## 2. 兩個 Repo 的檔案對應關係（最重要！）

| 父 Repo（設計用） | 進化版 Repo（執行用） | 是否需要修改 |
|-------------------|----------------------|------------|
| `config/agents.yaml` | `config/agents.yaml` | ✅ **整份複製** |
| `config/tasks.yaml` | `config/tasks.yaml` | ✅ **整份複製** |
| `src/crews/simulation_crew.py` | `src/crews/simulation_crew.py` | ✅ **整份複製** |
| `src/tools/interaction_tool_wrapper.py` | `src/tools/interaction_tool_wrapper.py` | ⛔ 通常不動 |
| `src/flows/serving_flow.py` | `src/flows/serving_flow.py` | ⛔ **不要動**（OpenEvolve 已加了 `agents_config_path` 機制） |
| `crewai_simulation_agent.py` | `crewai_simulation_agent.py` | ⛔ **不要動**（同上） |
| `run_pipeline.py` / `run_test.py` | `run_test.py` | ⛔ 直接使用既有的 |
| ❌ 沒有 | `config/agents_evolving.yaml` | 🆕 **新建**（從 `agents.yaml` 加上 `EVOLVE-BLOCK`） |
| ❌ 沒有 | `config/openevolve_config.yaml` | ⛔ 通常用預設值即可 |
| ❌ 沒有 | `openevolve_evaluator.py` | ⛔ 完全不動 |

### 2.1 資料目錄結構（兩個 repo 完全一致）

兩個 repo 使用相同的資料目錄命名，所以**從父 repo 搬資料過來不需要改任何路徑或檔名**：

```
dummy_dataset/
├── item.json                # 商品/商家資料
├── user.json                # 用戶資料
├── review.json              # 歷史評論資料
├── test_review_subset.json  # 取樣用（create_sampled_dataset.py 的輸入）
└── lmdb_cache/              # LMDB 索引（首次執行自動建立，可刪可重建）

dummy_tasks/                 # 模擬任務（task_1.json ~ task_N.json）
dummy_groundtruth/           # 對應的真實答案（groundtruth_*.json）
```

如果你自己用 `create_sampled_dataset.py` 或 `data_process.py` 產生新的資料：

| 你在父 repo 產出的東西 | 進化版要放到哪裡 |
|-----------------------|-----------------|
| `dummy_tasks/task_*.json` | 同名 `dummy_tasks/`（直接複製） |
| `dummy_groundtruth/groundtruth_*.json` | 同名 `dummy_groundtruth/`（直接複製） |
| `dummy_dataset/{item,user,review}.json` | 同名 `dummy_dataset/`（直接複製，記得清掉舊的 `lmdb_cache/`） |

> 💡 路徑寫死在 `openevolve_evaluator.py`、`run_test.py`、`src/utils/create_sampled_dataset.py` 中，但**因為命名完全對齊父 repo，你不需要動這些檔案**。

---

## 3. 完整遷移流程（5 個步驟）

### 假設你已完成

- 在父 repo 設計好一套 Multi-Agent Crew（例如 `user_analyst` + `item_analyst` + `prediction_modeler`）
- 在父 repo 跑過 `uv run python run_pipeline.py` 並得到合理的 metrics

### Step 1：Clone 進化版 Repo 並初始化

```bash
cd ~/WorkSpace/your_workspace/
git clone https://github.com/yuchieh/AgentSocietyChallenge_OpenEvolve.git
cd AgentSocietyChallenge_OpenEvolve

cp .env.example .env
# 編輯 .env 填入你的 OPENAI_API_KEY 與 OPENAI_API_BASE（NVIDIA NIM 等）

make install      # uv sync
```

### Step 2：搬移你的 Agent 設計（3 個檔案）

**設你的父 repo 位置是** `~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/`

```bash
# 從父 repo 整份複製到進化版的同名位置
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/config/agents.yaml             config/agents.yaml
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/config/tasks.yaml              config/tasks.yaml
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/src/crews/simulation_crew.py   src/crews/simulation_crew.py
```

### Step 3：跑 smoke test 確認移植成功

```bash
make smoke
```

預期輸出：
```
✅ 整合測試完成！
overall_quality: 0.7x ~ 0.8x  ← 跟你在父 repo 看到的應該差不多
```

> **如果 smoke test 失敗**，先回去父 repo 確認那邊跑得起來，再對照 Step 2 的檔案路徑。**這一步不通就不要往下走**。

### Step 4：建立 `agents_evolving.yaml` — 標記哪些 Agent 要被演化

複製一份 `agents.yaml` 出來：

```bash
cp config/agents.yaml config/agents_evolving.yaml
```

接著編輯 `config/agents_evolving.yaml`，**用 `EVOLVE-BLOCK` 標記框出你想讓 LLM 突變的範圍**。

#### 規則（非常重要）

| 規則 | 說明 |
|------|------|
| 🔴 **整份檔案恰好一個 block** | OpenEvolve 不接受多個 block（雖然技術上能跑，但會收斂變慢） |
| 🟢 **有 tool 的 Agent 放 block 外** | 因為它的 prompt 跟 `tasks.yaml` 裡的 ReAct 格式緊密耦合，被改壞就無法呼叫 tool |
| 🟢 **純推理 Agent 放 block 內** | 這些 Agent 沒有 tool，只是組合資訊產生輸出，最適合被演化 |
| 🟢 **`llm:` 欄位也放 block 內** | LLM 自然會保留它，但放外面更穩 |

#### 實例：以父 repo 的 3-agent 結構為例

```yaml
# config/agents_evolving.yaml

# === 有工具的 Agent 放在 EVOLVE-BLOCK 外（保護 tool calling 格式）===
user_analyst:
  role: >
    Yelp User Profiler
  goal: >
    Analyze user {user_id}'s historical reviews and preferences
  backstory: >
    You are an expert behavior analyst...
  llm: openai/minimaxai/minimax-m2.7

item_analyst:
  role: >
    Yelp Restaurant Analyst
  goal: >
    Analyze business {item_id}'s characteristics and reputation
  backstory: >
    You are a restaurant critic...
  llm: openai/minimaxai/minimax-m2.7

# === 純推理 Agent 放在 EVOLVE-BLOCK 內，讓 LLM 演化 prompt ===
# EVOLVE-BLOCK-START
prediction_modeler:
  role: >
    Review Prediction Expert
  goal: >
    Predict the exact Star rating (1.0 to 5.0) and generate a mock review text...
  backstory: >
    You are a master of predicting human behavior...
  llm: openai/minimaxai/minimax-m2.7
# EVOLVE-BLOCK-END
```

### Step 5：啟動進化

```bash
# 預設 10 iter × 5 tasks
make evolve

# 自訂參數
make evolve ITERS=50 TASKS=3

# 從 checkpoint 繼續（如果之前的跑被中斷）
make evolve-resume CHECKPOINT=config/openevolve_output/checkpoints/checkpoint_10
```

進化過程中可以另開終端機看視覺化：

```bash
make visualize
# 打開瀏覽器 http://127.0.0.1:8080
```

---

## 4. 進化完成後拿走最佳結果

```bash
# 1. 找出最終最佳分數
cat config/openevolve_output/best/best_program_info.json

# 2. 取出最佳的 agents YAML
cat config/openevolve_output/best/best_program.yaml

# 3. 套回正式推論用的 agents.yaml
cp config/openevolve_output/best/best_program.yaml config/agents.yaml

# 4. 用最佳版本跑一次完整測試
make test
```

> 注意：`agents_evolving.yaml` 裡的 `EVOLVE-BLOCK` 標記只會出現在進化過程的暫存檔，
> `best_program.yaml` 也會帶有這些註解。複製到 `agents.yaml` 後可以保留也可以刪掉，
> CrewAI 都會忽略 `#` 開頭的 YAML 註解。

---

## 5. 常見錯誤排查

| 症狀 | 可能原因 | 解法 |
|------|---------|------|
| `make smoke` 拿到 `overall_quality: 0.27` | LLM API 失敗，fallback 給最低分 | 檢查 `.env` 的 API key / base URL；確認 `make test-mock` 跑得起來（mock 模式不用 token） |
| Evaluator 顯示 `⏱  Simulation exceeded 900s` | 單次 simulation 被 LLM rate limit 卡住 | 減少 `TASKS=N`、增加 `OPENEVOLVE_SIM_TIMEOUT=1800`、或換 API |
| LLM 突變後 YAML 結構爛掉 → 全部 evaluation 失敗 | OpenEvolve 預設用 diff 模式，YAML 不適合 | 確認 `config/openevolve_config.yaml` 設 `diff_based_evolution: false` |
| `combined_score` 完全沒進步 | EVOLVE-BLOCK 範圍太小，演化空間不足 | 把更多 agent 移進 block；或增加 `ITERS` |
| Tool calling 完全失敗（agent 不呼叫工具） | 把有 tool 的 agent 放進了 EVOLVE-BLOCK，LLM 把 role 改壞 | 把有 tool 的 agent 移出 block |

---

## 6. 進化框架的執行流程速覽

當你執行 `make evolve` 時，幕後發生的事：

```
[每個 iteration 重複 ITERS 次]

  1. OpenEvolve 從 population 採樣一個 parent YAML
                ▼
  2. LLM 看著 parent YAML（含你寫的 EVOLVE-BLOCK 標記），
     full-rewrite 出一份突變版 YAML
                ▼
  3. 突變版寫進 /tmp/xxxxx.yaml
                ▼
  4. OpenEvolve 呼叫 openevolve_evaluator.py 的 evaluate(/tmp/xxxxx.yaml)
                ▼
  5. evaluate() 設定環境變數 OPENEVOLVE_AGENTS_YAML=/tmp/xxxxx.yaml
                ▼
  6. evaluate() 呼叫 simulator.run_simulation(N=TASKS)
                ▼
  7. Simulator 對每個 task 啟動 CrewAISimulationAgent
                ▼
  8. CrewAISimulationAgent 讀環境變數 → 載入 /tmp/xxxxx.yaml
     → 用這份突變 agents 設定組裝 SimulationCrew → 執行 3 個 Tasks
                ▼
  9. 每個 task 產出 {stars, review}，Simulator 對比 groundtruth 算 metrics
                ▼
 10. evaluate() 回傳 {"combined_score": overall_quality}
                ▼
 11. OpenEvolve 把這個 fitness 寫進 MAP-Elites 資料庫
                ▼
 12. 下一輪 iteration 採樣時，高分 YAML 比較容易被選為 parent
```

---

## 7. 額外資源

| 想了解什麼 | 看哪裡 |
|------|------|
| OpenEvolve 怎麼運作 | https://github.com/algorithmicsuperintelligence/openevolve |
| EVOLVE-BLOCK 怎麼寫 | `examples/README.md` 在 OpenEvolve repo 內 |
| 競賽框架（Simulator / InteractionTool）詳細 API | `docs/legacy/tutorials/agent_development.md` |
| 我們對 OpenEvolve 做的整合細節 | `CLAUDE.md` |

---

## TL;DR — 一頁懶人指南

1. **Clone** `AgentSocietyChallenge_OpenEvolve`，`cp .env.example .env` 填好 API key
2. **複製 3 個檔案** 從父 repo 到進化版：`agents.yaml`、`tasks.yaml`、`simulation_crew.py`
3. **`make smoke`** 確認沒壞
4. **`cp agents.yaml agents_evolving.yaml`**，編輯後者，把純推理 Agent 包進**一個** `EVOLVE-BLOCK`
5. **`make evolve ITERS=50 TASKS=3`** 啟動演化
6. 結束後 **`cp config/openevolve_output/best/best_program.yaml config/agents.yaml`** 套用最佳版本

🎉 享受 LLM 自動幫你優化 prompt 的成果。
