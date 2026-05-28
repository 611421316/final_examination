# CLAUDE.md — AgentSocietyChallenge_OpenEvolve

此 Repo 是在 `AgentSocietyChallenge_w_CrewAI` 的基礎上，加入 **OpenEvolve** 進化式 Prompt 最佳化整合的沙盒版本。

---

## Git 工作流程

### 核心原則
**`main` 只接受 PR merge，不直接 commit。**

所有開發工作在 Claude Code 建立的 worktree 分支上進行，完成後透過 PR merge 回 `main`。

### Session 開始時

```bash
# 1. 拉取遠端最新的 main
git -C /Users/jack.ho/WorkSpace/LLM_APP_2026/AgentSocietyChallenge_OpenEvolve pull origin main

# 2. 將 worktree 分支 rebase 到最新 main 上
git -C <worktree路徑> rebase main
```

### Session 結束時

```
Push worktree 分支 → 開 PR → merge 進 main → 本地 pull main
```

---

## 專案結構

| 路徑 | 說明 | 可否修改 |
|------|------|---------|
| `websocietysimulator/` | 競賽核心框架 | ❌ 不可修改 |
| `crewai_simulation_agent.py` | 官方 Simulator ↔ CrewAI 轉接器 | ❌ 不可修改 |
| `src/flows/serving_flow.py` | Flow 與 JSON 解析 | ⚠️ 僅 OpenEvolve 相關部分（如 `agents_config_path`）可修改 |
| `src/crews/simulation_crew.py` | CrewAI Crew 組裝 | ✅ 可修改 |
| `config/agents.yaml` | Agent 角色定義（正式推論用） | ✅ 可修改 |
| `config/agents_evolving.yaml` | OpenEvolve 進化初始程式（含 EVOLVE-BLOCK） | ✅ 可修改 |
| `config/tasks.yaml` | Task 指令設計 | ✅ 可修改 |
| `config/openevolve_config.yaml` | OpenEvolve 執行設定 | ✅ 可修改 |
| `openevolve_evaluator.py` | OpenEvolve 評估函式 | ✅ 可修改 |

---

## 資料合約

- Track 1 輸出格式：`{"stars": float, "review": str}`
- Pydantic State 欄位：`predicted_rating`、`generated_review`（名稱不可改）
- InteractionTool 查詢類型：`"user"`, `"item"`, `"review_by_user"`, `"review_by_item"`

---

## 資料目錄

```
dummy_dataset/
├── item.json                  # 商家/商品資料
├── user.json                  # 用戶資料
├── review.json                # 歷史評論資料
├── test_review_subset.json    # 取樣用評論資料（create_sampled_dataset.py 的輸入）
└── lmdb_cache/                # LMDB 索引快取（首次執行自動建立）

dummy_tasks/                   # 5 個模擬任務（task_1.json ~ task_5.json）
dummy_groundtruth/             # 對應 5 個真實答案
```

> 路徑命名刻意與父 repo [`AgentSocietyChallenge_w_CrewAI`](https://github.com/yuchieh/AgentSocietyChallenge_w_CrewAI) 對齊，讓學生從父 repo 搬程式過來時不需要修改任何路徑。

---

## 環境管理：永遠使用 uv

本專案**只使用 uv** 管理環境，禁止使用 `pip install`、`poetry`、`conda` 或直接呼叫系統 `python3`。

```bash
# ✅ 正確
uv run python run_test.py
uv sync
uv add <package>

# ❌ 禁止
python3 run_test.py
pip install <package>
```

---

## 初次設定

```bash
cp .env.example .env       # 填入 OPENAI_API_KEY 與 OPENAI_API_BASE
make install               # 等同 uv sync
```

---

## 執行測試（推薦使用 Makefile）

```bash
make test-mock             # Mock 模式（零成本結構驗證）
make test                  # 真實 LLM（NVIDIA NIM）
make smoke                 # Smoke test（只跑 1 筆）
```

或使用完整指令：

```bash
uv run --env-file .env python run_test.py --mock
uv run --env-file .env python run_test.py
uv run --env-file .env python run_test.py --tasks 1
```

---

## 執行 OpenEvolve（推薦使用 Makefile）

```bash
make evolve                              # 預設 10 iterations、5 tasks
make evolve ITERS=20 TASKS=3             # 自訂參數
make evolve-resume CHECKPOINT=config/openevolve_output/checkpoints/checkpoint_10
make evolve-test                         # 本地 evaluator 整合測試
make visualize                           # 啟動視覺化伺服器
```

或使用完整指令：

```bash
OPENEVOLVE_NUM_TASKS=5 uv run --env-file .env python -m openevolve.cli \
  config/agents_evolving.yaml openevolve_evaluator.py \
  --config config/openevolve_config.yaml \
  --output config/openevolve_output \
  --iterations 10
```

---

## OpenEvolve 整合注意事項

- `agents_evolving.yaml` 必須包含**恰好一個** `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` 區塊
- `evaluate()` 必須是**模組頂層函式**，簽名為 `evaluate(program_path: str) -> dict`，且回傳 dict 必須包含 `combined_score` key
- `config/openevolve_config.yaml` 已設定 `diff_based_evolution: false`（full rewrite 模式）與 `cascade_evaluation: false`
- **三個 agent（包含 `data_retriever`）皆在 EVOLVE-BLOCK 內**；但 `data_retriever` 的 tool calling 格式由 `tasks.yaml` 強制，OpenEvolve 的 system_message 也提醒 LLM 不要破壞它
- `data_retriever` 會做 **4 次** tool 呼叫（user / item / review_by_user / review_by_item）— 第 4 次取回該商品的「全站他人評分分佈」，作為校準依據
