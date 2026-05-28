# Makefile — AgentSocietyChallenge_OpenEvolve
#
# 統一封裝常用指令，避免重複輸入冗長的 uv 命令。
# 所有 target 都會自動讀取 .env 內的環境變數。

.DEFAULT_GOAL := help

# OpenEvolve 進化參數（可由 CLI 覆寫，例如：make evolve ITERS=20 TASKS=3）
ITERS ?= 10
TASKS ?= 5
OUTPUT ?= config/openevolve_output

# ============================================================================
# 環境
# ============================================================================
.PHONY: install
install:  ## 同步依賴（uv sync）
	uv sync

# ============================================================================
# 測試
# ============================================================================
.PHONY: test-mock
test-mock:  ## Mock 模式整合測試（零成本）
	uv run --env-file .env python run_test.py --mock

.PHONY: test
test:  ## 真實 LLM 整合測試（全部 task）
	uv run --env-file .env python run_test.py

.PHONY: smoke
smoke:  ## Smoke test（只跑 1 個 task）
	uv run --env-file .env python run_test.py --tasks 1

# ============================================================================
# OpenEvolve 進化
# ============================================================================
.PHONY: evolve
evolve:  ## 啟動 OpenEvolve 進化（可調 ITERS=N TASKS=N）
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output $(OUTPUT) \
	    --iterations $(ITERS)

.PHONY: evolve-resume
evolve-resume:  ## 從最新 checkpoint 繼續（指定 CHECKPOINT=path）
	@if [ -z "$(CHECKPOINT)" ]; then \
	    echo "ERROR: 必須指定 CHECKPOINT 變數，例如：make evolve-resume CHECKPOINT=config/openevolve_output/checkpoints/checkpoint_10"; \
	    exit 1; \
	fi
	OPENEVOLVE_NUM_TASKS=$(TASKS) \
	uv run --env-file .env python -m openevolve.cli \
	    config/agents_evolving.yaml \
	    openevolve_evaluator.py \
	    --config config/openevolve_config.yaml \
	    --output $(OUTPUT) \
	    --checkpoint $(CHECKPOINT) \
	    --iterations $(ITERS)

.PHONY: evolve-test
evolve-test:  ## 本地整合測試 evaluator（不啟動進化）
	uv run --env-file .env python openevolve_evaluator.py

.PHONY: visualize
visualize:  ## 啟動 OpenEvolve 視覺化伺服器（http://127.0.0.1:8080）
	uv run python /Users/jack.ho/WorkSpace/openevolve/scripts/visualizer.py \
	    --path $(OUTPUT)

# ============================================================================
# 資料處理
# ============================================================================
.PHONY: sample-dataset
sample-dataset:  ## 從 data/test_review_subset.json 取樣評估資料集
	uv run python src/utils/create_sampled_dataset.py --n 5

# ============================================================================
# 維護
# ============================================================================
.PHONY: clean
clean:  ## 清理 __pycache__ 與 .pyc
	find . -type d -name __pycache__ ! -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" ! -path "./.venv/*" -delete 2>/dev/null || true

.PHONY: clean-output
clean-output:  ## 清除 OpenEvolve 輸出（⚠️ 會刪除 checkpoints！）
	@echo "⚠️  這會刪除 $(OUTPUT) 下所有 checkpoints。確定？(Ctrl+C 取消，Enter 繼續)"
	@read CONFIRM
	rm -rf $(OUTPUT)

# ============================================================================
# Help
# ============================================================================
.PHONY: help
help:  ## 列出所有 target
	@echo "可用指令："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
