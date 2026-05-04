"""
run_simulator_test.py — AgentSociety + CrewAI 整合測試腳本
支援兩種模式：
  1. 真實 LLM 模式 (預設)：透過 NVIDIA NIM API 進行推論
  2. Mock 模式：攔截 OpenAI API 呼叫，使用假回覆進行快速結構驗證
用法：
  uv run python run_simulator_test.py              # 真實 LLM
  uv run python run_simulator_test.py --mock       # Mock 模式
"""
import sys
import os
import logging
import json

# ======================================================================
# [模式切換] Mock vs. Real LLM
# ======================================================================
USE_MOCK = "--mock" in sys.argv

if USE_MOCK:
    from unittest.mock import patch
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    import time

    def fake_completion(*args, **kwargs):
        return ChatCompletion(
            id="mock-id",
            choices=[Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content='{"stars": 4.8, "review": "[Mocked LLM] Awesome place, highly recommended!"}',
                    role="assistant"
                )
            )],
            created=int(time.time()),
            model="gpt-4",
            object="chat.completion"
        )

    patcher = patch('openai.resources.chat.completions.Completions.create', side_effect=fake_completion)
    patcher.start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  模式: Mock LLM (不消耗 Token)")
else:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    print(f"⚙️  模式: 真實 LLM (NVIDIA NIM)")
    print(f"🔑 API Key: {'✅ 已設定' if api_key else '❌ 未設定'}")
    print(f"🌐 Base URL: {api_base or '❌ 未設定'}")

# ======================================================================
# 載入框架
# ======================================================================
from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

logging.basicConfig(level=logging.INFO)

print("\n" + "=" * 60)
print("[RUN] 啟動 AgentSociety CrewAI 整合測試 (End-to-End)")
print("=" * 60)

try:
    # 1. 建立 Simulator，使用 Toy Dataset + Cache
    print(">>> 載入 Toy Dataset (dummy_dataset)...")
    simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir="dummy_tasks", groundtruth_dir="dummy_groundtruth")
    simulator.set_agent(CrewAISimulationAgent)

    # 2. 運行模擬
    print("\n[...] 開始推論...")
    outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=True, max_workers=2)

    print("\n[OUT] 引擎運算完畢，最終產出:")
    print("-" * 60)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("-" * 60)

    # 3. 官方評分
    print("\n[EVAL] 呼叫官方評分系統 (simulator.evaluate())...")
    evaluation_results = simulator.evaluate()
    print("[EVAL] 競賽衡量結果:")
    print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

    print("\n[OK] 整合測試完成！")

except Exception as e:
    print(f"\n[FAIL] 測試中斷: {e}")
    import traceback
    traceback.print_exc()
