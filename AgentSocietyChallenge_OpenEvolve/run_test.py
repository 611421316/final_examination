"""
run_test.py — AgentSociety + CrewAI 整合測試腳本

支援三種模式：
  1. 真實 LLM 模式 (預設)：透過 NVIDIA NIM API 進行推論
  2. Mock 模式：攔截 OpenAI API 呼叫，使用假回覆進行快速結構驗證
  3. Smoke test：只跑 1 個 task，快速驗證端到端管線

用法：
  uv run python run_test.py                       # 真實 LLM，全部任務
  uv run python run_test.py --mock                # Mock 模式
  uv run python run_test.py --tasks 1             # Smoke test（取代 run_nvidia_test.py）
  uv run python run_test.py --tasks 3 --threads 2 # 自訂任務數與執行緒
"""
import argparse
import os
import sys
import logging
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentSociety + CrewAI integration test")
    p.add_argument("--mock", action="store_true",
                   help="使用 Mock LLM，不消耗 token")
    p.add_argument("--tasks", type=int, default=None,
                   help="要跑的 task 數量（預設：全部）")
    p.add_argument("--threads", type=int, default=2,
                   help="ThreadPool 的 max_workers（預設：2）")
    return p.parse_args()


def install_mock_llm() -> None:
    """攔截 OpenAI client 的 chat completion 呼叫，回傳固定內容。"""
    from unittest.mock import patch

    def fake_completion(*args, **kwargs):
        class FakeMessage:
            content = '{"stars": 4.8, "review": "[Mocked LLM] Awesome place, highly recommended!"}'
            tool_calls = None
        class FakeChoice:
            message = FakeMessage()
            finish_reason = "stop"
        class FakeResponse:
            choices = [FakeChoice()]
            id = "mock-id"
            model = "gpt-4"
            usage = None
        return FakeResponse()

    patcher = patch(
        "openai.resources.chat.completions.Completions.create",
        side_effect=fake_completion,
    )
    patcher.start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  模式: Mock LLM (不消耗 Token)")


def report_real_llm_env() -> None:
    """從 .env 載入並回報 API key / base URL 狀態。"""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    print("⚙️  模式: 真實 LLM (NVIDIA NIM)")
    print(f"🔑 API Key: {'✅ 已設定' if api_key else '❌ 未設定'}")
    print(f"🌐 Base URL: {api_base or '❌ 未設定'}")


def main() -> int:
    args = parse_args()

    if args.mock:
        install_mock_llm()
    else:
        report_real_llm_env()

    # 框架 import 必須在 mock 注入之後
    from websocietysimulator import Simulator
    from crewai_simulation_agent import CrewAISimulationAgent

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🚀 啟動 AgentSociety CrewAI 整合測試 (End-to-End)")
    print("=" * 60)

    try:
        # 1. 建立 Simulator
        print(">>> 載入 Dataset (data)...")
        simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
        simulator.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth",
        )
        simulator.set_agent(CrewAISimulationAgent)

        # 2. 執行模擬
        task_desc = "全部" if args.tasks is None else str(args.tasks)
        print(f"\n⚙️  開始推論（tasks={task_desc}，threads={args.threads}）...")
        outputs = simulator.run_simulation(
            number_of_tasks=args.tasks,
            enable_threading=True,
            max_workers=args.threads,
        )

        print("\n🏆 引擎運算完畢，最終產出:")
        print("-" * 60)
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        print("-" * 60)

        # 3. 官方評分
        print("\n📊 呼叫官方評分系統 (simulator.evaluate())...")
        evaluation_results = simulator.evaluate()
        print("💡 競賽衡量結果:")
        print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

        print("\n✅ 整合測試完成！")
        return 0

    except Exception as e:
        print(f"\n❌ 測試中斷: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
