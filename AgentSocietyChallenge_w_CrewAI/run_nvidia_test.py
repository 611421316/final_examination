"""
run_nvidia_test.py — 驗證 NVIDIA NIM API 是否能與 CrewAI + AgentSociety 完整串通
此腳本不使用任何 Mock，會真實呼叫 NVIDIA 端點的 LLM。
"""
import os
import logging
import json

# ======================================================================
# [第零步] 從 .env 載入 API Key 與 Base URL
# ======================================================================
from dotenv import load_dotenv
load_dotenv()

# 驗證環境變數是否正確載入
api_key = os.environ.get("OPENAI_API_KEY", "")
api_base = os.environ.get("OPENAI_API_BASE", "")
print(f"🔑 API Key 載入狀態: {'✅ 已設定' if api_key.startswith('nvapi') else '❌ 未偵測到 nvapi key!'}")
print(f"🌐 API Base URL: {api_base or '❌ 未設定'}")

from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

logging.basicConfig(level=logging.INFO)

print("\n" + "="*60)
print("🚀 啟動 NVIDIA NIM + CrewAI 真實 LLM 整合驗證")
print("="*60)

try:
    # 1. 載入 Toy Dataset
    print(">>> 載入 Toy Dataset (dummy_dataset)...")
    simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
    simulator.set_task_and_groundtruth(task_dir="dummy_tasks", groundtruth_dir="dummy_groundtruth")
    simulator.set_agent(CrewAISimulationAgent)

    # 2. 真實呼叫 NVIDIA LLM 進行推論
    print("\n⚙️  開始真實 LLM 推論 (NVIDIA NIM → minimaxai/minimax-m2.7)...")
    outputs = simulator.run_simulation(number_of_tasks=1)
    
    print("\n🏆 引擎運算完畢，真實 LLM 產出:")
    print("-" * 60)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    print("-" * 60)

    # 3. 官方評分
    print("\n📊 呼叫官方評分系統...")
    evaluation_results = simulator.evaluate()
    print("💡 競賽衡量結果:")
    print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))
    print("\n✅ NVIDIA NIM API 驗證成功！整條管線暢通無阻！")

except Exception as e:
    print(f"\n❌ 測試中斷: {e}")
    import traceback
    traceback.print_exc()
