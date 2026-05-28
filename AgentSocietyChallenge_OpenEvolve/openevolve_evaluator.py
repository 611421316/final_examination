import os
import tempfile
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

# 整個 simulation 的 hard timeout（秒）。超時則回傳 fallback fitness 讓 OpenEvolve 繼續。
# 預設 15 分鐘，可由 OPENEVOLVE_SIM_TIMEOUT env var 覆寫。
SIM_TIMEOUT_SEC = int(os.environ.get("OPENEVOLVE_SIM_TIMEOUT", 900))

# ---------------------------------------------------------------------------
# Lazy singleton: Simulator is expensive to initialize (loads LMDB dataset).
# OpenEvolve imports this module once and calls evaluate() many times, so we
# initialize on the first call and reuse the same instance afterward.
# ---------------------------------------------------------------------------
_simulator: Simulator = None

def _get_simulator() -> Simulator:
    global _simulator
    if _simulator is None:
        logging.getLogger().setLevel(logging.WARNING)
        print("[Evaluator] Initializing Simulator with sampled dataset (one-time)...")
        _simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
        _simulator.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth"
        )
        _simulator.set_agent(CrewAISimulationAgent)
        print("[Evaluator] Simulator ready.")
    return _simulator


def evaluate(program_path: str) -> dict:
    """
    Module-level function required by OpenEvolve.

    OpenEvolve writes the mutated YAML to a temp file (suffix configured as
    .yaml) and passes the FILE PATH here as the sole argument.

    Returns a dict with 'combined_score' as the primary fitness metric (required
    by OpenEvolve), plus individual sub-metrics for MAP-Elites feature tracking.

    combined_score = overall_quality (0–1):
      overall_quality = (preference_estimation + review_generation) / 2
    where preference_estimation = 1 - normalized_star_MAE.
    """
    simulator = _get_simulator()
    try:
        # 1. Tell CrewAISimulationAgent to load this YAML config for the run
        os.environ["OPENEVOLVE_AGENTS_YAML"] = program_path

        num_tasks = int(os.environ.get("OPENEVOLVE_NUM_TASKS", 5))
        print(f"\n[Evaluator] Running simulation: {program_path}  (tasks={num_tasks}, timeout={SIM_TIMEOUT_SEC}s)")

        # Hard timeout 包住整個 simulation。如果 simulator/CrewAI/LiteLLM 內部卡住
        # （例如 rate limit retry 死循環），這層會在 SIM_TIMEOUT_SEC 後強制中止，
        # 讓 evaluator 回傳 fallback 分數讓 OpenEvolve 能繼續下一個 iteration。
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    simulator.run_simulation,
                    number_of_tasks=num_tasks,
                    enable_threading=True,
                    max_workers=2,
                )
                future.result(timeout=SIM_TIMEOUT_SEC)
        except FuturesTimeout:
            print(f"[Evaluator] ⏱  Simulation exceeded {SIM_TIMEOUT_SEC}s — returning fallback score")
            return {"combined_score": 0.0}

        # 2. Compute official metrics
        # eval_results structure:
        #   {"type": "simulation", "metrics": <SimulationMetrics.__dict__>, "data_info": {...}}
        print("[Evaluator] Calculating official metrics...")
        eval_results = simulator.evaluate()

        metrics           = eval_results.get("metrics", {}) if isinstance(eval_results, dict) else {}
        overall_quality   = metrics.get("overall_quality", 0.0)
        pref_estimation   = metrics.get("preference_estimation", 0.0)
        review_generation = metrics.get("review_generation", 0.0)

        print(
            f"[Evaluator] preference_estimation={pref_estimation:.4f}, "
            f"review_generation={review_generation:.4f}, "
            f"overall_quality={overall_quality:.4f}  →  combined_score={overall_quality:.4f}"
        )

        return {"combined_score": float(overall_quality)}

    except Exception as e:
        print(f"[Evaluator] ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"combined_score": 0.0}


if __name__ == "__main__":
    # Lightweight integration test — write initial YAML to a temp file,
    # then call evaluate() exactly as OpenEvolve would.
    import tempfile
    yaml_path = os.path.join(project_dir, "config", "agents_evolving.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            fitness = evaluate(tmp_path)
            print(f"Test execution completed with evaluated fitness score: {fitness}")
        finally:
            os.remove(tmp_path)
