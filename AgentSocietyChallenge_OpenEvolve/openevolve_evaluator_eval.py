import os
import tempfile
import sys
import logging
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

# 整個 simulation 的 hard timeout（秒）。超時則回傳 fallback fitness 讓 OpenEvolve 繼續。
# 預設 15 分鐘，可由 OPENEVOLVE_SIM_TIMEOUT env var 覆寫。
SIM_TIMEOUT_SEC = int(os.environ.get("OPENEVOLVE_SIM_TIMEOUT", 2000))

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


def _list_task_files(task_dir: str):
    """Return sorted regular task files from task_dir."""
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    files = [
        os.path.join(task_dir, name)
        for name in os.listdir(task_dir)
        if not name.startswith(".") and os.path.isfile(os.path.join(task_dir, name))
    ]
    files.sort()

    if not files:
        raise FileNotFoundError(f"No task files found in {task_dir}")

    return files


def _find_groundtruth_file(task_path: str, groundtruth_dir: str) -> str:
    """
    Find the matching groundtruth file for a selected task file.

    Supported mappings:
      1. Same filename:
         dummy_tasks/task_1.json -> dummy_groundtruth/task_1.json

      2. task_* -> groundtruth_*:
         dummy_tasks/task_1.json -> dummy_groundtruth/groundtruth_1.json

      3. Same numeric suffix:
         task_1.json -> groundtruth_1.json / gt_1.json / any file containing 1
    """
    task_name = os.path.basename(task_path)

    same_name = os.path.join(groundtruth_dir, task_name)
    if os.path.exists(same_name):
        return same_name

    candidate_names = []
    if "task" in task_name:
        candidate_names.append(task_name.replace("task", "groundtruth", 1))
        candidate_names.append(task_name.replace("task", "gt", 1))

    for candidate_name in candidate_names:
        candidate_path = os.path.join(groundtruth_dir, candidate_name)
        if os.path.exists(candidate_path):
            return candidate_path

    # Fallback: match by the last number in the task filename.
    import re
    task_numbers = re.findall(r"\d+", task_name)
    if task_numbers:
        task_id = task_numbers[-1]
        for name in sorted(os.listdir(groundtruth_dir)):
            if name.startswith("."):
                continue
            path = os.path.join(groundtruth_dir, name)
            if os.path.isfile(path) and task_id in re.findall(r"\d+", name):
                return path

    raise FileNotFoundError(
        f"No matching groundtruth file found for task '{task_name}' in {groundtruth_dir}"
    )


def _prepare_random_task_dirs(
    task_dir: str = "dummy_tasks",
    groundtruth_dir: str = "dummy_groundtruth",
    num_tasks: int = 1,
):
    """
    Randomly select num_tasks task files, then copy the selected task files and
    their matching groundtruth files into temporary directories.

    This prevents simulator.run_simulation(number_of_tasks=1) from always using
    the first task file.
    """
    task_files = _list_task_files(task_dir)
    k = min(max(int(num_tasks), 1), len(task_files))

    seed = os.environ.get("OPENEVOLVE_TASK_SEED")
    rng = random.Random(int(seed)) if seed is not None else random.Random()

    selected_task_files = rng.sample(task_files, k=k)

    temp_task_dir = tempfile.mkdtemp(prefix="openevolve_tasks_")
    temp_gt_dir = tempfile.mkdtemp(prefix="openevolve_groundtruth_")

    print("[Evaluator] Random selected task file(s):")
    for task_path in selected_task_files:
        task_name = os.path.basename(task_path)
        gt_path = _find_groundtruth_file(task_path, groundtruth_dir)
        gt_name = os.path.basename(gt_path)

        shutil.copy2(task_path, os.path.join(temp_task_dir, task_name))
        shutil.copy2(gt_path, os.path.join(temp_gt_dir, gt_name))

        print(f" - task={task_name} | groundtruth={gt_name}")

    print(f"[Evaluator] Temp task dir: {temp_task_dir}")
    print(f"[Evaluator] Temp groundtruth dir: {temp_gt_dir}")
    print(f"[Evaluator] Temp tasks: {sorted(os.listdir(temp_task_dir))}")
    print(f"[Evaluator] Temp groundtruth: {sorted(os.listdir(temp_gt_dir))}")

    return temp_task_dir, temp_gt_dir


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
    temp_task_dir = None
    temp_gt_dir = None

    try:
        # 1. Tell CrewAISimulationAgent to load this YAML config for the run
        os.environ["OPENEVOLVE_EVAL_YAML"] = program_path

        num_tasks = int(os.environ.get("OPENEVOLVE_NUM_TASKS", 5))

        # Randomize the selected task files before passing them to the simulator.
        # Without this, TASKS=1 usually evaluates only the first task.
        temp_task_dir, temp_gt_dir = _prepare_random_task_dirs(
            task_dir=os.environ.get("OPENEVOLVE_TASK_DIR", "dummy_tasks"),
            groundtruth_dir=os.environ.get("OPENEVOLVE_GROUNDTRUTH_DIR", "dummy_groundtruth"),
            num_tasks=num_tasks,
        )

        simulator.set_task_and_groundtruth(
            task_dir=temp_task_dir,
            groundtruth_dir=temp_gt_dir,
        )

        print(
            f"\n[Evaluator] Running simulation: {program_path} "
            f"(random_tasks={num_tasks}, timeout={SIM_TIMEOUT_SEC}s)"
        )

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
            return {
                "combined_score": 0.0,
                "preference_estimation": 0.0,
                "review_generation": 0.0,
            }

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

        return {
            "combined_score": float(overall_quality),
            "preference_estimation": float(pref_estimation),
            "review_generation": float(review_generation),
        }

    except Exception as e:
        print(f"[Evaluator] ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "combined_score": 0.0,
            "preference_estimation": 0.0,
            "review_generation": 0.0,
        }

    finally:
        if temp_task_dir and os.path.exists(temp_task_dir):
            shutil.rmtree(temp_task_dir, ignore_errors=True)

        if temp_gt_dir and os.path.exists(temp_gt_dir):
            shutil.rmtree(temp_gt_dir, ignore_errors=True)


if __name__ == "__main__":
    # Lightweight integration test — write initial YAML to a temp file,
    # then call evaluate() exactly as OpenEvolve would.
    import tempfile
    yaml_path = os.path.join(project_dir, "config", "eval_evolving.yaml")
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
