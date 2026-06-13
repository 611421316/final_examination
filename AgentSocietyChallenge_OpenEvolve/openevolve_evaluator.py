import os
import re
import tempfile
import sys
import logging
import yaml
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


# ---------------------------------------------------------------------------
# Multi-file bundle helpers
# ---------------------------------------------------------------------------

def _is_bundle(program_path: str) -> bool:
    """Return True if the program file is a multi-section evolve bundle."""
    try:
        with open(program_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ("=== SECTION: agents ===" in content and
                "=== SECTION: tasks ===" in content and
                "=== SECTION: crew ===" in content)
    except Exception:
        return False


def _extract_section(content: str, section_name: str) -> str:
    """
    Extract text between '=== SECTION: <name> ===' and the next '=== SECTION:' or end of EVOLVE block.
    Returns the extracted text stripped of leading/trailing whitespace.
    """
    # Match from the section header to the next section header or EVOLVE-BLOCK-END
    pattern = (
        r"# === SECTION: " + re.escape(section_name) + r" ===\n"
        r"(.*?)"
        r"(?=# === SECTION:|# EVOLVE-BLOCK-END|\Z)"
    )
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        raise ValueError(f"Section '{section_name}' not found in bundle.")
    return match.group(1).strip()


def _unpack_bundle(bundle_path: str) -> tuple:
    """
    Parse the multi-file bundle and write three temp files.
    Returns (agents_yaml_path, tasks_yaml_path, crew_config_json_path, lookup_text, flow_text, interaction_text).
    All temp files are written with delete=False so the caller must clean them up.
    """
    with open(bundle_path, "r", encoding="utf-8") as f:
        content = f.read()

    # ── Extract sections ──────────────────────────────────────────────────
    agents_text = _extract_section(content, "agents")
    tasks_text  = _extract_section(content, "tasks")
    crew_text   = _extract_section(content, "crew")

    try:
        lookup_text = _extract_section(content, "lookup")
    except ValueError:
        lookup_text = None

    try:
        flow_text = _extract_section(content, "flow")
    except ValueError:
        flow_text = None

    try:
        interaction_text = _extract_section(content, "interaction")
    except ValueError:
        interaction_text = None

    # ── Validate YAML sections ────────────────────────────────────────────
    try:
        yaml.safe_load(agents_text)
    except yaml.YAMLError as e:
        raise ValueError(f"[Bundle] agents section is not valid YAML: {e}")

    try:
        yaml.safe_load(tasks_text)
    except yaml.YAMLError as e:
        raise ValueError(f"[Bundle] tasks section is not valid YAML: {e}")

    # ── Extract CREW_CONFIG dict from crew section ────────────────────────
    # We allow the crew section to be either valid YAML or a Python-style
    # CREW_CONFIG = {...} literal. We try both.
    crew_config = None
    # Try Python literal eval of the CREW_CONFIG assignment
    crew_match = re.search(r"CREW_CONFIG\s*=\s*(\{.*?\})\s*$", crew_text, re.DOTALL | re.MULTILINE)
    if crew_match:
        import ast
        try:
            crew_config = ast.literal_eval(crew_match.group(1))
        except Exception:
            crew_config = None
    # Fallback: try YAML
    if crew_config is None:
        try:
            crew_config = yaml.safe_load(crew_text)
        except Exception:
            crew_config = None

    # ── Write temp files ──────────────────────────────────────────────────
    agents_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    agents_tmp.write(agents_text)
    agents_tmp.close()

    tasks_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tasks_tmp.write(tasks_text)
    tasks_tmp.close()

    import json
    crew_tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(crew_config or {}, crew_tmp)
    crew_tmp.close()

    return agents_tmp.name, tasks_tmp.name, crew_tmp.name, lookup_text, flow_text, interaction_text


def evaluate(program_path: str) -> dict:
    """
    Module-level function required by OpenEvolve.

    Supports two modes:
      1. Single-file mode (legacy): program_path points to an agents YAML file.
         Behaves exactly as before.
      2. Multi-file bundle mode: program_path points to a evolve_bundle.py file
         containing === SECTION: agents ===, === SECTION: tasks ===, and
         === SECTION: crew === blocks.
         The evaluator unpacks the bundle into temp files and sets the
         appropriate env vars before running the simulation.

    Returns a dict with 'combined_score' as the primary fitness metric (required
    by OpenEvolve), plus individual sub-metrics for MAP-Elites feature tracking.

    combined_score = overall_quality (0-1):
      overall_quality = (preference_estimation + review_generation) / 2
    where preference_estimation = 1 - normalized_star_MAE.
    """
    simulator = _get_simulator()

    # Temp file paths to clean up after evaluation
    _tmp_files = []
    _orig_files = {}

    try:
        if _is_bundle(program_path):
            print(f"[Evaluator] Detected multi-file bundle: {program_path}")
            agents_tmp, tasks_tmp, crew_tmp, lookup_text, flow_text, interaction_text = _unpack_bundle(program_path)
            _tmp_files.extend([agents_tmp, tasks_tmp, crew_tmp])

            # Set env vars so CrewAISimulationAgent / SimulationCrew pick them up
            os.environ["OPENEVOLVE_AGENTS_YAML"] = agents_tmp
            os.environ["OPENEVOLVE_TASKS_YAML"]  = tasks_tmp
            os.environ["OPENEVOLVE_CREW_JSON"]   = crew_tmp
            print(f"[Evaluator] Bundle unpacked → agents={agents_tmp}, tasks={tasks_tmp}, crew={crew_tmp}")

            if lookup_text:
                lookup_path = os.path.join(project_dir, "src", "tools", "exact_lookup_tools.py")
                with open(lookup_path, "r", encoding="utf-8") as f:
                    _orig_files[lookup_path] = f.read()
                with open(lookup_path, "w", encoding="utf-8") as f:
                    f.write(lookup_text)
                print(f"[Evaluator] Injected src/tools/exact_lookup_tools.py from bundle")

            if flow_text:
                flow_path = os.path.join(project_dir, "src", "flows", "serving_flow.py")
                with open(flow_path, "r", encoding="utf-8") as f:
                    _orig_files[flow_path] = f.read()
                with open(flow_path, "w", encoding="utf-8") as f:
                    f.write(flow_text)
                print(f"[Evaluator] Injected src/flows/serving_flow.py from bundle")

            if interaction_text:
                interaction_path = os.path.join(project_dir, "src", "tools", "interaction_tool_wrapper.py")
                with open(interaction_path, "r", encoding="utf-8") as f:
                    _orig_files[interaction_path] = f.read()
                with open(interaction_path, "w", encoding="utf-8") as f:
                    f.write(interaction_text)
                print(f"[Evaluator] Injected src/tools/interaction_tool_wrapper.py from bundle")

        else:
            # Legacy single-file mode (agents YAML only)
            os.environ["OPENEVOLVE_AGENTS_YAML"] = program_path
            os.environ.pop("OPENEVOLVE_TASKS_YAML", None)
            os.environ.pop("OPENEVOLVE_CREW_JSON", None)

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

    finally:
        # Restore original source files
        for path, content in _orig_files.items():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError:
                pass

        # Clean up temp files created during bundle unpacking
        for tmp_path in _tmp_files:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # Lightweight integration test — supports both bundle and legacy YAML mode.
    import sys

    # Default: use bundle if it exists, else fall back to agents_evolving.yaml
    bundle_path = os.path.join(project_dir, "config", "evolve_bundle.py")
    legacy_path = os.path.join(project_dir, "config", "agents_evolving.yaml")

    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    elif os.path.exists(bundle_path):
        test_path = bundle_path
    else:
        test_path = legacy_path

    if os.path.exists(test_path):
        print(f"[Test] Running evaluate() with: {test_path}")
        fitness = evaluate(test_path)
        print(f"Test execution completed with evaluated fitness score: {fitness}")
    else:
        print(f"[Test] File not found: {test_path}")
