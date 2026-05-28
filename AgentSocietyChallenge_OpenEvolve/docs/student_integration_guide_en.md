# Student Integration Guide — Bringing Your CrewAI Agent Design into the OpenEvolve Framework

> This guide walks you through migrating the Multi-Agent CrewAI system you designed in
> **`AgentSocietyChallenge_w_CrewAI`** (the teaching parent repo) into
> **`AgentSocietyChallenge_OpenEvolve`** (the version integrated with OpenEvolve), and
> then launching an LLM-driven evolutionary optimization of your agent prompts.

---

## 0. The Role of Each Repo

| Repo | Role | What You Do |
|------|------|-------------|
| [`AgentSocietyChallenge_w_CrewAI`](https://github.com/yuchieh/AgentSocietyChallenge_w_CrewAI) | **Design sandbox** | Design, iterate, and test your Multi-Agent Crew |
| [`AgentSocietyChallenge_OpenEvolve`](https://github.com/yuchieh/AgentSocietyChallenge_OpenEvolve) | **Evolution framework** | Move your validated Crew here and let an LLM auto-evolve the best prompts |

The two repos share **the exact same underlying code** (same `websocietysimulator/` competition framework, same CrewAI integration layer). The only difference is that the evolution version wraps everything with an OpenEvolve evaluator.

---

## 1. The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ OpenEvolve (outer layer — exists only in the evolution repo)│
│   - LLM mutates the contents of agents.yaml                 │
│   - Runs evaluator → collects fitness → decides next mutation│
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ passes mutated YAML downward
┌─────────────────────────────────────────────────────────────┐
│ Your CrewAI System (middle layer — present in both repos)   │
│   - config/agents.yaml      ← your Agent role definitions   │
│   - config/tasks.yaml       ← your Task instructions        │
│   - src/crews/...           ← your Crew assembly logic      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ queries data, returns prediction
┌─────────────────────────────────────────────────────────────┐
│ Competition Simulator (bottom — present in both, immutable) │
│   - websocietysimulator/                                    │
│   - InteractionTool: get_user / get_item / get_reviews      │
│   - SimulationEvaluator: computes stars MAE + review sim    │
└─────────────────────────────────────────────────────────────┘
```

**Good news: your CrewAI system needs zero changes to be ported over.** Anything that runs in the parent repo will run in the evolution version too. The only addition is "a layer that can evolve your prompts on top".

> ✅ **Data paths are also identical**: both repos use the same directory layout (`dummy_dataset/`, `dummy_tasks/`, `dummy_groundtruth/`, `review.json`, etc.), so any `data_dir` / `task_dir` paths from your parent-repo code work as-is.

---

## 2. File Correspondence Between the Two Repos (Most Important!)

| Parent Repo (design) | Evolution Repo (execution) | Action |
|----------------------|----------------------------|--------|
| `config/agents.yaml` | `config/agents.yaml` | ✅ **Copy in full** |
| `config/tasks.yaml` | `config/tasks.yaml` | ✅ **Copy in full** |
| `src/crews/simulation_crew.py` | `src/crews/simulation_crew.py` | ✅ **Copy in full** |
| `src/tools/interaction_tool_wrapper.py` | `src/tools/interaction_tool_wrapper.py` | ⛔ Usually untouched |
| `src/flows/serving_flow.py` | `src/flows/serving_flow.py` | ⛔ **Do NOT modify** (OpenEvolve already added the `agents_config_path` mechanism) |
| `crewai_simulation_agent.py` | `crewai_simulation_agent.py` | ⛔ **Do NOT modify** (same reason as above) |
| `run_pipeline.py` / `run_test.py` | `run_test.py` | ⛔ Use the existing one |
| ❌ Does not exist | `config/agents_evolving.yaml` | 🆕 **Create new** (from `agents.yaml` + `EVOLVE-BLOCK` markers) |
| ❌ Does not exist | `config/openevolve_config.yaml` | ⛔ Defaults are usually fine |
| ❌ Does not exist | `openevolve_evaluator.py` | ⛔ Do not touch |

### 2.1 Data Directory Structure (Identical in Both Repos)

Both repos use the same data directory naming, so **bringing data over from the parent repo requires no path/filename changes**:

```
dummy_dataset/
├── item.json                # items / businesses
├── user.json                # users
├── review.json              # historical reviews
├── test_review_subset.json  # sampling source (input for create_sampled_dataset.py)
└── lmdb_cache/              # LMDB index (auto-built on first run; safe to delete)

dummy_tasks/                 # simulation tasks (task_1.json ~ task_N.json)
dummy_groundtruth/           # corresponding groundtruth (groundtruth_*.json)
```

If you generated new data in the parent repo via `create_sampled_dataset.py` or `data_process.py`:

| What you produced in the parent repo | Where it goes in the evolution repo |
|--------------------------------------|--------------------------------------|
| `dummy_tasks/task_*.json` | Same-named `dummy_tasks/` (just copy) |
| `dummy_groundtruth/groundtruth_*.json` | Same-named `dummy_groundtruth/` (just copy) |
| `dummy_dataset/{item,user,review}.json` | Same-named `dummy_dataset/` (just copy; remember to remove the old `lmdb_cache/`) |

> 💡 Paths are hard-coded in `openevolve_evaluator.py`, `run_test.py`, and `src/utils/create_sampled_dataset.py`, but **because the naming matches the parent repo, you don't need to edit those files**.

---

## 3. Complete Migration Workflow (5 Steps)

### Assumptions

- You have already designed a Multi-Agent Crew in the parent repo (e.g., `user_analyst` + `item_analyst` + `prediction_modeler`)
- You have run `uv run python run_pipeline.py` in the parent repo and obtained reasonable metrics

### Step 1: Clone the Evolution Repo and Initialize

```bash
cd ~/WorkSpace/your_workspace/
git clone https://github.com/yuchieh/AgentSocietyChallenge_OpenEvolve.git
cd AgentSocietyChallenge_OpenEvolve

cp .env.example .env
# Edit .env to fill in OPENAI_API_KEY and OPENAI_API_BASE (e.g., NVIDIA NIM)

make install      # uv sync
```

### Step 2: Port Your Agent Design (3 Files)

**Assuming your parent repo lives at** `~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/`:

```bash
# Copy from parent repo into the same locations in the evolution repo
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/config/agents.yaml             config/agents.yaml
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/config/tasks.yaml              config/tasks.yaml
cp ~/WorkSpace/your_workspace/AgentSocietyChallenge_w_CrewAI/src/crews/simulation_crew.py   src/crews/simulation_crew.py
```

### Step 3: Run a Smoke Test to Confirm the Port

```bash
make smoke
```

Expected output:
```
✅ Integration test complete!
overall_quality: 0.7x ~ 0.8x  ← should match what you saw in the parent repo
```

> **If smoke test fails**, go back to the parent repo and verify it still works there, then re-check the file paths in Step 2. **Do not move forward until smoke passes.**

### Step 4: Create `agents_evolving.yaml` — Mark Which Agents Should Evolve

Make a copy of `agents.yaml`:

```bash
cp config/agents.yaml config/agents_evolving.yaml
```

Then edit `config/agents_evolving.yaml` and **use `EVOLVE-BLOCK` markers to wrap the section you want the LLM to mutate**.

#### Rules (Critical!)

| Rule | Why |
|------|-----|
| 🔴 **Exactly one block per file** | OpenEvolve does not accept multiple blocks (it technically runs but converges much slower) |
| 🟢 **Agents WITH tools go OUTSIDE the block** | Their prompts are tightly coupled to the ReAct format in `tasks.yaml`; if mutated, tool calls break |
| 🟢 **Pure reasoning agents go INSIDE the block** | These agents have no tools, just synthesize information into output — perfect targets for evolution |
| 🟢 **Keep `llm:` field inside the block** | The LLM tends to preserve it naturally, but keeping it inside is safer |

#### Example: Parent Repo's 3-Agent Structure

```yaml
# config/agents_evolving.yaml

# === Agents WITH tools stay OUTSIDE EVOLVE-BLOCK (protect tool-calling format) ===
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

# === Pure reasoning agents go INSIDE the EVOLVE-BLOCK ===
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

### Step 5: Launch Evolution

```bash
# Default: 10 iter × 5 tasks
make evolve

# Custom parameters
make evolve ITERS=50 TASKS=3

# Resume from a checkpoint (if a previous run was interrupted)
make evolve-resume CHECKPOINT=config/openevolve_output/checkpoints/checkpoint_10
```

While evolution runs, open another terminal for the visualizer:

```bash
make visualize
# Open browser at http://127.0.0.1:8080
```

---

## 4. Use the Best Evolved Result

```bash
# 1. Check the final best score
cat config/openevolve_output/best/best_program_info.json

# 2. Inspect the best evolved YAML
cat config/openevolve_output/best/best_program.yaml

# 3. Apply it to the production agents.yaml
cp config/openevolve_output/best/best_program.yaml config/agents.yaml

# 4. Run a full real-LLM test with the best version
make test
```

> Note: the `EVOLVE-BLOCK` markers in `agents_evolving.yaml` will appear in
> `best_program.yaml` as YAML comments (`#`-prefixed). They are harmless — CrewAI
> ignores them — but you may delete them for cleanliness when copying to `agents.yaml`.

---

## 5. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `make smoke` returns `overall_quality: 0.27` | LLM API failed, evaluator returned fallback | Verify `.env` API key / base URL; confirm `make test-mock` works (mock mode doesn't use tokens) |
| Evaluator prints `⏱  Simulation exceeded 900s` | A single simulation got stuck on LLM rate limits | Reduce `TASKS=N`, raise `OPENEVOLVE_SIM_TIMEOUT=1800`, or switch API |
| Mutated YAML breaks → all evaluations fail | OpenEvolve uses diff mode by default, which corrupts YAML | Confirm `config/openevolve_config.yaml` has `diff_based_evolution: false` |
| `combined_score` shows no improvement | EVOLVE-BLOCK is too small, search space is limited | Move more agents inside the block, or increase `ITERS` |
| Tool calls fail entirely (agent never invokes tools) | You put a tool-using agent inside EVOLVE-BLOCK and the LLM ruined its role | Move tool-using agents OUTSIDE the block |

---

## 6. Quick Look: Evolution Loop Internals

When you run `make evolve`, here is what happens behind the scenes:

```
[Repeats for ITERS iterations]

  1. OpenEvolve samples a parent YAML from the population
                ▼
  2. LLM reads the parent YAML (including your EVOLVE-BLOCK markers)
     and produces a full-rewrite mutated YAML
                ▼
  3. The mutated YAML is written to /tmp/xxxxx.yaml
                ▼
  4. OpenEvolve calls openevolve_evaluator.py's evaluate(/tmp/xxxxx.yaml)
                ▼
  5. evaluate() sets env var OPENEVOLVE_AGENTS_YAML=/tmp/xxxxx.yaml
                ▼
  6. evaluate() calls simulator.run_simulation(N=TASKS)
                ▼
  7. Simulator spawns a CrewAISimulationAgent per task
                ▼
  8. CrewAISimulationAgent reads the env var → loads /tmp/xxxxx.yaml
     → assembles SimulationCrew with the mutated agents config
     → executes the 3 Tasks
                ▼
  9. Each task produces {stars, review}; Simulator computes metrics
     by comparing against groundtruth
                ▼
 10. evaluate() returns {"combined_score": overall_quality}
                ▼
 11. OpenEvolve writes this fitness into the MAP-Elites database
                ▼
 12. In the next iteration, high-fitness YAMLs are more likely
     to be sampled as parents
```

---

## 7. Additional Resources

| To learn about | See |
|----------------|-----|
| How OpenEvolve works | https://github.com/algorithmicsuperintelligence/openevolve |
| How to write EVOLVE-BLOCK markers | `examples/README.md` inside the OpenEvolve repo |
| Detailed Simulator / InteractionTool API | `docs/legacy/tutorials/agent_development.md` |
| Our specific OpenEvolve integration details | `CLAUDE.md` |

---

## TL;DR — One-Page Cheat Sheet

1. **Clone** `AgentSocietyChallenge_OpenEvolve`, then `cp .env.example .env` and fill in API keys
2. **Copy 3 files** from parent repo into evolution repo: `agents.yaml`, `tasks.yaml`, `simulation_crew.py`
3. **`make smoke`** to verify nothing broke
4. **`cp agents.yaml agents_evolving.yaml`**, edit the copy to wrap pure-reasoning agents in **one** `EVOLVE-BLOCK`
5. **`make evolve ITERS=50 TASKS=3`** to launch evolution
6. After completion: **`cp config/openevolve_output/best/best_program.yaml config/agents.yaml`** to deploy the best version

🎉 Enjoy automatic LLM-driven prompt optimization for your Multi-Agent system.
