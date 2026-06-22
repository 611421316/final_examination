# OpenEvolve-based Optimization for Yelp Review Agent Crew

**Final Examination Project**  
**Course:** LLM & APP  
**Team:** Vo Cong Vinh (611421316), ERIC (611321215)  
**GitHub Repository:** <https://github.com/611421316/final_examination/>

---

## 1. Project Overview

This project optimizes a sequential CrewAI-based Yelp review generation pipeline using OpenEvolve. The system receives a `user_id` and an `item_id`, retrieves exact evidence from local Yelp-style JSON files, computes a deterministic rating score, and then generates a grounded Yelp-style review.

The main idea is not only prompt tuning. The project separates **rating computation** from **review text generation**:

- The Python retrieval/scoring tool computes `predicted_stars`.
- The final LLM agent copies `predicted_stars` exactly.
- The final LLM only writes the review text.
- OpenEvolve searches over agent design, task contracts, and rating-policy parameters.

The best final result came from **Round 2 Agent Evolution**:

| Metric | Score |
|---|---:|
| Combined score | **0.8785** |
| Preference estimation | **0.8800** |
| Review generation | **0.8769** |

---

## 2. Core Novelty

### Deterministic Rating-Scoring Mechanism

The core novelty is a deterministic rating-scoring mechanism that converts raw evidence into `predicted_stars` before the LLM writes the review.

The scoring policy uses three evidence sources:

1. **User evidence**: user rating tendency and profile-level rating behavior.
2. **Item evidence**: business quality and item-level metadata.
3. **Direct review evidence**: exact historical user-item review, when available.

The LLM does **not** recalculate the rating. It only writes a review using the locked rating.

```text
Raw evidence
   ↓
signals
   ↓
confidence-adjusted weights
   ↓
weighted score
   ↓
safety guards
   ↓
predicted_stars
   ↓
final JSON review
```

### Key Message

> The rating policy owns the stars. The LLM writes the review.

---

## 3. Crew Sequential Pipeline

The system uses a sequential CrewAI pipeline:

```text
Input user_id + item_id
        ↓
1. Retrieve + Score
        ↓
2. Expose Case
        ↓
3. Analyze User
        ↓
4. Analyze Item
        ↓
5. Predict Review
```

### Step 1: Retrieve + Score

The data retriever uses lookup tools to read evidence from the dataset and build `prediction_context`.

Internally, it performs:

```text
lookup evidence → detect flags → compute rating
```

It returns:

- `case`
- `user`
- `item`
- `review_style`
- `review_policy`
- `predicted_stars`
- `rating_weight_trace`

### Step 2: Expose Case

This step does not compute the rating again. It only reads the case flags from `prediction_context` and exposes them as compact JSON so later agents know which evidence exists and which claims are safe.

### Step 3: Analyze User

The user analyst converts user history into voice and style signals:

- rating tendency
- sentiment strength
- vocabulary level
- review length
- opening and closing style

### Step 4: Analyze Item

The item analyst converts item and direct-review evidence into grounded review-writing guidance:

- safe item facts
- direct review anchors
- forbidden claims
- tone guidance

### Step 5: Predict Review

The final review simulator outputs only:

```json
{
  "stars": 4.7,
  "review": "..."
}
```

The `stars` value must equal `predicted_stars` exactly.

---

## 4. Dataset and Evidence Sources

The system uses exact local JSON lookup only.

| File | Rows | Main fields used |
|---|---:|---|
| `user.json` | 38 | `user_id`, `average_stars`, `review_count`, `yelping_since`, `elite`, `fans`, `useful`, `funny`, `cool` |
| `item.json` | 432 | `item_id` / `business_id`, `stars`, `review_count`, `name`, `categories`, `attributes`, `city`, `state`, `is_open` |
| `review.json` | 4,164 | `user_id`, `item_id` / `business_id`, `stars`, `text`, `date` |
| `task_i.json` | 5 pairs | input `user_id + item_id` |
| `groundtruth_i.json` | 5 pairs | target stars and reference review |

The pipeline does not use ChromaDB fallback or unsupported facts. Missing evidence is explicitly handled by case routing.

---

## 5. Agent Design

| Agent | Role / Goal | Tool Use / Constraint |
|---|---|---|
| Exact Yelp Data Retriever | Looks up user, item, direct review, user history, and item history; builds `prediction_context` with `predicted_stars` and trace. | Lookup tools allowed only here. |
| Yelp Case Detection Analyst | Classifies evidence availability and exposes the case route. | No extra tools. |
| Yelp User Behavior Analyst | Converts user history into rating tendency, writing style, vocabulary, and structure. | No tools; no star recalculation. |
| Yelp Item Review Context Analyst | Converts item and direct-review evidence into safe facts, anchors, and forbidden claims. | No tools; no invented facts. |
| Deterministic Review Simulator | Writes final JSON review using user voice and grounded item content. | No tools; must copy `predicted_stars`. |

### Boundaries

- **Tool boundary:** only retrieval/scoring uses lookup tools.
- **Evidence boundary:** missing user/item/direct-review evidence cannot be invented.
- **Rating boundary:** final agent writes review text but cannot change `predicted_stars`.

---

## 6. Task Design

| Task | Purpose | Input | Expected Output | Guardrail |
|---|---|---|---|---|
| `analyze_user_behavior_task` | Convert user history into voice and personalization signals. | user profile and history | user average, bias, sentiment map, writing style | no invented preferences |
| `analyze_item_review_context_task` | Convert item and direct-review evidence into safe review-writing guidance. | item, direct review, review policy | safe item evidence, anchors, phrasing constraints | do not change `predicted_stars` |
| `predict_review_task` | Write final JSON review using all analysis outputs. | all analyses and `predicted_stars` | `{stars, review}` | `stars = predicted_stars` exactly |

---

## 7. Rating Policy Parameters

| Group | Main settings | Meaning |
|---|---|---|
| Fallback + rounding | `default_prior`, rounding mode, min/max stars | Use a safe fallback when no evidence exists and standardize rating output. |
| Evidence weights | `base_user_weight`, `base_item_weight`, `direct_review_weight` | Decide which evidence matters most. Direct review is strongest when available. |
| Confidence policy | user/item divisors, bonuses, multiplier range | Increase user/item weight when review counts make the evidence more reliable. |
| History blend | user/item history blend weights and minimum counts | Blend profile averages with actual review history. |
| Anchor policy | user/item anchor review-count thresholds and max deltas | Prevent rating from jumping too far from reliable user/item averages. |
| Anti-saturation | direct review requirement for 5.0, no-direct cap, direct adjustment | Avoid unsupported 5-star predictions and prevent drift away from direct review evidence. |

---

## 8. General Rating Formula

### 8.1 Build Signals

```text
user_signal =
(1 - user_history_blend_weight) × user.average_stars
+ user_history_blend_weight × user_history_average_stars
```

```text
item_signal =
(1 - item_history_blend_weight) × item.stars
+ item_history_blend_weight × item_history_average_stars
```

```text
direct_signal = direct_review_stars
```

### 8.2 Compute Confidence-Adjusted Weights

```text
user_weight =
base_user_weight ×
[1 + user_confidence_bonus × min(user.review_count / user_confidence_divisor, 1)]
```

```text
item_weight =
base_item_weight ×
[1 + item_confidence_bonus × min(item.review_count / item_confidence_divisor, 1)]
```

```text
direct_weight = direct_review_weight
```

### 8.3 Score Available Evidence

```text
score = Σ(signal × weight) / Σ(available weights)
```

If a signal is missing, its weight is removed and the remaining evidence is renormalized.

### 8.4 Apply Safety Guards

Anchor policy:

```text
If user.review_count ≥ user_anchor_min_review_count:
score stays within user.average_stars ± user_anchor_max_delta
```

```text
If item.review_count ≥ item_anchor_min_review_count:
score stays within item.stars ± item_anchor_max_delta
```

Anti-saturation:

```text
If no direct review and require_direct_review_for_5 is true:
score ≤ no_direct_review_max_stars
```

```text
If direct_review_stars < max_stars:
score ≤ direct_review_stars + direct_review_max_adjustment
```

### 8.5 Final Rating

```text
predicted_stars = round_half_up(score, decimal_places)
```

---

## 9. Evidence-Aware Case Routing

| Case | User | Item | Direct Review | Policy |
|---:|---|---|---|---|
| 1 | ✓ | ✓ | ✓ | direct review dominates; user and item calibrate |
| 2 | ✓ | ✓ | ✗ | user + item; no direct-review claim |
| 3 | ✗ | ✓ | ✓ | direct review + item |
| 4 | ✗ | ✓ | ✗ | item only |
| 5 | ✓ | ✗ | ✓ | direct review + user |
| 6 | ✓ | ✗ | ✗ | user only |
| 7 | ✗ | ✗ | ✓ | direct review only |
| 8 | ✗ | ✗ | ✗ | default prior |

This case routing is important because it prevents unsupported claims:

- no fake user preferences
- no fake item facts
- no fake direct-review details
- no claim that the user visited the item when direct review is missing

---

## 10. Review Text Generation Mechanism

The review generation step uses `prediction_context` but does not recalculate stars.

Inputs:

- `predicted_stars`
- case flags
- user evidence
- item evidence
- review style
- review policy

The final simulator follows three contracts:

### Star Lock

```text
stars = predicted_stars exactly
```

### Grounding Rule

Every concrete detail must come from `prediction_context`. Missing evidence means lower specificity.

### Style Rule

Use user voice when available. If user evidence is missing, use neutral Yelp-style wording matched to the locked stars.

Final output schema:

```json
{
  "stars": "<predicted_stars>",
  "review": "<generated review text>"
}
```

No extra fields are allowed.

---

## 11. OpenEvolve Methodology

The project uses a two-round, layer-wise OpenEvolve strategy.

### Round 1

The initial crew is evolved separately in three layers:

```text
R1 Agent Evolution
R1 Task Evolution
R1 Rating Policy Parameter Evolution
```

The best components are then composed into a new baseline.

### Round 2

The composed-best baseline is evolved again:

```text
R2 Agent Evolution
R2 Task Evolution
R2 Rating Policy Parameter Evolution
```

This reveals cross-layer effects after recomposition.

### Run Commands

```bash
# Agent layer
make evolve-agent ITERS=50 TASKS=1

# Task layer
make evolve-task ITERS=50 TASKS=1

# Eval-policy / RPP layer
make evolve-eval ITERS=50 TASKS=1
```

Run settings:

| Setting | Value |
|---|---|
| Model | `minimaxai/minimax-m2.7` |
| Iterations | 50 per run |
| Tasks | 1 sampled pair |
| Timeout | 2000 seconds |
| Targets | agent / task / eval YAML |
| Metrics | preference + review + combined |

---

## 12. Evolution Analysis

### 12.1 Round 1 Agent Evolution

| What changed | Why it helped | Trade-off |
|---|---|---|
| Final simulator became style-aware: openings, detail density, closing patterns. | Review sounds closer to the target user instead of generic Yelp text. | Can feel formulaic if style rules are too rigid. |
| Grounding rule became stricter: use only verified item evidence. | Reduces hallucinated dishes, service issues, names, or unsupported facts. | Sparse evidence leads to less detailed reviews. |
| Added sentiment intensity and leniency/harshness behavior. | Tone better matches user personality while stars stay locked. | May exaggerate user bias when evidence is weak. |

Key message:

> OpenEvolve improved the final writer, not the rating formula. Style became more authentic while `predicted_stars` stayed locked.

### 12.2 Round 1 Task Evolution

| What changed | Why it helped | Trade-off |
|---|---|---|
| User task became signal-based: user average, deviation, bias, style markers. | Gives downstream agents clear behavioral signals. | Less useful when user history is sparse. |
| Item task became review-writing guidance: anchors, constraints, emotional arc. | Converts item/direct evidence into safe wording cues. | Improves review text more than rating accuracy. |
| Final task locked stars and simplified output to raw JSON. | Prevents rating drift and schema errors. | Final agent cannot repair wrong `predicted_stars`. |

Key message:

> Tasks evolved from vague summaries into reusable signal interfaces; the final agent became a constrained JSON writer.

### 12.3 Round 1 RPP / Eval Policy Evolution

| What changed | Why it helped | Trade-off |
|---|---|---|
| Direct review became dominant. | Exact user-item evidence gives the most reliable rating signal. | User/item context has less influence. |
| Prior, anchors, and anti-saturation became stricter. | Avoids unstable jumps and unsupported 5.0 ratings. | More conservative in sparse-evidence cases. |
| Tone and case guidance became more detailed. | Review text better matches stars and evidence case. | Reviews can become more formulaic. |

Key message:

> The evolved eval policy became more stable and evidence-aware, but less flexible when direct review is missing.

### 12.4 Round 2 Agent Evolution

Round 2 Agent Evolution achieved the best final score.

| What changed | Why it helped | Trade-off |
|---|---|---|
| User analyst became a preference and style profiler. | Captures rating bias, sentiment strength, and writing fingerprints. | Can overfit when user history is sparse. |
| Item analyst aligns safe item evidence with user preferences. | Makes personalization grounded instead of generic. | Sparse item evidence may limit detail. |
| Generator prioritizes user fingerprints while keeping `predicted_stars` locked. | Improves review generation and keeps schema stable. | Too many style constraints can make output rigid. |

Key message:

> R2 Agent improved personalization and review quality, but validators must still enforce schema and star lock.

### 12.5 Round 2 Task Evolution

Round 2 Task Evolution produced a lower combined score than Round 2 Agent Evolution.

| What changed | Why it helped | Trade-off |
|---|---|---|
| Tasks output compact signals instead of long summaries. | Gives downstream agents clearer reusable guidance. | Less nuance in user/item analysis. |
| Item/direct evidence became safer wording guidance. | Reduces hallucinated facts and improves grounded review text. | Sparse evidence makes reviews more generic. |
| Final task locks `predicted_stars` and outputs raw JSON only. | Prevents rating drift and schema errors. | Cannot fix wrong or conservative ratings. |

Key message:

> Task evolution made the pipeline more reliable and grounded, but rating quality still depends on the scoring policy.

### 12.6 Round 2 RPP / Eval Policy Evolution

| What changed | Why it helped | Trade-off |
|---|---|---|
| Direct review weight increased. | Exact user-item evidence drives rating and tone. | Less room for user/item context to correct noisy reviews. |
| Confidence, history, and anchor rules became stricter. | Strong evidence is trusted more; weak signals affect less. | Sparse cases may become conservative. |
| Case guidance became direct-review-centered. | Review tone follows the strongest available evidence. | Needs grounding rules to avoid fake details. |

Key message:

> The evolved policy improved rating control, but introduced a calibration trade-off.

---

## 13. Gen-0 vs Evolved Results

| Run | Gen-0 | Best | Δ | Main improvement |
|---|---:|---:|---:|---|
| R1 Agent | 0.8160 | 0.8677 | +0.0517 | specialization |
| R1 Task | 0.8086 | 0.8691 | +0.0605 | structured outputs |
| R1 RPP | 0.8204 | 0.8702 | +0.0498 | direct-review weights |

Pattern:

- The strongest early improvements appeared from better structure and specialization.
- Task evolution gave the largest Round 1 jump.
- Rating policy evolution improved stability and direct-review calibration.

---

## 14. Checkpoint Analysis

Across six runs:

| Statistic | Value |
|---|---:|
| Configured iterations | 300 |
| Retained programs | 252 |
| Not retained / failed candidate attempts | 48 |
| Retention rate | 84% |

Per-run checkpoint statistics:

| Run | Total retained | Not retained |
|---|---:|---:|
| R1 Agent | 40 | 10 |
| R1 Task | 47 | 3 |
| R1 RPP | 42 | 8 |
| R2 Agent | 42 | 8 |
| R2 Task | 42 | 8 |
| R2 RPP | 39 | 11 |

Important note:

> The 48 not-retained candidates should not all be called timeouts. Visualizer data alone cannot prove exact failure causes.

---

## 15. Performance Summary

| Run | Combined | Preference | Review generation |
|---|---:|---:|---:|
| R2 Agent | **0.8785** | 0.8800 | **0.8769** |
| R2 RPP | 0.8715 | 0.8800 | 0.8630 |
| R1 RPP | 0.8702 | 0.8800 | 0.8605 |
| R1 Task | 0.8691 | 0.8800 | 0.8582 |
| R1 Agent | 0.8677 | 0.8800 | 0.8554 |
| R2 Task | 0.8634 | 0.8800 | 0.8468 |

Selected final run:

> Round 2 Agent Evolution after composing best Round 1 components.

Key interpretation:

- Preference stayed stable at 0.8800.
- Main gain came from review generation.
- R2 Task is a useful negative result.
- The final winner is R2 Agent.

---

## 16. Metric Interpretation

The preference metric is calculated as:

```text
preference = 1 - |gt_stars - predicted_stars| / 5
```

Example:

```text
1 - |5.0 - 4.4| / 5 = 0.8800
```

Why preference stayed at 0.8800:

1. The final agent must copy `predicted_stars` exactly.
2. Anti-saturation avoids unsupported 5.0 predictions.
3. Rating calibration is conservative.
4. Review generation can improve without changing rating preference.

Future improvement should safely tune rating calibration, not only make prettier reviews.

---

## 17. Limitations and Future Work

### Current Limitations

- Preference is capped at 0.8800 in the current setting.
- `TASKS=1` is noisy because each run optimizes on one sampled pair.
- Visualizer cannot identify exact failure causes without logs.

### Contract Controls

- Output schema is locked.
- Star value is locked.
- Validators enforce the final output contract.

### Future Work

- Use `TASKS=5` or k-fold task sampling.
- Add log-based failure diagnosis.
- Improve safe rating calibration.
- Compare robustness across multiple sampled user-item cases.

---

## 18. Conclusion

This project demonstrates a sequential CrewAI Yelp review pipeline optimized with OpenEvolve. The key contribution is the separation of rating computation and review generation.

The deterministic rating policy converts evidence into `predicted_stars`, while the LLM focuses on grounded and personalized review text. OpenEvolve improves the system by refining agents, task contracts, and rating-policy parameters. The best final result is achieved by Round 2 Agent Evolution, which improves review quality while preserving the locked rating contract.

---

## 19. Repository

GitHub Repository:

<https://github.com/611421316/final_examination/>

