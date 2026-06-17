"""
Exact lookup tools for Yelp data.

This version uses ONLY these three files:

- src/data/filtered_user.json
- src/data/filtered_item.json
- src/data/train_review.json

No dummy_dataset.
No old data/user_subset.json.
No ChromaDB fallback.

Main rule:
- Python/tool retrieves exact data and computes predicted_stars.
- LLM agents must use predicted_stars exactly.
"""

import json
import os
import statistics
from pathlib import Path
from typing import Any

from crewai.tools import tool
import yaml


# =============================================================================
# Project paths
# =============================================================================

def _find_project_root() -> Path:
    """
    Find project root by locating src/data.
    Expected structure:

    AgentSocietyChallenge_OpenEvolve/
    └── src/
        └── data/
            ├── filtered_user.json
            ├── filtered_item.json
            └── train_review.json
    """
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "src" / "data").exists():
            return parent

    cwd = Path.cwd()
    if (cwd / "src" / "data").exists():
        return cwd

    return current.parent


_PROJECT_ROOT = _find_project_root()
_DATA_DIR = _PROJECT_ROOT / "src" / "data"

_USER_JSON_PATH = _DATA_DIR / "filtered_user.json"
_ITEM_JSON_PATH = _DATA_DIR / "filtered_item.json"
_REVIEW_JSON_PATH = _DATA_DIR / "train_review.json"


# =============================================================================
# Basic helpers
# =============================================================================

_EVAL_POLICY_CACHE = None
_EVAL_POLICY_CACHE_PATH = None


def _get_eval_policy_path() -> Path:
    """
    During OpenEvolve:
        OPENEVOLVE_EVAL_YAML=/tmp/tmpxxxxx.yaml

    Normal run:
        config/eval_evolving.yaml
    """
    env_path = os.environ.get("OPENEVOLVE_EVAL_YAML")
    if env_path:
        return Path(env_path)

    return _PROJECT_ROOT / "config" / "eval_evolving.yaml"


def _load_eval_policy() -> dict:
    global _EVAL_POLICY_CACHE
    global _EVAL_POLICY_CACHE_PATH

    path = _get_eval_policy_path()

    if _EVAL_POLICY_CACHE is not None and _EVAL_POLICY_CACHE_PATH == str(path):
        return _EVAL_POLICY_CACHE

    if not path.exists():
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = str(path)
        return _EVAL_POLICY_CACHE

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            data = {}

        _EVAL_POLICY_CACHE = data
        _EVAL_POLICY_CACHE_PATH = str(path)
        return data

    except Exception as e:
        print(f"[EVAL POLICY WARNING] Cannot load {path}: {e}", flush=True)
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = str(path)
        return {}

_EVAL_POLICY_CACHE = None
_EVAL_POLICY_CACHE_PATH = None


def _get_eval_policy_path() -> Path:
    """
    During OpenEvolve:
        OPENEVOLVE_EVAL_YAML=/tmp/tmpxxxxx.yaml

    Normal run:
        config/eval_evolving.yaml
    """
    env_path = os.environ.get("OPENEVOLVE_EVAL_YAML")
    if env_path:
        return Path(env_path)

    return _PROJECT_ROOT / "config" / "eval_evolving.yaml"


def _load_eval_policy() -> dict:
    global _EVAL_POLICY_CACHE
    global _EVAL_POLICY_CACHE_PATH

    path = _get_eval_policy_path()

    if _EVAL_POLICY_CACHE is not None and _EVAL_POLICY_CACHE_PATH == str(path):
        return _EVAL_POLICY_CACHE

    if not path.exists():
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = str(path)
        return _EVAL_POLICY_CACHE

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            data = {}

        _EVAL_POLICY_CACHE = data
        _EVAL_POLICY_CACHE_PATH = str(path)
        return data

    except Exception as e:
        print(f"[EVAL POLICY WARNING] Cannot load {path}: {e}", flush=True)
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = str(path)
        return {}

def _policy_get(path: list[str], default: Any = None) -> Any:
    data = _load_eval_policy()

    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]

    return cur

def _json_loads_safe(value: Any, default: Any = None) -> Any:
    if isinstance(value, (dict, list)):
        return value

    if not isinstance(value, str):
        return default

    try:
        return json.loads(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def _clamp(value: float, low: float = 1.0, high: float = 5.0) -> float:
    return max(low, min(high, value))


def _round_half(value: float) -> float:
    return round(value * 2.0) / 2.0


def _rating_tendency(avg: float) -> str:
    if avg <= 3.2:
        return "harsh"
    if avg >= 4.2:
        return "generous"
    return "moderate"


def _compact_text(text: str, limit: int = 380) -> str:
    text = str(text or "").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text[:limit]


def _word_count(text: str) -> int:
    return len(str(text or "").split())


def _item_value(obj: dict) -> str:
    """
    Support both internal item_id and Yelp original business_id.
    """
    if not isinstance(obj, dict):
        return ""
    return obj.get("item_id") or obj.get("business_id") or ""


def _matches_item(obj: dict, item_id: str) -> bool:
    return _item_value(obj) == item_id


def _matches_user(obj: dict, user_id: str) -> bool:
    return isinstance(obj, dict) and obj.get("user_id") == user_id


def _normalize_item_id(obj: dict) -> dict:
    """
    Normalize business_id to item_id for downstream pipeline consistency.
    """
    if isinstance(obj, dict) and "item_id" not in obj and "business_id" in obj:
        obj["item_id"] = obj["business_id"]
    return obj


def _iter_json_records(path: Path):
    """
    Read JSONL or JSON array.

    Supports:
    - one JSON object per line
    - a full JSON array file
    """
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        # JSON array file
        if first == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                return
            except Exception:
                return

        # JSONL file
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if isinstance(obj, dict):
                yield obj


def _exact_file_search(path: Path, key: str, target_id: str) -> dict:
    """
    Exact lookup by key in one JSON/JSONL file.
    """
    if not path.exists():
        return {}

    for obj in _iter_json_records(path):
        if obj.get(key) == target_id:
            return obj

    return {}


# =============================================================================
# Exact lookup functions
# =============================================================================

def _get_user_exact_dict(user_id: str) -> dict:
    """
    Look up user only from src/data/filtered_user.json.
    """
    uid = user_id.strip().strip("'\"")

    data = _exact_file_search(_USER_JSON_PATH, "user_id", uid)
    if data:
        return data

    return {}


def _get_item_exact_dict(item_id: str) -> dict:
    """
    Look up item only from src/data/filtered_item.json.

    Supports both:
    - item_id
    - business_id
    """
    iid = item_id.strip().strip("'\"")

    data = _exact_file_search(_ITEM_JSON_PATH, "item_id", iid)

    if not data:
        data = _exact_file_search(_ITEM_JSON_PATH, "business_id", iid)

    if data:
        return _normalize_item_id(data)

    return {}


def _lookup_reviews_by_user_and_item_impl(user_id: str = "", item_id: str = "") -> str:
    """
    Look up reviews only from src/data/train_review.json.

    Can search by:
    - user_id only
    - item_id/business_id only
    - exact user_id + item_id/business_id pair
    """
    uid = user_id.strip().strip("'\"") if user_id else ""
    iid = item_id.strip().strip("'\"") if item_id else ""

    if not uid and not iid:
        return "Error: must provide at least one of user_id or item_id."

    results: list[dict] = []

    if not _REVIEW_JSON_PATH.exists():
        return f"Review file not found: {_REVIEW_JSON_PATH}"

    for r in _iter_json_records(_REVIEW_JSON_PATH):
        if uid and iid:
            if _matches_user(r, uid) and _matches_item(r, iid):
                results.append(_normalize_item_id(r))

        elif uid:
            if _matches_user(r, uid):
                results.append(_normalize_item_id(r))

        elif iid:
            if _matches_item(r, iid):
                results.append(_normalize_item_id(r))

        if len(results) >= 60:
            break

    # Deduplicate
    seen = set()
    deduped = []

    for r in results:
        key = (
            r.get("review_id", ""),
            r.get("user_id", ""),
            _item_value(r),
            r.get("date", ""),
            str(r.get("text", ""))[:50],
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(r)

    deduped = sorted(
        deduped,
        key=lambda x: str(x.get("date", "")),
        reverse=True,
    )[:60]

    if not deduped:
        return f"No reviews found for user_id={uid!r}, item_id={iid!r}"

    return json.dumps(deduped, ensure_ascii=False)


def _get_reviews_list(user_id: str, item_id: str) -> list[dict]:
    raw = _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id=item_id)
    parsed = _json_loads_safe(raw, [])

    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]

    return []


# =============================================================================
# Review style and rating logic
# =============================================================================

def _summarize_review_style(reviews: list[dict], user_id: str, item_id: str) -> dict:
    uid = user_id.strip().strip("'\"")
    iid = item_id.strip().strip("'\"")

    direct_reviews = [
        r for r in reviews
        if _matches_user(r, uid) and _matches_item(r, iid)
    ]

    user_reviews = [
        r for r in reviews
        if _matches_user(r, uid)
    ]

    item_reviews = [
        r for r in reviews
        if _matches_item(r, iid)
    ]

    # Prefer user writing style. If unavailable, use direct review. If still empty, use related reviews.
    style_source = user_reviews if user_reviews else direct_reviews
    if not style_source:
        style_source = reviews[:10]

    word_counts = [
        _word_count(r.get("text", ""))
        for r in style_source
        if r.get("text")
    ]

    user_ratings = [
        _as_float(r.get("stars"), 0.0)
        for r in user_reviews
        if r.get("stars") is not None
    ]
    user_ratings = [x for x in user_ratings if 1.0 <= x <= 5.0]

    item_ratings = [
        _as_float(r.get("stars"), 0.0)
        for r in item_reviews
        if r.get("stars") is not None
    ]
    item_ratings = [x for x in item_ratings if 1.0 <= x <= 5.0]

    examples = []
    for r in sorted(style_source, key=lambda x: str(x.get("date", "")), reverse=True)[:3]:
        examples.append(
            {
                "stars": _as_float(r.get("stars"), 3.8),
                "date": r.get("date", ""),
                "text_excerpt": _compact_text(r.get("text", ""), 360),
            }
        )

    direct = direct_reviews[0] if direct_reviews else None

    return {
        "direct_review_exists": bool(direct),
        "direct_review_stars": _as_float(direct.get("stars"), 3.8) if direct else None,
        "direct_review_excerpt": _compact_text(direct.get("text", ""), 500) if direct else "",
        "user_review_count_used": len(user_reviews),
        "item_review_count_used": len(item_reviews),
        "avg_word_count": int(statistics.mean(word_counts)) if word_counts else 60,
        "median_word_count": int(statistics.median(word_counts)) if word_counts else 60,
        "user_history_average_stars": round(statistics.mean(user_ratings), 3) if user_ratings else None,
        "item_history_average_stars": round(statistics.mean(item_ratings), 3) if item_ratings else None,
        "style_examples": examples,
    }


def _compute_stars(user: dict, item: dict, style: dict) -> float:
    """
    Deterministic predicted stars controlled by config/eval_evolving.yaml.

    OpenEvolve mutates eval_evolving.yaml.
    This function loads that policy and applies the evolved parameters.
    """
    default_prior = _as_float(
        _policy_get(["rating_policy", "default_prior"], 3.8),
        3.8,
    )

    min_stars = _as_float(
        _policy_get(["rating_policy", "rounding", "min_stars"], 1.0),
        1.0,
    )
    max_stars = _as_float(
        _policy_get(["rating_policy", "rounding", "max_stars"], 5.0),
        5.0,
    )

    direct_exists = bool(style.get("direct_review_exists"))
    direct_stars = style.get("direct_review_stars")

    # Case 1, 3, 5, 7
    # Direct review stars remain strongest evidence.
    if direct_exists and direct_stars is not None:
        return float(_round_half(_clamp(_as_float(direct_stars, default_prior), min_stars, max_stars)))

    user_exists = bool(user)
    item_exists = bool(item)

    user_avg = _as_float(user.get("average_stars"), default_prior) if user_exists else default_prior
    item_avg = _as_float(item.get("stars"), default_prior) if item_exists else default_prior

    user_count = _as_int(user.get("review_count"), 0) if user_exists else 0
    item_count = _as_int(item.get("review_count"), 0) if item_exists else 0

    # Case 8
    if not user_exists and not item_exists:
        return float(_round_half(_clamp(default_prior, min_stars, max_stars)))

    # Case 6
    if user_exists and not item_exists:
        return float(_round_half(_clamp(user_avg, min_stars, max_stars)))

    # Case 4
    if not user_exists and item_exists:
        return float(_round_half(_clamp(item_avg, min_stars, max_stars)))

    # Case 2: evolved weighted blend
    case2_path = ["rating_policy", "case_rules", "case_2"]

    base_user_weight = _as_float(
        _policy_get(case2_path + ["base_user_weight"], 0.65),
        0.65,
    )
    base_item_weight = _as_float(
        _policy_get(case2_path + ["base_item_weight"], 0.35),
        0.35,
    )

    user_confidence_divisor = _as_float(
        _policy_get(case2_path + ["user_confidence_divisor"], 50.0),
        50.0,
    )
    item_confidence_divisor = _as_float(
        _policy_get(case2_path + ["item_confidence_divisor"], 100.0),
        100.0,
    )

    user_confidence_bonus = _as_float(
        _policy_get(case2_path + ["user_confidence_bonus"], 0.25),
        0.25,
    )
    item_confidence_bonus = _as_float(
        _policy_get(case2_path + ["item_confidence_bonus"], 0.20),
        0.20,
    )

    historical_user_review_blend_weight = _as_float(
        _policy_get(case2_path + ["historical_user_review_blend_weight"], 0.30),
        0.30,
    )
    historical_user_review_min_count = _as_int(
        _policy_get(case2_path + ["historical_user_review_min_count"], 3),
        3,
    )

    user_anchor_min_review_count = _as_int(
        _policy_get(case2_path + ["user_anchor_min_review_count"], 30),
        30,
    )
    user_anchor_max_delta = _as_float(
        _policy_get(case2_path + ["user_anchor_max_delta"], 0.70),
        0.70,
    )

    # Safety clamps so evolution cannot produce extreme broken values.
    base_user_weight = _clamp(base_user_weight, 0.05, 5.0)
    base_item_weight = _clamp(base_item_weight, 0.05, 5.0)

    user_confidence_divisor = max(user_confidence_divisor, 1.0)
    item_confidence_divisor = max(item_confidence_divisor, 1.0)

    user_confidence_bonus = _clamp(user_confidence_bonus, 0.0, 3.0)
    item_confidence_bonus = _clamp(item_confidence_bonus, 0.0, 3.0)

    historical_user_review_blend_weight = _clamp(
        historical_user_review_blend_weight,
        0.0,
        0.9,
    )

    user_anchor_max_delta = _clamp(user_anchor_max_delta, 0.0, 4.0)

    user_conf = min(user_count / user_confidence_divisor, 1.0)
    item_conf = min(item_count / item_confidence_divisor, 1.0)

    user_weight = base_user_weight + user_confidence_bonus * user_conf
    item_weight = base_item_weight + item_confidence_bonus * item_conf

    denominator = user_weight + item_weight
    if denominator <= 0:
        rating = default_prior
    else:
        rating = (user_weight * user_avg + item_weight * item_avg) / denominator

    hist_avg = style.get("user_history_average_stars")
    user_review_count_used = _as_int(style.get("user_review_count_used"), 0)

    if hist_avg is not None and user_review_count_used >= historical_user_review_min_count:
        hist_avg = _as_float(hist_avg, rating)
        w = historical_user_review_blend_weight
        rating = (1.0 - w) * rating + w * hist_avg

    if user_count >= user_anchor_min_review_count:
        rating = max(
            user_avg - user_anchor_max_delta,
            min(user_avg + user_anchor_max_delta, rating),
        )

    return float(_round_half(_clamp(rating, min_stars, max_stars)))


# =============================================================================
# 8-case detection
# =============================================================================

def _detect_case_from_flags(
    user_exists: bool,
    item_exists: bool,
    direct_review_exists: bool,
) -> dict:
    if user_exists and item_exists and direct_review_exists:
        return {
            "case_number": 1,
            "case_name": "Case 1: user exists, item exists, direct historical review exists",
            "dominant_evidence": "direct_review",
            "fallback_policy": "use direct review first, then user and item context",
        }

    if user_exists and item_exists and not direct_review_exists:
        return {
            "case_number": 2,
            "case_name": "Case 2: user exists, item exists, no direct historical review",
            "dominant_evidence": "user_and_item_profiles",
            "fallback_policy": "combine user behavior and item quality",
        }

    if not user_exists and item_exists and direct_review_exists:
        return {
            "case_number": 3,
            "case_name": "Case 3: user missing, item exists, direct historical review exists",
            "dominant_evidence": "direct_review_and_item_profile",
            "fallback_policy": "use direct review first, then item context",
        }

    if not user_exists and item_exists and not direct_review_exists:
        return {
            "case_number": 4,
            "case_name": "Case 4: user missing, item exists, no direct historical review",
            "dominant_evidence": "item_profile",
            "fallback_policy": "use item stars and category context",
        }

    if user_exists and not item_exists and direct_review_exists:
        return {
            "case_number": 5,
            "case_name": "Case 5: user exists, item missing, direct historical review exists",
            "dominant_evidence": "direct_review_and_user_profile",
            "fallback_policy": "use direct review first, then user behavior",
        }

    if user_exists and not item_exists and not direct_review_exists:
        return {
            "case_number": 6,
            "case_name": "Case 6: user exists, item missing, no direct historical review",
            "dominant_evidence": "user_profile",
            "fallback_policy": "use user rating tendency and avoid item-specific details",
        }

    if not user_exists and not item_exists and direct_review_exists:
        return {
            "case_number": 7,
            "case_name": "Case 7: user missing, item missing, direct historical review exists",
            "dominant_evidence": "direct_review_only",
            "fallback_policy": "use direct review evidence only and avoid unsupported profile or item facts",
        }

    return {
        "case_number": 8,
        "case_name": "Case 8: user missing, item missing, no direct historical review",
        "dominant_evidence": "default_prior",
        "fallback_policy": "use generic fallback and avoid unsupported details",
    }


# =============================================================================
# CrewAI tools
# =============================================================================

@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """
    Look up a user's complete profile by exact user_id.
    Source: src/data/filtered_user.json
    """
    uid = user_id.strip().strip("'\"")
    data = _get_user_exact_dict(uid)

    if data:
        return json.dumps(data, ensure_ascii=False)

    return f"No user found with user_id: {uid}"


@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """
    Look up a business/item profile by exact item_id or business_id.
    Source: src/data/filtered_item.json
    """
    iid = item_id.strip().strip("'\"")
    data = _get_item_exact_dict(iid)

    if data:
        return json.dumps(data, ensure_ascii=False)

    return f"No item found with item_id: {iid}"


@tool("lookup_reviews_by_user_and_item")
def lookup_reviews_by_user_and_item(user_id: str = "", item_id: str = "") -> str:
    """
    Look up historical reviews by exact user_id and/or item_id.
    Source: src/data/train_review.json
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id=item_id)


@tool("lookup_reviews_by_user")
def lookup_reviews_by_user(user_id: str) -> str:
    """
    Look up historical reviews by exact user_id.
    Source: src/data/train_review.json
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id="")


@tool("lookup_reviews_by_item")
def lookup_reviews_by_item(item_id: str) -> str:
    """
    Look up historical reviews by exact item_id or business_id.
    Source: src/data/train_review.json
    """
    return _lookup_reviews_by_user_and_item_impl(user_id="", item_id=item_id)


@tool("build_prediction_context")
def build_prediction_context(user_id: str, item_id: str) -> str:
    """
    Build deterministic prediction context for final Yelp review generation.

    Sources:
    - src/data/filtered_user.json
    - src/data/filtered_item.json
    - src/data/train_review.json

    The returned JSON contains predicted_stars.
    Final agent must use predicted_stars exactly.
    """
    uid = user_id.strip().strip("'\"")
    iid = item_id.strip().strip("'\"")

    user = _get_user_exact_dict(uid)
    item = _get_item_exact_dict(iid)
    reviews = _get_reviews_list(uid, iid)

    direct_reviews = [
        r for r in reviews
        if _matches_user(r, uid) and _matches_item(r, iid)
    ]

    print("=" * 100, flush=True)
    print("[LOOKUP DEBUG]", flush=True)
    print(f"user_id              : {uid}", flush=True)
    print(f"item_id              : {iid}", flush=True)
    print(f"user_found           : {bool(user)}", flush=True)
    print(f"user_field_count     : {len(user) if isinstance(user, dict) else 0}", flush=True)
    print(f"item_found           : {bool(item)}", flush=True)
    print(f"item_field_count     : {len(item) if isinstance(item, dict) else 0}", flush=True)
    print(f"reviews_found        : {len(reviews) if isinstance(reviews, list) else 0}", flush=True)
    print(f"direct_reviews_found : {len(direct_reviews)}", flush=True)
    print(f"user_file            : {_USER_JSON_PATH}", flush=True)
    print(f"item_file            : {_ITEM_JSON_PATH}", flush=True)
    print(f"review_file          : {_REVIEW_JSON_PATH}", flush=True)
    print("=" * 100, flush=True)

    style = _summarize_review_style(reviews, uid, iid)
    case_info = _detect_case_from_flags(
        user_exists=bool(user),
        item_exists=bool(item),
        direct_review_exists=bool(style.get("direct_review_exists")),
    )
    predicted_stars = _compute_stars(user, item, style)
    calculation_trace = {
        "policy_path": str(_get_eval_policy_path()),
        "default_prior": _policy_get(["rating_policy", "default_prior"], 3.8),
        "case_number": case_info["case_number"] if "case_info" in locals() else None,
    }

    user_avg = _as_float(user.get("average_stars"), 3.8) if user else 3.8
    item_avg = _as_float(item.get("stars"), 3.8) if item else 3.8

    context = {
        "case": {
            "user_exists": bool(user),
            "item_exists": bool(item),
            "direct_review_exists": bool(style.get("direct_review_exists")),
            "case_number": case_info["case_number"],
            "case_name": case_info["case_name"],
            "dominant_evidence": case_info["dominant_evidence"],
            "fallback_policy": case_info["fallback_policy"],
            "calculation_trace": calculation_trace,
        },
        "user": {
            "average_stars": user_avg,
            "review_count": _as_int(user.get("review_count"), 0) if user else 0,
            "yelping_since": user.get("yelping_since", "") if user else "",
            "rating_tendency": _rating_tendency(user_avg),
            "elite": user.get("elite", "") if user else "",
            "fans": _as_int(user.get("fans"), 0) if user else 0,
            "useful": _as_int(user.get("useful"), 0) if user else 0,
            "funny": _as_int(user.get("funny"), 0) if user else 0,
            "cool": _as_int(user.get("cool"), 0) if user else 0,
        },
        "item": {
            "stars": item_avg,
            "review_count": _as_int(item.get("review_count"), 0) if item else 0,
            "name": item.get("name", "") if item else "",
            "categories": item.get("categories", "Restaurants") if item else "Restaurants",
            "city": item.get("city", "") if item else "",
            "state": item.get("state", "") if item else "",
            "attributes": item.get("attributes", {}) if item else {},
            "is_open": item.get("is_open", None) if item else None,
        },
        "review_style": style,
        "predicted_stars": predicted_stars,
        "final_instruction": (
            "Use predicted_stars exactly. Generate one natural Yelp-style review. "
            "Do not mention IDs, formulas, tool names, or unsupported facts. "
            "Output only valid JSON."
        ),
    }

    return json.dumps(context, ensure_ascii=False)


@tool("determine_prediction_case")
def determine_prediction_case(user_id: str, item_id: str) -> str:
    """
    Determine prediction case for exact user_id and item_id.

    Full 8-case logic:
    - Case 1: user exists, item exists, direct historical review exists.
    - Case 2: user exists, item exists, no direct historical review.
    - Case 3: user missing, item exists, direct historical review exists.
    - Case 4: user missing, item exists, no direct historical review.
    - Case 5: user exists, item missing, direct historical review exists.
    - Case 6: user exists, item missing, no direct historical review.
    - Case 7: user missing, item missing, direct historical review exists.
    - Case 8: user missing, item missing, no direct historical review.
    """
    uid = user_id.strip().strip("'\"")
    iid = item_id.strip().strip("'\"")

    user = _get_user_exact_dict(uid)
    item = _get_item_exact_dict(iid)
    reviews = _get_reviews_list(uid, iid)
    style = _summarize_review_style(reviews, uid, iid)

    user_exists = bool(user)
    item_exists = bool(item)
    direct_review_exists = bool(style.get("direct_review_exists"))

    result = _detect_case_from_flags(
        user_exists=user_exists,
        item_exists=item_exists,
        direct_review_exists=direct_review_exists,
    )

    result["flags"] = {
        "user_exists": user_exists,
        "item_exists": item_exists,
        "direct_review_exists": direct_review_exists,
    }

    return json.dumps(result, ensure_ascii=False)

