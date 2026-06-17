"""
Exact lookup tools for Yelp data + deterministic rating policy + grounded review context.

Research-quality version for CrewAI + OpenEvolve.

Data sources only:
- src/data/filtered_user.json
- src/data/filtered_item.json
- src/data/train_review.json

No dummy dataset.
No ChromaDB fallback.
No unsupported facts.

Core design:
1. Python retrieves exact evidence:
   - user profile
   - item/business profile
   - direct user-item review
   - user review history
   - item review history
2. Python computes predicted_stars deterministically from config/eval_evolving.yaml.
3. CrewAI final agent must use predicted_stars exactly and only generate JSON review text.
4. OpenEvolve mutates only the YAML policy inside EVOLVE-BLOCK.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml
from crewai.tools import tool


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
# Global caches
# =============================================================================

_EVAL_POLICY_CACHE: dict | None = None
_EVAL_POLICY_CACHE_PATH: str | None = None

_USER_INDEX: dict[str, dict] | None = None
_ITEM_INDEX: dict[str, dict] | None = None
_REVIEW_INDEX: dict[str, Any] | None = None


# =============================================================================
# Generic helpers
# =============================================================================


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
    global _EVAL_POLICY_CACHE, _EVAL_POLICY_CACHE_PATH

    path = _get_eval_policy_path()
    path_str = str(path)

    if _EVAL_POLICY_CACHE is not None and _EVAL_POLICY_CACHE_PATH == path_str:
        return _EVAL_POLICY_CACHE

    if not path.exists():
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = path_str
        return _EVAL_POLICY_CACHE

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            data = {}

        _EVAL_POLICY_CACHE = data
        _EVAL_POLICY_CACHE_PATH = path_str
        return data
    except Exception as e:
        print(f"[EVAL POLICY WARNING] Cannot load {path}: {e}", flush=True)
        _EVAL_POLICY_CACHE = {}
        _EVAL_POLICY_CACHE_PATH = path_str
        return {}


def _policy_get(path: list[str], default: Any = None) -> Any:
    cur: Any = _load_eval_policy()

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
    """
    Standard half-up rounding to the nearest 0.5.

    Avoid Python banker's rounding:
        round(4.25 * 2) / 2 -> 4.0
    We want:
        4.25 -> 4.5
    """
    return math.floor(value * 2.0 + 0.5) / 2.0


def _compact_text(text: Any, limit: int = 380) -> str:
    text = str(text or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def _word_count(text: Any) -> int:
    return len(str(text or "").split())


def _rating_tendency(avg: float) -> str:
    if avg <= 3.2:
        return "harsh"
    if avg >= 4.2:
        return "generous"
    return "moderate"


def _safe_date(value: Any) -> str:
    return str(value or "")


def _item_value(obj: dict) -> str:
    """Support both internal item_id and Yelp original business_id."""
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("item_id") or obj.get("business_id") or "").strip()


def _user_value(obj: dict) -> str:
    if not isinstance(obj, dict):
        return ""
    return str(obj.get("user_id") or "").strip()


def _matches_item(obj: dict, item_id: str) -> bool:
    return _item_value(obj) == str(item_id or "").strip()


def _matches_user(obj: dict, user_id: str) -> bool:
    return _user_value(obj) == str(user_id or "").strip()


def _normalize_item_id(obj: dict) -> dict:
    """Normalize business_id to item_id for downstream consistency."""
    if isinstance(obj, dict) and "item_id" not in obj and "business_id" in obj:
        obj = dict(obj)
        obj["item_id"] = obj["business_id"]
    return obj


def _iter_json_records(path: Path) -> Iterable[dict]:
    """
    Read JSONL or JSON array.

    Supports:
    - one JSON object per line
    - full JSON array file
    """
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict):
                            yield obj
                return
            except Exception as e:
                print(f"[JSON WARNING] Cannot parse array file {path}: {e}", flush=True)
                return

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


def _review_key(r: dict) -> tuple:
    return (
        str(r.get("review_id", "")),
        _user_value(r),
        _item_value(r),
        str(r.get("date", "")),
        str(r.get("text", ""))[:100],
    )


def _dedupe_reviews(reviews: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    deduped: list[dict] = []

    for r in reviews:
        if not isinstance(r, dict):
            continue
        key = _review_key(r)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(_normalize_item_id(r))

    return deduped


def _sort_recent_reviews(reviews: list[dict], limit: int = 60) -> list[dict]:
    return sorted(reviews, key=lambda x: _safe_date(x.get("date", "")), reverse=True)[:limit]


# =============================================================================
# Indexed exact lookup
# =============================================================================


def _build_user_index() -> dict[str, dict]:
    global _USER_INDEX
    if _USER_INDEX is not None:
        return _USER_INDEX

    index: dict[str, dict] = {}
    for obj in _iter_json_records(_USER_JSON_PATH):
        uid = _user_value(obj)
        if uid and uid not in index:
            index[uid] = obj

    _USER_INDEX = index
    return index


def _build_item_index() -> dict[str, dict]:
    global _ITEM_INDEX
    if _ITEM_INDEX is not None:
        return _ITEM_INDEX

    index: dict[str, dict] = {}
    for obj in _iter_json_records(_ITEM_JSON_PATH):
        obj = _normalize_item_id(obj)
        iid = _item_value(obj)
        bid = str(obj.get("business_id") or "").strip()

        if iid and iid not in index:
            index[iid] = obj
        if bid and bid not in index:
            index[bid] = obj

    _ITEM_INDEX = index
    return index


def _build_review_index() -> dict[str, Any]:
    """
    Build indexes once per process.

    Indexes:
    - by_user[user_id] -> list[review]
    - by_item[item_id/business_id] -> list[review]
    - by_pair[(user_id, item_id)] -> list[review]
    """
    global _REVIEW_INDEX
    if _REVIEW_INDEX is not None:
        return _REVIEW_INDEX

    by_user: dict[str, list[dict]] = defaultdict(list)
    by_item: dict[str, list[dict]] = defaultdict(list)
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for obj in _iter_json_records(_REVIEW_JSON_PATH):
        r = _normalize_item_id(obj)
        uid = _user_value(r)
        iid = _item_value(r)
        bid = str(r.get("business_id") or "").strip()

        if uid:
            by_user[uid].append(r)
        if iid:
            by_item[iid].append(r)
        if bid and bid != iid:
            by_item[bid].append(r)
        if uid and iid:
            by_pair[(uid, iid)].append(r)
        if uid and bid and bid != iid:
            by_pair[(uid, bid)].append(r)

    by_user = {k: _sort_recent_reviews(_dedupe_reviews(v), limit=5000) for k, v in by_user.items()}
    by_item = {k: _sort_recent_reviews(_dedupe_reviews(v), limit=5000) for k, v in by_item.items()}
    by_pair = {k: _sort_recent_reviews(_dedupe_reviews(v), limit=200) for k, v in by_pair.items()}

    _REVIEW_INDEX = {
        "by_user": by_user,
        "by_item": by_item,
        "by_pair": by_pair,
    }
    return _REVIEW_INDEX


def _get_user_exact_dict(user_id: str) -> dict:
    uid = str(user_id or "").strip().strip("'\"")
    return _build_user_index().get(uid, {})


def _get_item_exact_dict(item_id: str) -> dict:
    iid = str(item_id or "").strip().strip("'\"")
    return _build_item_index().get(iid, {})


def _get_direct_reviews(user_id: str, item_id: str, limit: int = 60) -> list[dict]:
    uid = str(user_id or "").strip().strip("'\"")
    iid = str(item_id or "").strip().strip("'\"")
    index = _build_review_index()
    reviews = index["by_pair"].get((uid, iid), [])
    return _sort_recent_reviews(_dedupe_reviews(reviews), limit=limit)


def _get_user_history_reviews(user_id: str, limit: int = 60) -> list[dict]:
    uid = str(user_id or "").strip().strip("'\"")
    index = _build_review_index()
    reviews = index["by_user"].get(uid, [])
    return _sort_recent_reviews(_dedupe_reviews(reviews), limit=limit)


def _get_item_history_reviews(item_id: str, limit: int = 60) -> list[dict]:
    iid = str(item_id or "").strip().strip("'\"")
    index = _build_review_index()
    reviews = index["by_item"].get(iid, [])
    return _sort_recent_reviews(_dedupe_reviews(reviews), limit=limit)


def _lookup_reviews_impl(user_id: str = "", item_id: str = "", limit: int = 60) -> str:
    uid = str(user_id or "").strip().strip("'\"")
    iid = str(item_id or "").strip().strip("'\"")

    if not uid and not iid:
        return "Error: must provide at least one of user_id or item_id."

    if not _REVIEW_JSON_PATH.exists():
        return f"Review file not found: {_REVIEW_JSON_PATH}"

    if uid and iid:
        reviews = _get_direct_reviews(uid, iid, limit=limit)
    elif uid:
        reviews = _get_user_history_reviews(uid, limit=limit)
    else:
        reviews = _get_item_history_reviews(iid, limit=limit)

    if not reviews:
        return f"No reviews found for user_id={uid!r}, item_id={iid!r}"

    return json.dumps(reviews, ensure_ascii=False)


# =============================================================================
# Review analysis helpers
# =============================================================================


def _extract_categories(item: dict) -> list[str]:
    raw = item.get("categories", "") if isinstance(item, dict) else ""
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [x.strip() for x in str(raw or "").split(",") if x.strip()]


def _common_terms_from_texts(texts: list[str], limit: int = 8) -> list[str]:
    stop = {
        "the", "and", "but", "for", "with", "this", "that", "was", "were", "are", "you", "have",
        "had", "not", "too", "very", "really", "just", "they", "there", "here", "place", "food",
        "good", "great", "like", "from", "out", "all", "get", "got", "one", "our", "can", "will",
    }
    counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z']{2,}", str(text or "").lower()):
            if token not in stop:
                counter[token] += 1
    return [w for w, _ in counter.most_common(limit)]


def _sample_style_examples(reviews: list[dict], limit: int = 3) -> list[dict]:
    examples: list[dict] = []
    for r in _sort_recent_reviews(reviews, limit=limit):
        examples.append(
            {
                "stars": _as_float(r.get("stars"), 3.8),
                "date": r.get("date", ""),
                "text_excerpt": _compact_text(r.get("text", ""), 360),
            }
        )
    return examples


def _valid_ratings(reviews: list[dict]) -> list[float]:
    ratings = [_as_float(r.get("stars"), 0.0) for r in reviews if r.get("stars") is not None]
    return [x for x in ratings if 1.0 <= x <= 5.0]


def _summarize_review_style(
    direct_reviews: list[dict],
    user_history_reviews: list[dict],
    item_history_reviews: list[dict],
    user_id: str,
    item_id: str,
) -> dict:
    """
    Summarize direct evidence + user style + item evidence.

    Important:
    - direct_reviews are only exact user-item reviews.
    - user_history_reviews are all/recent reviews by this user.
    - item_history_reviews are all/recent reviews for this item.
    """
    uid = str(user_id or "").strip().strip("'\"")
    iid = str(item_id or "").strip().strip("'\"")

    direct_reviews = [r for r in direct_reviews if _matches_user(r, uid) and _matches_item(r, iid)]
    user_reviews = [r for r in user_history_reviews if _matches_user(r, uid)]
    item_reviews = [r for r in item_history_reviews if _matches_item(r, iid)]

    direct = _sort_recent_reviews(direct_reviews, limit=1)[0] if direct_reviews else None

    style_source = user_reviews or direct_reviews or item_reviews
    style_source = _sort_recent_reviews(_dedupe_reviews(style_source), limit=10)

    word_counts = [_word_count(r.get("text", "")) for r in style_source if r.get("text")]

    user_ratings = _valid_ratings(user_reviews)
    item_ratings = _valid_ratings(item_reviews)
    direct_rating = _as_float(direct.get("stars"), 3.8) if direct else None

    user_texts = [str(r.get("text", "")) for r in user_reviews if r.get("text")]
    positive_user_texts = [str(r.get("text", "")) for r in user_reviews if _as_float(r.get("stars"), 0.0) >= 4.0]
    negative_user_texts = [str(r.get("text", "")) for r in user_reviews if _as_float(r.get("stars"), 0.0) <= 2.0]

    return {
        "direct_review_exists": bool(direct),
        "direct_review_stars": direct_rating,
        "direct_review_excerpt": _compact_text(direct.get("text", ""), 500) if direct else "",
        "direct_review_date": direct.get("date", "") if direct else "",

        "user_review_count_used": len(user_reviews),
        "item_review_count_used": len(item_reviews),
        "user_history_average_stars": round(statistics.mean(user_ratings), 3) if user_ratings else None,
        "item_history_average_stars": round(statistics.mean(item_ratings), 3) if item_ratings else None,

        "avg_word_count": int(statistics.mean(word_counts)) if word_counts else 60,
        "median_word_count": int(statistics.median(word_counts)) if word_counts else 60,
        "style_examples": _sample_style_examples(style_source, limit=3),

        "user_common_positive_terms": _common_terms_from_texts(positive_user_texts, limit=8),
        "user_common_negative_terms": _common_terms_from_texts(negative_user_texts, limit=8),
        "user_common_terms": _common_terms_from_texts(user_texts, limit=8),
    }


# =============================================================================
# Rating policy engine
# =============================================================================


def _compute_stars(user: dict, item: dict, style: dict) -> float:
    """
    Deterministic predicted stars controlled by config/eval_evolving.yaml.

    OpenEvolve mutates eval_evolving.yaml.
    This function loads that policy and applies evolved parameters.
    """
    default_prior = _as_float(_policy_get(["rating_policy", "default_prior"], 3.8), 3.8)
    min_stars = _as_float(_policy_get(["rating_policy", "rounding", "min_stars"], 1.0), 1.0)
    max_stars = _as_float(_policy_get(["rating_policy", "rounding", "max_stars"], 5.0), 5.0)

    direct_exists = bool(style.get("direct_review_exists"))
    direct_stars = style.get("direct_review_stars")

    if direct_exists and direct_stars is not None:
        return float(_round_half(_clamp(_as_float(direct_stars, default_prior), min_stars, max_stars)))

    user_exists = bool(user)
    item_exists = bool(item)

    user_avg = _as_float(user.get("average_stars"), default_prior) if user_exists else default_prior
    item_avg = _as_float(item.get("stars"), default_prior) if item_exists else default_prior

    user_count = _as_int(user.get("review_count"), 0) if user_exists else 0
    item_count = _as_int(item.get("review_count"), 0) if item_exists else 0

    if not user_exists and not item_exists:
        return float(_round_half(_clamp(default_prior, min_stars, max_stars)))

    if user_exists and not item_exists:
        hist_avg = style.get("user_history_average_stars")
        if hist_avg is not None:
            user_avg = 0.75 * user_avg + 0.25 * _as_float(hist_avg, user_avg)
        return float(_round_half(_clamp(user_avg, min_stars, max_stars)))

    if not user_exists and item_exists:
        hist_item_avg = style.get("item_history_average_stars")
        if hist_item_avg is not None:
            item_avg = 0.75 * item_avg + 0.25 * _as_float(hist_item_avg, item_avg)
        return float(_round_half(_clamp(item_avg, min_stars, max_stars)))

    case2_path = ["rating_policy", "case_rules", "case_2"]

    base_user_weight = _as_float(_policy_get(case2_path + ["base_user_weight"], 0.65), 0.65)
    base_item_weight = _as_float(_policy_get(case2_path + ["base_item_weight"], 0.35), 0.35)

    user_confidence_divisor = _as_float(_policy_get(case2_path + ["user_confidence_divisor"], 50.0), 50.0)
    item_confidence_divisor = _as_float(_policy_get(case2_path + ["item_confidence_divisor"], 100.0), 100.0)

    user_confidence_bonus = _as_float(_policy_get(case2_path + ["user_confidence_bonus"], 0.25), 0.25)
    item_confidence_bonus = _as_float(_policy_get(case2_path + ["item_confidence_bonus"], 0.20), 0.20)

    historical_user_review_blend_weight = _as_float(
        _policy_get(case2_path + ["historical_user_review_blend_weight"], 0.30),
        0.30,
    )
    historical_user_review_min_count = _as_int(
        _policy_get(case2_path + ["historical_user_review_min_count"], 3),
        3,
    )

    historical_item_review_blend_weight = _as_float(
        _policy_get(case2_path + ["historical_item_review_blend_weight"], 0.15),
        0.15,
    )
    historical_item_review_min_count = _as_int(
        _policy_get(case2_path + ["historical_item_review_min_count"], 3),
        3,
    )

    user_anchor_min_review_count = _as_int(_policy_get(case2_path + ["user_anchor_min_review_count"], 30), 30)
    user_anchor_max_delta = _as_float(_policy_get(case2_path + ["user_anchor_max_delta"], 0.70), 0.70)

    item_anchor_min_review_count = _as_int(_policy_get(case2_path + ["item_anchor_min_review_count"], 80), 80)
    item_anchor_max_delta = _as_float(_policy_get(case2_path + ["item_anchor_max_delta"], 1.10), 1.10)

    base_user_weight = _clamp(base_user_weight, 0.05, 5.0)
    base_item_weight = _clamp(base_item_weight, 0.05, 5.0)
    user_confidence_divisor = max(user_confidence_divisor, 1.0)
    item_confidence_divisor = max(item_confidence_divisor, 1.0)
    user_confidence_bonus = _clamp(user_confidence_bonus, 0.0, 3.0)
    item_confidence_bonus = _clamp(item_confidence_bonus, 0.0, 3.0)
    historical_user_review_blend_weight = _clamp(historical_user_review_blend_weight, 0.0, 0.9)
    historical_item_review_blend_weight = _clamp(historical_item_review_blend_weight, 0.0, 0.7)
    user_anchor_max_delta = _clamp(user_anchor_max_delta, 0.0, 4.0)
    item_anchor_max_delta = _clamp(item_anchor_max_delta, 0.0, 4.0)

    user_conf = min(user_count / user_confidence_divisor, 1.0)
    item_conf = min(item_count / item_confidence_divisor, 1.0)

    user_weight = base_user_weight + user_confidence_bonus * user_conf
    item_weight = base_item_weight + item_confidence_bonus * item_conf

    denominator = user_weight + item_weight
    if denominator <= 0:
        rating = default_prior
    else:
        rating = (user_weight * user_avg + item_weight * item_avg) / denominator

    hist_user_avg = style.get("user_history_average_stars")
    user_review_count_used = _as_int(style.get("user_review_count_used"), 0)
    if hist_user_avg is not None and user_review_count_used >= historical_user_review_min_count:
        hist_user_avg = _as_float(hist_user_avg, rating)
        w = historical_user_review_blend_weight
        rating = (1.0 - w) * rating + w * hist_user_avg

    hist_item_avg = style.get("item_history_average_stars")
    item_review_count_used = _as_int(style.get("item_review_count_used"), 0)
    if hist_item_avg is not None and item_review_count_used >= historical_item_review_min_count:
        hist_item_avg = _as_float(hist_item_avg, rating)
        w = historical_item_review_blend_weight
        rating = (1.0 - w) * rating + w * hist_item_avg

    if user_count >= user_anchor_min_review_count:
        rating = max(user_avg - user_anchor_max_delta, min(user_avg + user_anchor_max_delta, rating))

    if item_count >= item_anchor_min_review_count:
        rating = max(item_avg - item_anchor_max_delta, min(item_avg + item_anchor_max_delta, rating))

    return float(_round_half(_clamp(rating, min_stars, max_stars)))


# =============================================================================
# 8-case detection
# =============================================================================


def _detect_case_from_flags(user_exists: bool, item_exists: bool, direct_review_exists: bool) -> dict:
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
            "dominant_evidence": "user_and_item_profiles_plus_histories",
            "fallback_policy": "combine user behavior, user history, item quality, and item history",
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
            "dominant_evidence": "item_profile_and_item_history",
            "fallback_policy": "use item stars, item review history, and category context",
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
            "dominant_evidence": "user_profile_and_user_history",
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
# Context guidance for final agent
# =============================================================================


def _get_review_policy_for_context(case_number: int, predicted_stars: float) -> dict:
    review_policy = _policy_get(["review_policy"], {}) or {}
    if not isinstance(review_policy, dict):
        review_policy = {}

    case_guidance = _policy_get(["review_policy", "case_review_guidance", f"case_{case_number}"], "")

    sentiment_guidance: dict | str = ""
    sentiment_by_star = _policy_get(["review_policy", "sentiment_by_star"], {}) or {}
    if isinstance(sentiment_by_star, dict):
        for _, rule in sentiment_by_star.items():
            if not isinstance(rule, dict):
                continue
            star_range = rule.get("range")
            if (
                isinstance(star_range, list)
                and len(star_range) == 2
                and _as_float(star_range[0], 1.0) <= predicted_stars <= _as_float(star_range[1], 5.0)
            ):
                sentiment_guidance = {
                    "tone": rule.get("tone", ""),
                    "guidance": rule.get("guidance", ""),
                }
                break

    return {
        "global_rules": review_policy.get("global_rules", []),
        "case_guidance": case_guidance,
        "sentiment_guidance": sentiment_guidance,
    }


def _build_evidence_summary(user: dict, item: dict, style: dict, case_info: dict, predicted_stars: float) -> dict:
    user_avg = _as_float(user.get("average_stars"), 3.8) if user else None
    item_avg = _as_float(item.get("stars"), 3.8) if item else None

    summary = {
        "case_number": case_info.get("case_number"),
        "predicted_stars": predicted_stars,
        "rating_basis": case_info.get("dominant_evidence", ""),
        "direct_review_signal": "",
        "user_signal": "",
        "item_signal": "",
        "history_signal": "",
        "generation_note": "",
    }

    if style.get("direct_review_exists"):
        summary["direct_review_signal"] = (
            f"Direct historical review exists with stars={style.get('direct_review_stars')}. "
            "This is the strongest rating and sentiment evidence."
        )
    else:
        summary["direct_review_signal"] = "No direct historical review exists for this user-item pair."

    if user:
        summary["user_signal"] = (
            f"User profile exists: average_stars={user_avg}, "
            f"review_count={_as_int(user.get('review_count'), 0)}, "
            f"rating_tendency={_rating_tendency(user_avg or 3.8)}."
        )
    else:
        summary["user_signal"] = "User profile is missing; avoid user-specific claims."

    if item:
        categories = ", ".join(_extract_categories(item)[:6])
        summary["item_signal"] = (
            f"Item profile exists: stars={item_avg}, "
            f"review_count={_as_int(item.get('review_count'), 0)}, "
            f"categories={categories}."
        )
    else:
        summary["item_signal"] = "Item profile is missing; avoid item-specific claims."

    summary["history_signal"] = (
        f"User history reviews used={style.get('user_review_count_used', 0)}, "
        f"user_history_average_stars={style.get('user_history_average_stars')}; "
        f"item history reviews used={style.get('item_review_count_used', 0)}, "
        f"item_history_average_stars={style.get('item_history_average_stars')}."
    )

    case_number = case_info.get("case_number")
    if case_number == 1:
        summary["generation_note"] = "Preserve direct review sentiment; item/user context may only support wording."
    elif case_number == 2:
        summary["generation_note"] = "Use user tendency and item facts, but never claim the user reviewed this exact item before."
    elif case_number == 3:
        summary["generation_note"] = "Use direct review and item facts; do not infer user history."
    elif case_number == 4:
        summary["generation_note"] = "Use item facts and item history only; avoid user-specific claims."
    elif case_number == 5:
        summary["generation_note"] = "Use direct review and user profile; avoid unsupported item-specific details."
    elif case_number == 6:
        summary["generation_note"] = "Use user profile and user history only; keep item details generic."
    elif case_number == 7:
        summary["generation_note"] = "Use direct review only; avoid unsupported user or item facts."
    else:
        summary["generation_note"] = "Use generic, low-specificity fallback review."

    return summary


def _minimal_user_context(user: dict) -> dict:
    user_avg = _as_float(user.get("average_stars"), 3.8) if user else 3.8
    return {
        "average_stars": user_avg,
        "review_count": _as_int(user.get("review_count"), 0) if user else 0,
        "yelping_since": user.get("yelping_since", "") if user else "",
        "rating_tendency": _rating_tendency(user_avg),
        "elite": user.get("elite", "") if user else "",
        "fans": _as_int(user.get("fans"), 0) if user else 0,
        "useful": _as_int(user.get("useful"), 0) if user else 0,
        "funny": _as_int(user.get("funny"), 0) if user else 0,
        "cool": _as_int(user.get("cool"), 0) if user else 0,
    }


def _minimal_item_context(item: dict) -> dict:
    item_avg = _as_float(item.get("stars"), 3.8) if item else 3.8
    return {
        "stars": item_avg,
        "review_count": _as_int(item.get("review_count"), 0) if item else 0,
        "name": item.get("name", "") if item else "",
        "categories": item.get("categories", "Restaurants") if item else "Restaurants",
        "city": item.get("city", "") if item else "",
        "state": item.get("state", "") if item else "",
        "attributes": item.get("attributes", {}) if item else {},
        "is_open": item.get("is_open", None) if item else None,
    }


# =============================================================================
# CrewAI tools
# =============================================================================


@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """Look up a user's complete profile by exact user_id from src/data/filtered_user.json."""
    uid = str(user_id or "").strip().strip("'\"")
    data = _get_user_exact_dict(uid)
    if data:
        return json.dumps(data, ensure_ascii=False)
    return f"No user found with user_id: {uid}"


@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """Look up a business/item profile by exact item_id or business_id from src/data/filtered_item.json."""
    iid = str(item_id or "").strip().strip("'\"")
    data = _get_item_exact_dict(iid)
    if data:
        return json.dumps(data, ensure_ascii=False)
    return f"No item found with item_id: {iid}"


@tool("lookup_reviews_by_user_and_item")
def lookup_reviews_by_user_and_item(user_id: str = "", item_id: str = "") -> str:
    """Look up direct historical reviews by exact user_id and item_id from src/data/train_review.json."""
    return _lookup_reviews_impl(user_id=user_id, item_id=item_id, limit=60)


@tool("lookup_reviews_by_user")
def lookup_reviews_by_user(user_id: str) -> str:
    """Look up recent historical reviews by exact user_id from src/data/train_review.json."""
    return _lookup_reviews_impl(user_id=user_id, item_id="", limit=60)


@tool("lookup_reviews_by_item")
def lookup_reviews_by_item(item_id: str) -> str:
    """Look up recent historical reviews by exact item_id or business_id from src/data/train_review.json."""
    return _lookup_reviews_impl(user_id="", item_id=item_id, limit=60)


@tool("build_prediction_context")
def build_prediction_context(user_id: str, item_id: str) -> str:
    """
    Build deterministic prediction context for final Yelp review generation.

    Returned JSON contains predicted_stars.
    Final agent must use predicted_stars exactly.
    """
    uid = str(user_id or "").strip().strip("'\"")
    iid = str(item_id or "").strip().strip("'\"")

    user = _get_user_exact_dict(uid)
    item = _get_item_exact_dict(iid)

    direct_reviews = _get_direct_reviews(uid, iid, limit=60)
    user_history_reviews = _get_user_history_reviews(uid, limit=60)
    item_history_reviews = _get_item_history_reviews(iid, limit=60)

    merged_reviews = _dedupe_reviews(direct_reviews + user_history_reviews + item_history_reviews)

    style = _summarize_review_style(
        direct_reviews=direct_reviews,
        user_history_reviews=user_history_reviews,
        item_history_reviews=item_history_reviews,
        user_id=uid,
        item_id=iid,
    )

    case_info = _detect_case_from_flags(
        user_exists=bool(user),
        item_exists=bool(item),
        direct_review_exists=bool(style.get("direct_review_exists")),
    )

    predicted_stars = _compute_stars(user, item, style)

    calculation_trace = {
        "policy_path": str(_get_eval_policy_path()),
        "default_prior": _policy_get(["rating_policy", "default_prior"], 3.8),
        "case_number": case_info["case_number"],
        "rounding_mode": _policy_get(["rating_policy", "rounding", "mode"], "nearest_half_up"),
        "direct_reviews_found": len(direct_reviews),
        "user_history_reviews_found": len(user_history_reviews),
        "item_history_reviews_found": len(item_history_reviews),
        "merged_reviews_found": len(merged_reviews),
    }

    print("=" * 100, flush=True)
    print("[LOOKUP DEBUG]", flush=True)
    print(f"user_id                    : {uid}", flush=True)
    print(f"item_id                    : {iid}", flush=True)
    print(f"user_found                 : {bool(user)}", flush=True)
    print(f"user_field_count           : {len(user) if isinstance(user, dict) else 0}", flush=True)
    print(f"item_found                 : {bool(item)}", flush=True)
    print(f"item_field_count           : {len(item) if isinstance(item, dict) else 0}", flush=True)
    print(f"direct_reviews_found       : {len(direct_reviews)}", flush=True)
    print(f"user_history_reviews_found : {len(user_history_reviews)}", flush=True)
    print(f"item_history_reviews_found : {len(item_history_reviews)}", flush=True)
    print(f"merged_reviews_found       : {len(merged_reviews)}", flush=True)
    print(f"case_number                : {case_info['case_number']}", flush=True)
    print(f"predicted_stars            : {predicted_stars}", flush=True)
    print(f"policy_path                : {_get_eval_policy_path()}", flush=True)
    print(f"user_file                  : {_USER_JSON_PATH}", flush=True)
    print(f"item_file                  : {_ITEM_JSON_PATH}", flush=True)
    print(f"review_file                : {_REVIEW_JSON_PATH}", flush=True)
    print("=" * 100, flush=True)

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
        "user": _minimal_user_context(user),
        "item": _minimal_item_context(item),
        "review_style": style,
        "evidence_summary": _build_evidence_summary(user, item, style, case_info, predicted_stars),
        "review_policy": _get_review_policy_for_context(case_info["case_number"], predicted_stars),
        "predicted_stars": predicted_stars,
        "final_instruction": (
            "Use predicted_stars exactly as the output stars value. "
            "Generate one natural Yelp-style review grounded only in the provided context. "
            "Do not mention IDs, formulas, tools, agents, YAML, database fields, or unsupported facts. "
            "Output only one valid JSON object with keys: stars and review."
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
    uid = str(user_id or "").strip().strip("'\"")
    iid = str(item_id or "").strip().strip("'\"")

    user = _get_user_exact_dict(uid)
    item = _get_item_exact_dict(iid)
    direct_reviews = _get_direct_reviews(uid, iid, limit=60)

    direct_review_exists = bool(direct_reviews)
    result = _detect_case_from_flags(
        user_exists=bool(user),
        item_exists=bool(item),
        direct_review_exists=direct_review_exists,
    )

    result["flags"] = {
        "user_exists": bool(user),
        "item_exists": bool(item),
        "direct_review_exists": direct_review_exists,
    }

    result["counts"] = {
        "direct_reviews_found": len(direct_reviews),
        "user_history_reviews_found": len(_get_user_history_reviews(uid, limit=60)),
        "item_history_reviews_found": len(_get_item_history_reviews(iid, limit=60)),
    }

    return json.dumps(result, ensure_ascii=False)
