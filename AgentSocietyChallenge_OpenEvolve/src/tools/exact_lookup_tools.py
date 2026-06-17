"""
Exact and hybrid lookup tools for Yelp data.

Main goals:
- Prefer AgentSociety simulator interaction tool when available.
- Fall back to local ChromaDB vector collections.
- Fall back to JSONL files if available.
- Build deterministic prediction context for the multi-agent CrewAI pipeline.

Important:
- Final stars are computed here.
- LLM agents must use predicted_stars exactly.
"""

import json
import os
import statistics
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from crewai.tools import tool

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHROMA_DIR = str(_PROJECT_ROOT / "lmdb_cache" / "my_chroma")

_USER_COLLECTION = "benchmark_true_fresh_index_Filtered_User_3"
_ITEM_COLLECTION = "benchmark_true_fresh_index_Filtered_Item_3"
_REVIEW_COLLECTION = "benchmark_true_fresh_index_Filtered_Review_3"

_client = None
_embedder = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=_CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embedder


def _get_simulator_tool():
    try:
        from src.tools.interaction_tool_wrapper import _GLOBAL_INTERACTION_TOOL
        return _GLOBAL_INTERACTION_TOOL
    except Exception:
        return None


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


def _parse_documents(documents: list[str]) -> list[dict]:
    parsed = []

    for doc in documents:
        if not doc:
            continue

        for line in str(doc).strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return parsed


def _hybrid_search(
    collection_name: str,
    query_text: str,
    contains_id: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Query ChromaDB using semantic embedding plus document contains filter.
    Falls back to document contains query if embedding query fails.
    """
    client = _get_client()
    collection = client.get_collection(collection_name)

    documents = []

    try:
        embedder = _get_embedder()
        query_embedding = embedder.embed_query(query_text)

        results = collection.query(
            query_embeddings=[query_embedding],
            where_document={"$contains": contains_id},
            n_results=n_results,
            include=["documents", "distances"],
        )

        documents = results.get("documents", [[]])[0]

    except Exception:
        documents = []

    if not documents:
        try:
            results = collection.get(
                where_document={"$contains": contains_id},
                limit=n_results,
                include=["documents"],
            )
            documents = results.get("documents", [])
        except Exception:
            documents = []

    return _parse_documents(documents)


def _fallback_file_search(filepath: str, key: str, target_id: str) -> dict:
    if not os.path.exists(filepath):
        return {}

    target1 = f'"{key}":"{target_id}"'
    target2 = f'"{key}": "{target_id}"'

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if target1 not in line and target2 not in line:
                    continue

                try:
                    data = json.loads(line.strip())
                except Exception:
                    continue

                if data.get(key) == target_id:
                    return data

    except Exception:
        return {}

    return {}


def _get_user_exact_dict(user_id: str) -> dict:
    uid = user_id.strip().strip("'\"")

    simulator_tool = _get_simulator_tool()
    if simulator_tool is not None:
        try:
            data = simulator_tool.get_user(user_id=uid)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    try:
        results = _hybrid_search(
            _USER_COLLECTION,
            f"Find exact user profile for user_id {uid}",
            uid,
            n_results=5,
        )
        for r in results:
            if r.get("user_id") == uid:
                r.pop("_similarity_distance", None)
                return r
    except Exception:
        pass

    candidates = [
        _PROJECT_ROOT / "dummy_dataset" / "user.json",
        _PROJECT_ROOT / "data" / "user_subset.json",
        _PROJECT_ROOT / "data" / "user.json",
    ]

    for path in candidates:
        data = _fallback_file_search(str(path), "user_id", uid)
        if data:
            return data

    return {}


def _get_item_exact_dict(item_id: str) -> dict:
    iid = item_id.strip().strip("'\"")

    simulator_tool = _get_simulator_tool()
    if simulator_tool is not None:
        try:
            data = simulator_tool.get_item(item_id=iid)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass

    try:
        results = _hybrid_search(
            _ITEM_COLLECTION,
            f"Find exact business profile for item_id {iid}",
            iid,
            n_results=5,
        )
        for r in results:
            if r.get("item_id") == iid:
                r.pop("_similarity_distance", None)
                return r
    except Exception:
        pass

    candidates = [
        _PROJECT_ROOT / "dummy_dataset" / "item.json",
        _PROJECT_ROOT / "data" / "item_subset.json",
        _PROJECT_ROOT / "data" / "item.json",
    ]

    for path in candidates:
        data = _fallback_file_search(str(path), "item_id", iid)
        if data:
            return data

    return {}

def _lookup_reviews_by_user_and_item_impl(user_id: str = "", item_id: str = "") -> str:
    uid = user_id.strip().strip("'\"") if user_id else ""
    iid = item_id.strip().strip("'\"") if item_id else ""

    if not uid and not iid:
        return "Error: must provide at least one of user_id or item_id."

    simulator_tool = _get_simulator_tool()
    results: list[dict] = []

    if simulator_tool is not None:
        try:
            if uid and iid:
                user_reviews = simulator_tool.get_reviews(user_id=uid) or []
                exact_reviews = [
                    r for r in user_reviews
                    if isinstance(r, dict) and r.get("item_id") == iid
                ]
                if exact_reviews:
                    return json.dumps(exact_reviews, ensure_ascii=False)

            if uid:
                user_reviews = simulator_tool.get_reviews(user_id=uid) or []
                if isinstance(user_reviews, list):
                    results.extend([r for r in user_reviews if isinstance(r, dict)])

            if iid:
                item_reviews = simulator_tool.get_reviews(item_id=iid) or []
                if isinstance(item_reviews, list):
                    results.extend([r for r in item_reviews if isinstance(r, dict)])

        except Exception:
            pass

    if uid and iid:
        try:
            chroma_results = _hybrid_search(
                _REVIEW_COLLECTION,
                f"Find exact reviews for user_id {uid} and item_id {iid}",
                uid,
                n_results=50,
            )
            exact = [
                r for r in chroma_results
                if r.get("user_id") == uid and r.get("item_id") == iid
            ]
            if exact:
                return json.dumps(exact, ensure_ascii=False)
        except Exception:
            pass

    if not results and uid:
        try:
            user_results = _hybrid_search(
                _REVIEW_COLLECTION,
                f"Find reviews written by user_id {uid}",
                uid,
                n_results=30,
            )
            results.extend([r for r in user_results if r.get("user_id") == uid])
        except Exception:
            pass

    if not results and iid:
        try:
            item_results = _hybrid_search(
                _REVIEW_COLLECTION,
                f"Find reviews for item_id {iid}",
                iid,
                n_results=30,
            )
            results.extend([r for r in item_results if r.get("item_id") == iid])
        except Exception:
            pass

    if not results:
        file_candidates = [
            _PROJECT_ROOT / "dummy_dataset" / "review.json",
            _PROJECT_ROOT / "data" / "review_subset.json",
            _PROJECT_ROOT / "data" / "review.json",
        ]

        for path in file_candidates:
            if not path.exists():
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line.strip())
                        except Exception:
                            continue

                        if uid and iid:
                            if r.get("user_id") == uid and r.get("item_id") == iid:
                                results.append(r)
                        elif uid:
                            if r.get("user_id") == uid:
                                results.append(r)
                        elif iid:
                            if r.get("item_id") == iid:
                                results.append(r)

                        if len(results) >= 60:
                            break

            except Exception:
                continue

            if results:
                break

    seen = set()
    deduped = []

    for r in results:
        key = (
            r.get("review_id", ""),
            r.get("user_id", ""),
            r.get("item_id", ""),
            r.get("date", ""),
            str(r.get("text", ""))[:50],
        )

        if key in seen:
            continue

        seen.add(key)
        r.pop("_similarity_distance", None)
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


def _summarize_review_style(reviews: list[dict], user_id: str, item_id: str) -> dict:
    uid = user_id.strip().strip("'\"")
    iid = item_id.strip().strip("'\"")

    direct_reviews = [
        r for r in reviews
        if r.get("user_id") == uid and r.get("item_id") == iid
    ]

    user_reviews = [
        r for r in reviews
        if r.get("user_id") == uid
    ]

    item_reviews = [
        r for r in reviews
        if r.get("item_id") == iid
    ]

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
    Compute deterministic predicted stars.

    8-case aware logic:
    - If direct review exists, direct_review_stars is strongest.
    - If user and item exist, blend user average and item stars.
    - If only user exists, use user average.
    - If only item exists, use item stars.
    - If nothing exists, use default prior.
    """
    direct_exists = bool(style.get("direct_review_exists"))
    direct_stars = style.get("direct_review_stars")

    if direct_exists and direct_stars is not None:
        return float(_round_half(_clamp(_as_float(direct_stars, 3.8))))

    user_exists = bool(user)
    item_exists = bool(item)

    user_avg = _as_float(user.get("average_stars"), 3.8) if user_exists else 3.8
    item_avg = _as_float(item.get("stars"), 3.8) if item_exists else 3.8

    user_count = _as_int(user.get("review_count"), 0) if user_exists else 0
    item_count = _as_int(item.get("review_count"), 0) if item_exists else 0

    # Case 8: user missing, item missing, no direct review
    if not user_exists and not item_exists:
        return 3.8

    # Case 6: user exists, item missing, no direct review
    if user_exists and not item_exists:
        return float(_round_half(_clamp(user_avg)))

    # Case 4: user missing, item exists, no direct review
    if not user_exists and item_exists:
        return float(_round_half(_clamp(item_avg)))

    # Case 2: user exists, item exists, no direct review
    user_conf = min(user_count / 50.0, 1.0)
    item_conf = min(item_count / 100.0, 1.0)

    user_weight = 0.65 + 0.25 * user_conf
    item_weight = 0.35 + 0.20 * item_conf

    rating = (user_weight * user_avg + item_weight * item_avg) / (
        user_weight + item_weight
    )

    hist_avg = style.get("user_history_average_stars")
    if hist_avg is not None and style.get("user_review_count_used", 0) >= 3:
        rating = 0.70 * rating + 0.30 * float(hist_avg)

    if user_count >= 30:
        max_delta = 0.7
        rating = max(user_avg - max_delta, min(user_avg + max_delta, rating))

    return float(_round_half(_clamp(rating)))


@tool("lookup_user_by_id")
def lookup_user_by_id(user_id: str) -> str:
    """
    Look up a user's complete profile by exact user_id.
    """
    uid = user_id.strip().strip("'\"")
    data = _get_user_exact_dict(uid)

    if data:
        return json.dumps(data, ensure_ascii=False)

    return f"No user found with user_id: {uid}"


@tool("lookup_item_by_id")
def lookup_item_by_id(item_id: str) -> str:
    """
    Look up a business/item profile by exact item_id.
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
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id=item_id)


@tool("lookup_reviews_by_user")
def lookup_reviews_by_user(user_id: str) -> str:
    """
    Look up historical reviews by exact user_id.
    """
    return _lookup_reviews_by_user_and_item_impl(user_id=user_id, item_id="")


@tool("lookup_reviews_by_item")
def lookup_reviews_by_item(item_id: str) -> str:
    """
    Look up historical reviews by exact item_id.
    """
    return _lookup_reviews_by_user_and_item_impl(user_id="", item_id=item_id)

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

@tool("build_prediction_context")
def build_prediction_context(user_id: str, item_id: str) -> str:
    """
    Build deterministic prediction context for final Yelp review generation.

    The returned JSON contains predicted_stars.
    Final agent must use predicted_stars exactly.
    """
    uid = user_id.strip().strip("'\"")
    iid = item_id.strip().strip("'\"")

    user = _get_user_exact_dict(uid)
    item = _get_item_exact_dict(iid)
    reviews = _get_reviews_list(uid, iid)

    style = _summarize_review_style(reviews, uid, iid)
    predicted_stars = _compute_stars(user, item, style)
    case_info = _detect_case_from_flags(
    user_exists=bool(user),
    item_exists=bool(item),
    direct_review_exists=bool(style.get("direct_review_exists")),
)

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

    if user_exists and item_exists and direct_review_exists:
        case_number = 1
        case_name = "Case 1: user exists, item exists, direct historical review exists"
        dominant_evidence = "direct_review"
        fallback_policy = "use direct review first, then user and item context"

    elif user_exists and item_exists and not direct_review_exists:
        case_number = 2
        case_name = "Case 2: user exists, item exists, no direct historical review"
        dominant_evidence = "user_and_item_profiles"
        fallback_policy = "combine user behavior and item quality"

    elif not user_exists and item_exists and direct_review_exists:
        case_number = 3
        case_name = "Case 3: user missing, item exists, direct historical review exists"
        dominant_evidence = "direct_review_and_item_profile"
        fallback_policy = "use direct review first, then item context"

    elif not user_exists and item_exists and not direct_review_exists:
        case_number = 4
        case_name = "Case 4: user missing, item exists, no direct historical review"
        dominant_evidence = "item_profile"
        fallback_policy = "use item stars and category context"

    elif user_exists and not item_exists and direct_review_exists:
        case_number = 5
        case_name = "Case 5: user exists, item missing, direct historical review exists"
        dominant_evidence = "direct_review_and_user_profile"
        fallback_policy = "use direct review first, then user behavior"

    elif user_exists and not item_exists and not direct_review_exists:
        case_number = 6
        case_name = "Case 6: user exists, item missing, no direct historical review"
        dominant_evidence = "user_profile"
        fallback_policy = "use user rating tendency and avoid item-specific details"

    elif not user_exists and not item_exists and direct_review_exists:
        case_number = 7
        case_name = "Case 7: user missing, item missing, direct historical review exists"
        dominant_evidence = "direct_review_only"
        fallback_policy = "use direct review evidence only and avoid unsupported profile or item facts"

    else:
        case_number = 8
        case_name = "Case 8: user missing, item missing, no direct historical review"
        dominant_evidence = "default_prior"
        fallback_policy = "use generic fallback and avoid unsupported details"

    result = {
        "case_number": case_number,
        "case_name": case_name,
        "flags": {
            "user_exists": user_exists,
            "item_exists": item_exists,
            "direct_review_exists": direct_review_exists,
        },
        "dominant_evidence": dominant_evidence,
        "fallback_policy": fallback_policy,
    }

    return json.dumps(result, ensure_ascii=False)