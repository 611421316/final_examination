import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Allow running this file from anywhere if it is placed inside project root.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import src.tools.exact_lookup_tools as lookup_tools
except Exception:
    try:
        import exact_lookup_tools as lookup_tools
    except Exception as e:
        print("❌ Cannot import lookup module.")
        print("Run from project root, for example:")
        print("  cd /Users/vcv/Documents/ndhu/LLM/final_examination/final_examination/AgentSocietyChallenge_OpenEvolve")
        print("  PYTHONPATH=. uv run python test_lookup_fixed.py --uid <USER_ID> --iid <ITEM_ID>")
        print("Import error:", repr(e))
        raise


def pretty_json(data: Any, limit: int = 2500) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = repr(data)
    if len(text) > limit:
        return text[:limit] + f"\n... [truncated, total={len(text)} chars]"
    return text


def print_block(title: str, data: Any, limit: int = 2500) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    if isinstance(data, dict):
        print("type       : dict")
        print("found      :", bool(data))
        print("field_count:", len(data))
        print("keys       :", list(data.keys())[:30])
        print(pretty_json(data, limit=limit))
    elif isinstance(data, list):
        print("type       : list")
        print("count      :", len(data))
        if data and isinstance(data[0], dict):
            print("first keys :", list(data[0].keys())[:30])
        print(pretty_json(data[:5], limit=limit))
    else:
        print("type       :", type(data).__name__)
        print(pretty_json(data, limit=limit))


def get_func(*names: str) -> Optional[Callable[..., Any]]:
    for name in names:
        fn = getattr(lookup_tools, name, None)
        if callable(fn):
            return fn
    return None


def call_safely(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call lookup functions that may have different signatures."""
    try:
        return fn(*args, **kwargs)
    except TypeError:
        # Some CrewAI tools expose .run(input_dict)
        if hasattr(fn, "run"):
            return fn.run(kwargs if kwargs else dict(zip(["user_id", "item_id"], args)))
        raise


def run_tool_safely(tool_or_fn: Any, payload: dict) -> Any:
    if hasattr(tool_or_fn, "run"):
        return tool_or_fn.run(payload)
    return call_safely(tool_or_fn, **payload)


def test_user_lookup(uid: str) -> dict:
    fn = get_func("_get_user_exact_dict", "get_user_exact_dict", "fetch_user_data", "get_user")
    if fn is None:
        print("❌ No user lookup function found.")
        print_available_lookup_functions()
        return {}

    print(f"\n🔎 Testing user lookup function: {fn.__name__}({uid!r})")
    try:
        user = call_safely(fn, uid)
    except Exception as e:
        print("❌ USER LOOKUP ERROR:", repr(e))
        return {}

    print_block("USER LOOKUP RESULT", user)
    return user if isinstance(user, dict) else {}


def test_item_lookup(iid: str) -> dict:
    fn = get_func("_get_item_exact_dict", "get_item_exact_dict", "fetch_item_data", "get_item")
    if fn is None:
        print("❌ No item lookup function found.")
        print_available_lookup_functions()
        return {}

    print(f"\n🔎 Testing item lookup function: {fn.__name__}({iid!r})")
    try:
        item = call_safely(fn, iid)
    except Exception as e:
        print("❌ ITEM LOOKUP ERROR:", repr(e))
        return {}

    print_block("ITEM LOOKUP RESULT", item)
    return item if isinstance(item, dict) else {}


def split_reviews(reviews: Any, uid: str, iid: str) -> tuple[list, list, list]:
    if not isinstance(reviews, list):
        return [], [], []

    direct = []
    user_history = []
    item_history = []

    for r in reviews:
        if not isinstance(r, dict):
            continue
        r_uid = r.get("user_id") or r.get("uid")
        r_iid = r.get("item_id") or r.get("business_id") or r.get("iid")

        if r_uid == uid and r_iid == iid:
            direct.append(r)
        elif r_uid == uid:
            user_history.append(r)
        elif r_iid == iid:
            item_history.append(r)

    return direct, user_history, item_history


def test_review_lookup(uid: str, iid: str) -> list:
    fn = get_func("_get_reviews_list", "get_reviews_list", "fetch_review_data", "get_reviews")
    if fn is None:
        print("❌ No review lookup function found.")
        print_available_lookup_functions()
        return []

    print(f"\n🔎 Testing review lookup function: {fn.__name__}(uid={uid!r}, iid={iid!r})")
    try:
        reviews = call_safely(fn, uid, iid)
    except Exception as e:
        print("❌ REVIEW LOOKUP ERROR:", repr(e))
        return []

    direct, user_history, item_history = split_reviews(reviews, uid, iid)

    print_block("ALL MERGED REVIEWS RESULT", reviews, limit=3500)
    print_block("DIRECT REVIEWS user_id + item_id", direct, limit=2500)
    print_block("USER HISTORY REVIEWS same user_id", user_history, limit=2500)
    print_block("ITEM HISTORY REVIEWS same item_id", item_history, limit=2500)

    return reviews if isinstance(reviews, list) else []


def test_prediction_context(uid: str, iid: str) -> None:
    tool = getattr(lookup_tools, "build_prediction_context", None)
    if tool is None:
        print("\n⚠️ build_prediction_context not found; skipping context test.")
        return

    print("\n🔎 Testing build_prediction_context")
    try:
        raw = run_tool_safely(tool, {"user_id": uid, "item_id": iid})
        context = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        print("❌ BUILD CONTEXT ERROR:", repr(e))
        return

    print_block("PREDICTION CONTEXT", context, limit=5000)

    if isinstance(context, dict):
        case = context.get("case", {}) or {}
        review_style = context.get("review_style", {}) or {}
        print("\n[CONTEXT SUMMARY]")
        print("user_exists                :", case.get("user_exists"))
        print("item_exists                :", case.get("item_exists"))
        print("direct_review_exists       :", case.get("direct_review_exists"))
        print("case_number                :", case.get("case_number"))
        print("case_name                  :", case.get("case_name"))
        print("predicted_stars            :", context.get("predicted_stars"))
        print("direct_review_stars        :", review_style.get("direct_review_stars"))
        print("user_review_count_used     :", review_style.get("user_review_count_used"))
        print("item_review_count_used     :", review_style.get("item_review_count_used"))
        print("user_history_average_stars :", review_style.get("user_history_average_stars"))
        print("item_history_average_stars :", review_style.get("item_history_average_stars"))


def print_available_lookup_functions() -> None:
    print("\n[Available functions in lookup module]")
    names = []
    for name in dir(lookup_tools):
        if name.startswith("__"):
            continue
        obj = getattr(lookup_tools, name)
        if callable(obj):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(...)"
            names.append(f"- {name}{sig}")
    print("\n".join(names[:80]))


def test_all(uid: str, iid: str, include_context: bool = True) -> None:
    print("#" * 100)
    print("LOOKUP FUNCTION TEST")
    print("#" * 100)
    print("uid:", uid)
    print("iid:", iid)
    print("module:", getattr(lookup_tools, "__file__", lookup_tools))

    user = test_user_lookup(uid)
    item = test_item_lookup(iid)
    reviews = test_review_lookup(uid, iid)
    direct, user_history, item_history = split_reviews(reviews, uid, iid)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print("user_found                 :", bool(user))
    print("user_field_count           :", len(user))
    print("item_found                 :", bool(item))
    print("item_field_count           :", len(item))
    print("merged_reviews_found       :", len(reviews))
    print("direct_reviews_found       :", len(direct))
    print("user_history_reviews_found :", len(user_history))
    print("item_history_reviews_found :", len(item_history))

    if include_context:
        test_prediction_context(uid, iid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test exact lookup functions for user, item, and review data.")
    parser.add_argument("--uid", "--user-id", dest="uid", default="vWn1N7e-H276Z8Rii8_NIA")
    parser.add_argument("--iid", "--item-id", dest="iid", default="mLNSOU8Ki0Fm09xd6ZKkcA")
    parser.add_argument("--no-context", action="store_true", help="Skip build_prediction_context test.")
    parser.add_argument("--list-functions", action="store_true", help="Print available functions in lookup module.")
    args = parser.parse_args()

    if args.list_functions:
        print_available_lookup_functions()
        return

    test_all(args.uid, args.iid, include_context=not args.no_context)


if __name__ == "__main__":
    main()
