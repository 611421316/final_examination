"""
Compatibility wrapper for older CrewAI simulation pipelines.

This file keeps backward compatibility with code that imports:

    from src.tools.interaction_tool_wrapper import inject_simulator_tool

It also exposes:
- interaction_tool_wrapper
- get_interaction_tool
- inject_simulator_tool

The main exact lookup logic is still in:
    src/tools/exact_lookup_tools.py
"""

from __future__ import annotations

from typing import Any

from crewai.tools import tool

try:
    from src.tools.exact_lookup_tools import (
        lookup_user_by_id,
        lookup_item_by_id,
        lookup_reviews_by_user,
        lookup_reviews_by_item,
        lookup_reviews_by_user_and_item,
        build_prediction_context,
        determine_prediction_case,
    )
except Exception:
    from exact_lookup_tools import (
        lookup_user_by_id,
        lookup_item_by_id,
        lookup_reviews_by_user,
        lookup_reviews_by_item,
        lookup_reviews_by_user_and_item,
        build_prediction_context,
        determine_prediction_case,
    )


@tool("interaction_tool_wrapper")
def interaction_tool_wrapper(query_type: str, user_id: str = "", item_id: str = "") -> str:
    """
    Compatibility wrapper for older CrewAI pipelines.

    query_type:
    - user
    - item
    - review_by_user
    - review_by_item
    - review_by_user_and_item
    - prediction_context
    - prediction_case
    """
    try:
        query_type = str(query_type or "").strip()

        if query_type == "user":
            return lookup_user_by_id.run(user_id=user_id)

        if query_type == "item":
            return lookup_item_by_id.run(item_id=item_id)

        if query_type == "review_by_user":
            return lookup_reviews_by_user.run(user_id=user_id)

        if query_type == "review_by_item":
            return lookup_reviews_by_item.run(item_id=item_id)

        if query_type == "review_by_user_and_item":
            return lookup_reviews_by_user_and_item.run(
                user_id=user_id,
                item_id=item_id,
            )

        if query_type == "prediction_context":
            return build_prediction_context.run(
                user_id=user_id,
                item_id=item_id,
            )

        if query_type == "prediction_case":
            return determine_prediction_case.run(
                user_id=user_id,
                item_id=item_id,
            )

        return (
            "Invalid query_type. Use one of: user, item, review_by_user, "
            "review_by_item, review_by_user_and_item, prediction_context, "
            "prediction_case."
        )

    except Exception as e:
        return f"Error occurred during interaction_tool query: {str(e)}"


def get_interaction_tool():
    """
    Return the legacy wrapper tool.
    """
    return interaction_tool_wrapper


def inject_simulator_tool(*args: Any, **kwargs: Any):
    """
    Backward-compatible function expected by crewai_simulation_agent.py.

    Some older pipeline files import this function and expect it to return one
    or more CrewAI tools that should be injected into the simulator agent.

    We return the exact lookup tools plus the compatibility wrapper.
    """
    return [
        build_prediction_context,
        determine_prediction_case,
        lookup_user_by_id,
        lookup_item_by_id,
        lookup_reviews_by_user,
        lookup_reviews_by_item,
        lookup_reviews_by_user_and_item,
        interaction_tool_wrapper,
    ]
