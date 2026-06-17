from crewai.tools import tool

_GLOBAL_INTERACTION_TOOL = None


def inject_simulator_tool(tool_instance):
    global _GLOBAL_INTERACTION_TOOL
    _GLOBAL_INTERACTION_TOOL = tool_instance


@tool("Interaction Tool Wrapper")
def interaction_tool_wrapper(query_type: str, target_id: str) -> str:
    """
    Query AgentSociety simulator interaction tool.

    query_type must be one of:
    - user
    - item
    - review_by_user
    - review_by_item
    """
    if _GLOBAL_INTERACTION_TOOL is None:
        return "Error: InteractionTool has not been injected by the Simulator."

    try:
        if query_type == "user":
            return str(_GLOBAL_INTERACTION_TOOL.get_user(user_id=target_id))

        if query_type == "item":
            return str(_GLOBAL_INTERACTION_TOOL.get_item(item_id=target_id))

        if query_type == "review_by_user":
            return str(_GLOBAL_INTERACTION_TOOL.get_reviews(user_id=target_id))

        if query_type == "review_by_item":
            return str(_GLOBAL_INTERACTION_TOOL.get_reviews(item_id=target_id))

        return "Error: Unknown query_type. Use user, item, review_by_user, or review_by_item."

    except Exception as e:
        return f"Error occurred during interaction_tool query: {str(e)}"


def get_interaction_tool():
    return interaction_tool_wrapper