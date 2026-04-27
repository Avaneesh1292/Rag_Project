from langchain_core.tools import tool


@tool
def ask_for_clarification(reason: str) -> str:
    """Ask the user for missing or unclear context."""
    return reason
