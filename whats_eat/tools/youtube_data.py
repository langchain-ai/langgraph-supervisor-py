
from langchain_core.tools import tool
from typing import Dict, Any, List

@tool("yt_list_subscriptions")
def yt_list_subscriptions() -> List[Dict[str, Any]]:
    """Return the user's subscribed channels (name, id, subscribedAt)."""
    # TODO: call your OAuth'd code (reads token.json) and return rows
    return []

@tool("yt_list_liked_videos")
def yt_list_liked_videos(max_results: int = 50) -> List[Dict[str, Any]]:
    """Return user's liked videos (id, title, channel, publishedAt)."""
    # TODO: wrap your notebook logic
    return []
