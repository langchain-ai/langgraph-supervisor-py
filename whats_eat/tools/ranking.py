# tools/ranking.py
from langchain_core.tools import tool
from typing import List, Dict, Any

@tool("rank_restaurants")
def rank_restaurants(candidates: List[Dict[str, Any]], strategy: str = "hybrid") -> List[Dict[str, Any]]:
    """Score & rank candidate restaurants given YouTube/Places signals."""
    # TODO: implement your heuristics / model scoring here
    return candidates
