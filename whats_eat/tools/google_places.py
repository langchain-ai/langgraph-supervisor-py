
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import os

# You can swap this to googlemaps Python client if you prefer.
# Keep network logic isolated so agent tests can stub these functions.

@tool("places_text_search", return_direct=False)
def places_text_search(query: str, region: str = "SG") -> Dict[str, Any]:
    """Text search a place on Google Places API. Returns JSON-like dict."""
    # TODO: implement with your existing notebook logic (requests/googlemaps)
    # Expect: name, place_id, address, lat/lng
    return {"ok": False, "reason": "not implemented yet", "query": query, "region": region}

@tool("places_fetch_photos")
def places_fetch_photos(place_id: str, max_count: int = 4, max_w: int = 800, max_h: int = 800) -> List[str]:
    """Return direct photo URLs for a place_id (length <= max_count)."""
    # TODO: wrap your existing photo code; return list of URLs
    return []
