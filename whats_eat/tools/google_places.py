from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

_PLACES_BASE_URL = "https://places.googleapis.com/v1"
_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _require_api_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_MAPS_API_KEY environment variable for Google Places API access")
    return key


def _request_with_backoff(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    tries: int = 3,
    timeout: int = 20,
) -> requests.Response:
    """Execute an HTTP request with exponential backoff for retryable status codes."""
    last_error: Optional[Exception] = None
    for attempt in range(tries):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code in _RETRY_STATUS and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return response
        time.sleep(2 ** attempt)
    if last_error:
        raise RuntimeError(
            f"Failed to call {url}: {last_error}") from last_error
    raise RuntimeError(f"Failed to call {url}")


def _call_places(
    method: str,
    path: str,
    *,
    field_mask: Optional[str],
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrapper around the Places API handling headers and retries."""
    headers = {"X-Goog-Api-Key": _require_api_key()}
    if field_mask:
        headers["X-Goog-FieldMask"] = field_mask
    response = _request_with_backoff(
        method,
        f"{_PLACES_BASE_URL}{path}",
        headers=headers,
        json_body=json_body,
        params=params,
    )
    if response.headers.get("Content-Type", "").startswith("application/json"):
        return response.json()
    return {"raw": response.content}


def _normalize_place(place: Dict[str, Any]) -> Dict[str, Any]:
    display_name = place.get("displayName") or {}
    location = place.get("location") or {}
    place_id = place.get("id")
    if isinstance(place_id, str) and place_id.startswith("places/"):
        short_id = place_id.split("/", 1)[1]
    else:
        short_id = place_id
    photos = place.get("photos") or []
    normalized: Dict[str, Any] = {
        "place_id": short_id,
        "raw_place_id": place_id,
        "name": display_name.get("text"),
        "formatted_address": place.get("formattedAddress"),
        "location": (
            {
                "lat": location.get("latitude"),
                "lng": location.get("longitude"),
            }
            if location
            else None
        ),
        "google_maps_uri": place.get("googleMapsUri"),
        "rating": place.get("rating"),
        "user_ratings_total": place.get("userRatingCount") or place.get("userRatingsTotal"),
        "price_level": place.get("priceLevel"),
        "types": place.get("types") or [],
        "photo_names": [photo.get("name") for photo in photos if photo.get("name")],
    }
    summary = place.get("generativeSummary") or {}
    overview = summary.get("overview") or {}
    if overview.get("text"):
        normalized["summary"] = overview["text"]
    return normalized


def _geocode_address(address: str) -> Dict[str, Any]:
    """Geocode an address into coordinates using Google Geocoding API."""
    if not address:
        raise ValueError("address is required for geocoding")
    params = {"address": address, "key": _require_api_key()}
    resp = _request_with_backoff("GET", _GEOCODE_URL, params=params)
    data = resp.json()
    status = data.get("status")
    if status != "OK":
        raise RuntimeError(f"Geocoding failed: {status} {data.get('error_message')}")
    result = data["results"][0]
    loc = result["geometry"]["location"]
    return {
        "lat": loc["lat"],
        "lng": loc["lng"],
        "formatted": result.get("formatted_address"),
        "place_id": result.get("place_id"),
        "types": result.get("types") or [],
    }


def _ensure_place_path(place_id: str) -> str:
    return place_id if place_id.startswith("places/") else f"places/{place_id}"


def _photo_to_url(photo_name: str, max_w: int, max_h: int) -> Optional[str]:
    params = {
        "maxWidthPx": max_w,
        "maxHeightPx": max_h,
        "skipHttpRedirect": "true",
    }
    response = _request_with_backoff(
        "GET",
        f"{_PLACES_BASE_URL}/{photo_name}/media",
        headers={"X-Goog-Api-Key": _require_api_key()},
        params=params,
    )
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("application/json"):
        try:
            data = response.json()
        except ValueError:
            return None
        return data.get("photoUri")
    location = response.headers.get("Location")
    if location:
        return location
    return response.url


@tool("place_geocode")
def place_geocode(address: str) -> Dict[str, Any]:
    """Geocode an address (including postal code) into coordinates using Google Geocoding API.

    This tool converts any address or postal code into latitude and longitude coordinates.
    Returns a dict with: {"lat": ..., "lng": ..., "formatted": ..., "place_id": ..., "types": [...]}

    Requires env GOOGLE_MAPS_API_KEY.
    """
    return _geocode_address(address)


@tool("places_text_search", return_direct=False)
def places_text_search(query: str, region: str = "SG") -> Dict[str, Any]:
    """Text search a place on Google Places API. Returns JSON-like dict."""
    field_mask = ",".join(
        [
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.googleMapsUri",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.types",
            "places.generativeSummary",
        ]
    )
    payload: Dict[str, Any] = {"textQuery": query, "pageSize": 10}
    if region:
        payload["regionCode"] = region
    data = _call_places("POST", "/places:searchText",
                        field_mask=field_mask, json_body=payload)
    places = [_normalize_place(item) for item in data.get("places", [])]
    return {"query": query, "region": region, "candidates": places}


@tool("places_coordinate_search", return_direct=False)
def places_coordinate_search(
    latitude: float,
    longitude: float,
    radius: float = 1500.0,
    max_results: int = 20,
    rank_by: str = "POPULARITY",
) -> Dict[str, Any]:
    """Search for nearby restaurants using coordinates and radius on Google Places API.

    Args:
        latitude: Center point latitude
        longitude: Center point longitude
        radius: Search radius in meters (default: 1500.0)
        max_results: Maximum number of results to return (default: 20, max: 20)
        rank_by: Ranking preference - "POPULARITY" or "DISTANCE" (default: "POPULARITY")

    Returns JSON-like dict with nearby restaurant candidates.
    """
    field_mask = ",".join(
        [
            "places.id",
            "places.displayName",
            "places.formattedAddress",
            "places.location",
            "places.googleMapsUri",
            "places.rating",
            "places.userRatingCount",
            "places.priceLevel",
            "places.types",
            "places.generativeSummary",
        ]
    )

    # Ensure max_results doesn't exceed API limit
    max_results = min(max_results, 20)

    # Validate rank_by parameter
    rank_preference = rank_by.upper() if rank_by.upper() in {"POPULARITY", "DISTANCE"} else "POPULARITY"

    payload: Dict[str, Any] = {
        "includedTypes": ["restaurant"],
        "maxResultCount": max_results,
        "rankPreference": rank_preference,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius
            }
        }
    }

    data = _call_places("POST", "/places:searchNearby",
                        field_mask=field_mask, json_body=payload)
    places = [_normalize_place(item) for item in data.get("places", [])]
    return {
        "center": {"lat": latitude, "lng": longitude},
        "radius": radius,
        "rank_by": rank_preference,
        "candidates": places
    }


@tool("places_fetch_photos")
def places_fetch_photos(
    place_id: str,
    max_count: int = 4,
    max_w: int = 800,
    max_h: int = 800,
) -> List[str]:
    """Return direct photo URLs for a place_id (length <= max_count)."""
    field_mask = "photos.name"
    data = _call_places(
        "GET", f"/{_ensure_place_path(place_id)}", field_mask=field_mask)
    photo_entries = data.get("photos", [])[:max_count]
    urls: List[str] = []
    for photo in photo_entries:
        name = photo.get("name")
        if not name:
            continue
        url = _photo_to_url(name, max_w=max_w, max_h=max_h)
        if url:
            urls.append(url)
    return urls
