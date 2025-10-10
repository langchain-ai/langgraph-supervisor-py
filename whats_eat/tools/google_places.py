from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import tool

_PLACES_BASE_URL = "https://places.googleapis.com/v1"
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
