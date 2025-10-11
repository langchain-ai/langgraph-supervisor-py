from __future__ import annotations

import os
import time
import html
from typing import Any, Dict, Optional

import requests
from langchain_core.tools import tool

_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
_RETRY_STATUS = {429, 500, 502, 503, 504}


def _require_server_key() -> str:
    key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("SERVER_KEY")
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_MAPS_API_KEY (or SERVER_KEY) for Google Geocoding API access"
        )
    return key


def _require_browser_key() -> str:
    key = os.getenv("GOOGLE_MAPS_BROWSER_KEY") or os.getenv("BROWSER_KEY") or os.getenv(
        "GOOGLE_MAPS_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_MAPS_BROWSER_KEY (or BROWSER_KEY / GOOGLE_MAPS_API_KEY) for Maps JS"
        )
    return key


def _request_with_backoff(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    tries: int = 3,
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(tries):
        try:
            resp = requests.request(method, url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_err = exc
        else:
            if resp.status_code in _RETRY_STATUS and attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        time.sleep(2 ** attempt)
    if last_err:
        raise RuntimeError(f"HTTP request failed for {url}: {last_err}") from last_err
    raise RuntimeError(f"HTTP request failed for {url}")


def _geocode_one(address: str, key: str) -> Dict[str, Any]:
    if not address:
        raise ValueError("address is required for geocoding")
    resp = _request_with_backoff("GET", _GEOCODE_URL, params={"address": address, "key": key})
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


@tool("route_geocode")
def route_geocode(origin_address: str, dest_address: str) -> Dict[str, Any]:
    """Geocode origin and destination addresses into coordinates using Google Geocoding API.

    Returns a dict: {"origin": {...}, "destination": {...}}
    Requires env GOOGLE_MAPS_API_KEY (or SERVER_KEY).
    """
    server_key = _require_server_key()
    origin = _geocode_one(origin_address, server_key)
    dest = _geocode_one(dest_address, server_key)
    return {"origin": origin, "destination": dest}


def _sanitize_mode(mode: Optional[str]) -> str:
    m = (mode or "DRIVING").upper()
    allowed = {"DRIVING", "WALKING", "BICYCLING", "TRANSIT"}
    return m if m in allowed else "DRIVING"


def _build_html(origin_lat: float, origin_lng: float, dest_lat: float, dest_lng: float, *, browser_key: str, travel_mode: str) -> str:
    # Escape nothing but keep template readable; we embed only floats in JS and a known enum.
    tm = _sanitize_mode(travel_mode)
    # Small, self-contained HTML page
    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Route Map</title>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; }}
    #map {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id=\"map\"></div>
  <script>
    const ORIGIN = {{ lat: {origin_lat}, lng: {origin_lng} }};
    const DEST   = {{ lat: {dest_lat},  lng: {dest_lng}  }};
    const TRAVEL_MODE = '{tm}';

    let map, dirSvc, dirRenderer;
    function initMap() {{
      map = new google.maps.Map(document.getElementById('map'), {{
        center: ORIGIN, zoom: 13
      }});
      dirSvc = new google.maps.DirectionsService();
      dirRenderer = new google.maps.DirectionsRenderer({{ map: map }});

      dirSvc.route({{
        origin: ORIGIN,
        destination: DEST,
        travelMode: google.maps.TravelMode[TRAVEL_MODE]
      }}, (res, status) => {{
        if (status === 'OK') {{
          dirRenderer.setDirections(res);
        }} else {{
          alert('Directions request failed: ' + status);
        }}
      }});
    }}
    window.initMap = initMap;
  </script>
  <script async defer src=\"https://maps.googleapis.com/maps/api/js?key={html.escape(browser_key)}&callback=initMap\"></script>
</body>
</html>
"""
    return html_doc


@tool("route_build_map_html", return_direct=False)
def route_build_map_html(
    origin_address: Optional[str] = None,
    dest_address: Optional[str] = None,
    *,
    origin_lat: Optional[float] = None,
    origin_lng: Optional[float] = None,
    dest_lat: Optional[float] = None,
    dest_lng: Optional[float] = None,
    travel_mode: str = "DRIVING",
) -> str:
    """Build an interactive Google Maps HTML showing a route between origin and destination.

    You can pass addresses (origin_address, dest_address) to geocode server-side, or pass
    coordinates directly (origin_lat/origin_lng and dest_lat/dest_lng). If both are provided,
    coordinates take precedence.

    Requires env GOOGLE_MAPS_BROWSER_KEY (or falls back to GOOGLE_MAPS_API_KEY).
    """
    browser_key = _require_browser_key()

    if all(v is not None for v in (origin_lat, origin_lng, dest_lat, dest_lng)):
        o_lat, o_lng, d_lat, d_lng = float(origin_lat), float(origin_lng), float(dest_lat), float(dest_lng)
    else:
        if not origin_address or not dest_address:
            raise ValueError("Provide either coordinates or both origin_address and dest_address")
        server_key = _require_server_key()
        o = _geocode_one(origin_address, server_key)
        d = _geocode_one(dest_address, server_key)
        o_lat, o_lng, d_lat, d_lng = o["lat"], o["lng"], d["lat"], d["lng"]

    return _build_html(o_lat, o_lng, d_lat, d_lng, browser_key=browser_key, travel_mode=travel_mode)

