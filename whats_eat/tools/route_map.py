from __future__ import annotations

import os
import html
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from whats_eat.tools.google_places import _geocode_address


def _require_browser_key() -> str:
    key = os.getenv("GOOGLE_MAPS_BROWSER_KEY") or os.getenv("BROWSER_KEY") or os.getenv(
        "GOOGLE_MAPS_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "Missing GOOGLE_MAPS_BROWSER_KEY (or BROWSER_KEY / GOOGLE_MAPS_API_KEY) for Maps JS"
        )
    return key


# @tool("route_geocode")
# def route_geocode(origin_address: str, dest_address: str) -> Dict[str, Any]:
#     """Geocode origin and destination addresses into coordinates using Google Geocoding API.
#
#     Returns a dict: {"origin": {...}, "destination": {...}}
#     Requires env GOOGLE_MAPS_API_KEY.
#     """
#     origin = _geocode_address(origin_address)
#     dest = _geocode_address(dest_address)
#     return {"origin": origin, "destination": dest}


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
        o = _geocode_address(origin_address)
        d = _geocode_address(dest_address)
        o_lat, o_lng, d_lat, d_lng = o["lat"], o["lng"], d["lat"], d["lng"]

    return _build_html(o_lat, o_lng, d_lat, d_lng, browser_key=browser_key, travel_mode=travel_mode)
