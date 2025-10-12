from __future__ import annotations
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class Geo(BaseModel):
    lat: float
    lng: float
    radius: Optional[float] = Field(None, description="meters")

class QuerySpec(BaseModel):
    geo: Optional[Geo] = None
    price_band: Optional[str] = None      # "$" | "$$" | "$$$"...
    cuisines: List[str] = []
    diet_restrictions: List[str] = []
    party_size: Optional[int] = None
    time_window: Optional[str] = None     # "today 19:00-21:00"

class UserTasteProfile(BaseModel):
    cuisines: List[str] = []
    disliked: List[str] = []
    ambience: List[str] = []
    spice_level: Optional[str] = None
    price_prior: Optional[str] = None
    history_signals: Dict[str, dict] = {} # e.g., {"yt": {...}, "gmaps_cf": {...}}
    updated_at: Optional[str] = None

class RestaurantDoc(BaseModel):
    place_id: str
    name: str
    address: Optional[str] = None
    geo: Optional[Geo] = None
    price_level: Optional[str] = None
    cuisine_tags: List[str] = []
    features: List[str] = []
    short_desc: Optional[str] = None
    embedding_id: Optional[str] = None
    kg_node_id: Optional[str] = None

class RankedItem(BaseModel):
    place_id: str
    score: float
    why: List[str] = []
    filters_passed: List[str] = []
    cautions: List[str] = []

class RankedList(BaseModel):
    items: List[RankedItem] = []
    rationale: Optional[str] = None

class Evidence(BaseModel):
    photos: List[str] = []
    opening_hours: Optional[dict] = None
    deeplink: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None

class AuditEvent(BaseModel):
    stage: str
    inputs_hash: Optional[str] = None
    outputs_hash: Optional[str] = None
    notes: Optional[str] = None