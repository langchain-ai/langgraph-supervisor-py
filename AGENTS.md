# AGENTS.md — WhatsEat Supervisor-based Multi-Agent (Aligned with `langgraph_supervisor` style)

> This document is an **implementation guide + code conventions**. Requirements: **reuse** the existing capabilities in your fork’s `langgraph_supervisor/` (`create_supervisor`, `create_handoff_tool`, `create_forward_message_tool`, message and state types, etc.), **keep the same code style and naming habits**, and provide **clearly readable** directory and file names.  
> Convention: This document treats **`langgraph_supervisor`** in the repo as the authoritative package name (if historical docs contain the misspelling `langgraph_supervisior`, this file takes precedence).

------

## 0. Code style and naming conventions (consistent with `langgraph_supervisor`)

- **Naming**
  - Directories & files: `snake_case`, e.g., `places_agent.py`, `fetch_place_photos.py`.
  - Tool functions (tool names): `snake_case`, with a clear verb at the start: `place_text_search`, `build_gmaps_deeplink`.
  - Pydantic models: `PascalCase`, e.g., `QuerySpec`, `UserTasteProfile`.
  - Agent constructors: `build_<role>_agent()`.
- **Types & comments**
  - Full **type hints**; module-level docstrings briefly explain inputs/outputs.
  - Pydantic v2 (`BaseModel`) defines tool input parameters and cross-agent state objects.
- **I/O constraints**
  - Tools return **small and stable** JSON/models; do not pass through large raw payloads from third-party APIs.
  - Network tools must have `timeout` and **at most 2 exponential backoff retries**, and degrade behavior for 429/5xx.
- **Structured state**
  - Only put **structured objects** into `state`. Supervisor uses `output_mode="last_message"` to avoid bloated chat text.

------

## 1. Directory structure (at the fork’s root)

```
langgraph-supervisor-py/
├─ langgraph_supervisor/                 # Keep as-is (do not modify public APIs)
│  └─ ... (create_supervisor, create_handoff_tool, etc.)
├─ apps/whatseat/
│  ├─ agents/
│  │  ├─ uia_agent.py                    # User-Intent Agent
│  │  ├─ upa_agent.py                    # User Profile Agent
│  │  ├─ rpa_agent.py                    # Restaurant Profile Agent
│  │  ├─ pfa_agent.py                    # Preference Fusion Agent
│  │  └─ eea_agent.py                    # Evidence & Enrichment Agent
│  ├─ tools/
│  │  ├─ places.py                       # Places/TextSearch/Details/Deeplink
│  │  ├─ photos.py                       # Photos CDN URL
│  │  ├─ youtube.py                      # YouTube history & topics
│  │  ├─ scoring.py                      # Ranking / multi-objective fusion
│  │  ├─ kg.py                           # ER extraction & KG upsert (placeholder; can be added later)
│  │  └─ vector.py                       # Vectorization and ingestion (placeholder; can be added later)
│  ├─ supervisor/
│  │  ├─ prompt.py                       # Supervisor routing rules
│  │  └─ workflow.py                     # create_supervisor + handoff definitions
│  ├─ schemas.py                         # State & contract models (Pydantic)
│  ├─ config.py                          # Constants / env vars / retry strategy
│  ├─ run_demo.py                        # End-to-end example
│  └─ tests/                             # Unit & integration tests
└─ ...
```

------

## 2. Shared State and data contracts (`apps/whatseat/schemas.py`)

```
# apps/whatseat/schemas.py
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
```

> On the graph state, extend `MessagesState`: only add the following keys: `query_spec`, `user_profile`, `candidates`, `ranked`, `evidence`, `audit`.

------

## 3. Tools layer (`apps/whatseat/tools/*.py`)

> Expose with `@tool` + `args_schema`; **network tools** uniformly use `requests`, **timeout + 2 backoff retries**; degrade for 429/5xx. Below shows the “shells” and **minimal projections**—paste your notebook implementations into the `# TODO` parts.

### 3.1 Places (`apps/whatseat/tools/places.py`)

```
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import os, time, requests

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

class TextSearchInput(BaseModel):
    query: str
    region_code: Optional[str] = None
    location_bias: Optional[str] = Field(None, description="circle:2000@lat,lng")

def _get(url: str, params: dict, tries: int = 3, timeout: int = 20):
    for t in range(tries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** t)
            continue
        r.raise_for_status()
        return r
    return r  # Return the last response; caller can decide how to handle

@tool("place_text_search", args_schema=TextSearchInput)
def place_text_search(query: str, region_code: Optional[str] = None,
                      location_bias: Optional[str] = None) -> Dict[str, Any]:
    assert GOOGLE_API_KEY, "Missing GOOGLE_MAPS_API_KEY"
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_API_KEY}
    if region_code: params["region"] = region_code
    if location_bias: params["locationbias"] = location_bias
    r = _get(url, params)
    data = r.json()
    items = []
    for it in data.get("results", [])[:25]:
        items.append({
            "place_id": it.get("place_id"),
            "name": it.get("name"),
            "address": it.get("formatted_address"),
            "location": it.get("geometry", {}).get("location"),
            "photo_refs": [p.get("photo_reference") for p in it.get("photos", [])] if it.get("photos") else []
        })
    return {"candidates": items}

class DetailsBatchInput(BaseModel):
    place_ids: List[str]

@tool("place_details_batch", args_schema=DetailsBatchInput)
def place_details_batch(place_ids: List[str]) -> List[Dict[str, Any]]:
    assert GOOGLE_API_KEY, "Missing GOOGLE_MAPS_API_KEY"
    out: List[Dict[str, Any]] = []
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = "place_id,name,formatted_address,geometry,opening_hours,website,international_phone_number,rating,user_ratings_total,photo,price_level"
    for pid in place_ids:
        r = _get(url, {"place_id": pid, "key": GOOGLE_API_KEY, "fields": fields})
        res = r.json().get("result", {})
        out.append(res)
    return out

class DeeplinkInput(BaseModel):
    place_id: str
    mode: str = "driving"
    origin: Optional[str] = None

@tool("build_gmaps_deeplink", args_schema=DeeplinkInput)
def build_gmaps_deeplink(place_id: str, mode: str = "driving", origin: Optional[str] = None) -> str:
    base = "https://www.google.com/maps/dir/?api=1"
    dest = f"destination_place_id={place_id}"
    mode_q = f"&travelmode={mode}"
    ori_q = f"&origin={origin}" if origin else ""
    return f"{base}&{dest}{mode_q}{ori_q}"
```

### 3.2 Photos (`apps/whatseat/tools/photos.py`)

```
from typing import Dict
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import os, requests

GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

class PhotoInput(BaseModel):
    photo_reference: str
    max_w: int = Field(800, ge=1, le=1600)
    max_h: int = Field(800, ge=1, le=1600)

@tool("fetch_place_photos", args_schema=PhotoInput)
def fetch_place_photos(photo_reference: str, max_w: int = 800, max_h: int = 800) -> Dict[str, str]:
    assert GOOGLE_API_KEY, "Missing GOOGLE_MAPS_API_KEY"
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {"photoreference": photo_reference, "maxwidth": max_w, "maxheight": max_h, "key": GOOGLE_API_KEY}
    resp = requests.get(url, params=params, allow_redirects=False, timeout=20)
    final_url = resp.headers.get("Location") or resp.url
    return {"photo_url": final_url}
```

### 3.3 YouTube (`apps/whatseat/tools/youtube.py`)

```
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

class YTProfileInput(BaseModel):
    user_id: Optional[str] = Field(None, description="App user id for OAuth mapping")
    max_items: int = 50

@tool("yt_fetch_history", args_schema=YTProfileInput)
def yt_fetch_history(user_id: Optional[str] = None, max_items: int = 50) -> Dict[str, Any]:
    assert YOUTUBE_API_KEY, "Missing YOUTUBE_API_KEY"
    # TODO: Paste the Data API v3 implementation from your notebook; return VideoMeta[]
    return {"videos": []}

class YTInferInput(BaseModel):
    videos: List[Dict[str, Any]]

@tool("yt_topics_infer", args_schema=YTInferInput)
def yt_topics_infer(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    # TODO: Paste your topic/cuisine keyword extraction logic
    return {"topic_keywords": [], "cuisine_keywords": [], "creators_top": []}
```

> Other tools: `gmaps_likes_fetch`, `item_item_cf`, `profile_merge`, `entity_relation_extract`, `kg_upsert`, `vector_embed_and_upsert`, `scoring_pipeline` can be stubbed following the same style.

------

## 4. Sub-agents (`apps/whatseat/agents/*.py`)

> Uniformly build with `create_react_agent`; one constructor per file, named `build_<role>_agent()`. Prompts should **only state operating conventions** and **output contracts**.

Example: **RPA** (Restaurant Profile Agent, `rpa_agent.py`)

```
# apps/whatseat/agents/rpa_agent.py
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from apps.whatseat.tools.places import place_text_search, place_details_batch
from apps.whatseat.tools.kg import kg_upsert
from apps.whatseat.tools.vector import vector_embed_and_upsert

PROMPT = (
    "You are the Restaurant Profile Agent.\n"
    "- Use place_text_search to collect candidates (20-50).\n"
    "- Use place_details_batch to fetch details.\n"
    "- Project results to minimal RestaurantDoc fields.\n"
    "- Upsert ER/Vector if available (best-effort). Return only compact list."
)

def build_rpa_agent():
    return create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=[place_text_search, place_details_batch, kg_upsert, vector_embed_and_upsert],
        name="restaurant_profile_agent",
        prompt=PROMPT,
    )
```

Others: `uia_agent.py` (only extracts `QuerySpec`), `upa_agent.py` (YouTube/CF fusion), `pfa_agent.py` (scoring), `eea_agent.py` (evidence) follow the same pattern.

------

## 5. Supervisor and handoff (`apps/whatseat/supervisor/workflow.py` & `prompt.py`)

### 5.1 Routing prompt (`prompt.py`)

```
# apps/whatseat/supervisor/prompt.py
SUPERVISOR_PROMPT = """
You are the supervisor. Route one worker at a time.
Rules:
- Intent parsing -> UIA; user profile -> UPA; restaurant candidates -> RPA; scoring -> PFA; evidence -> EEA.
- Do not call external APIs yourself; always delegate via handoff tools.
- Prefer compact JSON; store structured objects in state, not in text history.
Fallbacks:
- If RPA candidates < 3, ask UIA to relax location/radius/price.
- If no OAuth, skip UPA and proceed with session-only preferences.
"""
```

### 5.2 Workflow and handoff (`workflow.py`)

```
# apps/whatseat/supervisor/workflow.py
from __future__ import annotations
from typing import Annotated
from langgraph_supervisor import (
    create_supervisor,
    create_handoff_tool,
    create_forward_message_tool,
)
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.graph.message import MessagesState
from langgraph.types import Command, Send
from langchain_openai import ChatOpenAI

from apps.whatseat.agents.uia_agent import build_uia_agent
from apps.whatseat.agents.upa_agent import build_upa_agent
from apps.whatseat.agents.rpa_agent import build_rpa_agent
from apps.whatseat.agents.pfa_agent import build_pfa_agent
from apps.whatseat.agents.eea_agent import build_eea_agent
from .prompt import SUPERVISOR_PROMPT

def _handoff_to(agent_name: str, tool_name: str, description: str):
    # Use the same handoff style as langgraph_supervisor: take a task description + inject state
    def _factory():
        @create_handoff_tool(agent_name=agent_name, name=tool_name, description=description)
        def _tool(
            task_description: Annotated[str, "What the next agent should do"],
            state: Annotated[MessagesState, InjectedState],
        ) -> Command:
            # Only pass necessary fields to avoid blowing up history
            payload = {
                **state,
                "messages": [{"role": "user", "content": task_description}],
                # Structured objects to be read downstream: query_spec/user_profile/candidates/ranked/evidence
            }
            return Command(goto=[Send(agent_name, payload)], graph=Command.PARENT)
        return _tool
    return _factory()

def build_workflow():
    model = ChatOpenAI(model="gpt-4o")

    agents = [
        build_uia_agent(),
        build_upa_agent(),
        build_rpa_agent(),
        build_pfa_agent(),
        build_eea_agent(),
    ]

    handoffs = [
        _handoff_to("user_intent_agent", "delegate_to_uia", "Extract/complete QuerySpec"),
        _handoff_to("user_profile_agent", "delegate_to_upa", "Build UserTasteProfile"),
        _handoff_to("restaurant_profile_agent", "delegate_to_rpa", "Collect & structure candidates"),
        _handoff_to("preference_fusion_agent", "delegate_to_pfa", "Score & rank candidates"),
        _handoff_to("evidence_agent", "delegate_to_eea", "Enrich Top-K with photos/hours/deeplinks"),
        create_forward_message_tool("supervisor"),
    ]

    return create_supervisor(
        agents=agents,
        tools=handoffs,
        model=model,
        prompt=SUPERVISOR_PROMPT,
        output_mode="last_message",
        add_handoff_messages=True,
        supervisor_name="supervisor",
    )
```

------

## 6. Running and memory (`apps/whatseat/run_demo.py`)

```
import uuid
from langgraph.checkpoint.memory import InMemorySaver
from apps.whatseat.supervisor.workflow import build_workflow

if __name__ == "__main__":
    app = build_workflow().compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    out = app.invoke(
        {"messages": [{"role": "user",
                       "content": "Find student-budget, heavy-flavor ramen near Changi, give 3 with images and navigation"}]},
        config=cfg
    )
    for m in out["messages"]:
        m.pretty_print()
```

------

## 7. Tests and quality gates (`apps/whatseat/tests/`)

- **Tool unit tests**: `test_tools_places.py`, `test_tools_photos.py`, `test_tools_youtube.py`
  - Validate input schemas, lifecycles (429/5xx retries), and stability of output fields.
- **Agent integration**: `test_agents_flow.py`
  - Happy path: UIA→UPA→RPA→PFA→EEA.
  - Degradations: no OAuth, insufficient RPA recall, missing images.
- **End-to-end**: fix 3 query samples; assert `RankedList.items[0].score` is within a reasonable range and contains `why[]`.

------

## 8. Unified structure for the frontend

```
{
  "cards": [
    {
      "place_id": "ChIJ...",
      "name": "Ramen Keisuke",
      "distance_km": 1.2,
      "price_level": "$",
      "tags": ["ramen","spicy","late-night"],
      "why": ["close by","matches YouTube: ramen","budget-friendly"],
      "photos": ["...jpg","...jpg"],
      "opens": {"today_is_open": true, "closes_at": "23:00"},
      "deeplink": "https://www.google.com/maps/dir/?api=1&..."
    }
  ],
  "rationale": "Based on student budget and heavy-flavor preference, selected 3 from 36 candidates within 2km…"
}
```

------

## 9. Suggested implementation order

1. `places.py` + `photos.py` → `rpa_agent.py` + `eea_agent.py` → get the minimal closed loop working (no profile).
2. `youtube.py` + `upa_agent.py` → `pfa_agent.py` for session/profile fusion.
3. Integrate `kg.py`/`vector.py` → in `rpa_agent.py` write and reference evidence to strengthen explainability.

------

### Appendix: Standard constructor signatures for each agent file

- `apps/whatseat/agents/uia_agent.py` → `def build_uia_agent(): ...` (outputs only `QuerySpec`)
- `apps/whatseat/agents/upa_agent.py` → `def build_upa_agent(): ...`
- `apps/whatseat/agents/rpa_agent.py` → `def build_rpa_agent(): ...`
- `apps/whatseat/agents/pfa_agent.py` → `def build_pfa_agent(): ...`
- `apps/whatseat/agents/eea_agent.py` → `def build_eea_agent(): ...`

> The above naming/style is consistent with the APIs and examples of `langgraph_supervisor`. After pasting your existing Jupyter code into the corresponding tool implementations, you can run directly via `run_demo.py`.
