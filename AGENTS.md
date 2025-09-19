# AGENTS.md — Supervisor-based Multi-Agent Design (WhatsEat)

> This document guides **code generation** (and reviewers) to implement our supervisor-orchestrated agent system with LangGraph. It specifies agent roles, tool contracts, handoff rules, shared state, prompts, wiring, and validation checks. The design follows LangGraph’s **supervisor + handoff tools** pattern and **prebuilt ReAct agents** for worker nodes. [LangChain AI+2LangChain AI+2](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)

------

## 1) High-level topology

- **Supervisor**: routes work, one agent at a time; never calls external APIs directly. Uses **handoff tools** to pass control and a minimal task payload to workers. [LangChain AI+1](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)
- **Workers (prebuilt agents)**: `create_react_agent` with small, stable toolboxes. They use tools in a loop until done, then return a compact JSON. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/agents/?utm_source=chatgpt.com)
- **Tools**: Python callables wrapped with `@tool` (Pydantic `args_schema`). Keep inputs/outputs **small and structured**; never dump raw API payloads. [LangChain+2LangChain+2](https://python.langchain.com/docs/how_to/custom_tools/?utm_source=chatgpt.com)

Sequence (baseline, serial): **UIA → UPA → RPA → PFA → EEA → user**. Parallel/conditional branches can be added later. [LangChain AI](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)

------

## 2) Shared state & data contracts

We extend LangGraph’s `MessagesState` with project fields. Workers read/write only what they need; Supervisor owns routing.

```
State = {
  "messages": [...],                            # LangGraph native
  "query_spec": QuerySpec | None,               # UIA output
  "user_profile": UserTasteProfile | None,      # UPA output (long/short term)
  "candidates": list[RestaurantDoc] | None,     # RPA output
  "ranked": RankedList | None,                  # PFA output
  "evidence": dict[str, Evidence] | None,       # EEA output by place_id
  "audit": list[AuditEvent],                    # scoring & routing snapshots
}
```

**Schemas (Pydantic recommended)**

- `QuerySpec{ geo{lat,lng,radius}, price_band, cuisines[], diet_restrictions[], party_size, time_window }`
- `UserTasteProfile{ cuisines[], disliked[], ambience[], spice_level?, price_prior, history_signals{yt,gmaps_cf}, updated_at }`
- `RestaurantDoc{ place_id, name, address, geo, price_level?, cuisine_tags[], features[], text_blob, embedding_id?, kg_node_id? }`
- `RankedList{ items:[{place_id, score, why[], filters_passed[], cautions[]}], rationale }`
- `Evidence{ photos:[url], opening_hours, deeplink, phone?, website? }`
- `AuditEvent{ stage, inputs_hash, outputs_hash, notes }`

> Use `output_mode="last_message"` on the supervisor so long histories don’t bloat; store structured objects in `state` instead of chat text. [LangChain AI](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)

------

## 3) Worker agents & prompts

Each worker is built with `create_react_agent(model, tools=[...], name=...)`. Keep prompts **operational**: when to use which tool, expected JSON, and guardrails. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/agents/?utm_source=chatgpt.com)

### 3.1 UIA — User-Intent & Constraints Agent

- **Goal**: extract/complete `QuerySpec` from user utterances; ask clarifying questions if needed; *no external API calls*.
- **Output**: a single `QuerySpec` JSON.

Prompt points:

- Only produce `QuerySpec`. If any of {location, price_band, cuisines, diet} is missing, ask a short follow-up.
- Do not search; return control to supervisor when `QuerySpec` is ready.

### 3.2 UPA — User Profile Construction Agent

- **Goal**: build `UserTasteProfile` from YouTube history, Google Maps Likes (OAuth), and item-item CF.
- **Tools**: `yt_fetch_history`, `yt_topics_infer`, `gmaps_likes_fetch`, `item_item_cf`, `profile_merge`.
- **Degrade** gracefully (session-only profile) when no OAuth.

Prompt points:

- Call only authorized tools; otherwise produce a session-level profile from `QuerySpec`.
- Output sparse, interpretable features; no raw dumps.

### 3.3 RPA — Restaurant Profile Construction Agent

- **Goal**: collect 20–50 candidates via Places search + Details; run ER extraction; upsert to KG & vector index; return compact `RestaurantDoc[]`.
- **Tools**: `place_text_search`, `place_details_batch`, `entity_relation_extract`, `kg_upsert`, `vector_embed_and_upsert`, `hybrid_collect_candidates`.

Prompt points:

- Keep each candidate minimal (`place_id, name, geo, price_level?, cuisine_tags, short_desc`).
- If recall < 3, retry with refined queries at most twice.

### 3.4 PFA — Preference Fusion & Scoring Agent

- **Goal**: fuse `QuerySpec + UserTasteProfile + RestaurantDoc[]` into `RankedList` with reasons.
- **Tools**: `scoring_pipeline` (performs filter → similarity/weights → MMR/diversity).

Prompt points:

- Apply **hard filters** first (radius/budget/diet); then multi-objective scoring; add `why[]`/`cautions[]` per item.

### 3.5 EEA — Evidence & Enrichment Agent

- **Goal**: enrich Top-K with photos, hours, deeplinks; produce `{place_id -> Evidence}`.
- **Tools**: `fetch_place_photos`, `opening_hours_normalize`, `build_gmaps_deeplink`.

Prompt points:

- Fetch ≤3 photos per place; include hours summary; produce a final card-friendly JSON.

### 3.6 FLA — Feedback & Learning Agent (optional)

- **Goal**: log user feedback; update weights for profile & ranker.
- **Tools**: `log_feedback`.

------

## 4) Tool layer (contracts & guidance)

Define tools with LangChain’s `@tool` and `args_schema`; keep schemas precise and outputs compact. Reference docs: custom tools, tool concepts, API reference for `@tool`. [LangChain+2LangChain+2](https://python.langchain.com/docs/how_to/custom_tools/?utm_source=chatgpt.com)

**UPA**

- `yt_fetch_history(user_id:str, max_items:int=50) -> list[VideoMeta]`
- `yt_topics_infer(videos:list[VideoMeta]) -> {topic_keywords:[], cuisine_keywords:[], creators_top:[]}`
- `gmaps_likes_fetch(user_id:str) -> list[str]  # place_id[]`
- `item_item_cf(seed_place_ids:list[str]) -> {similar_place_ids:[], weights:[]}`
- `profile_merge(query_spec:QuerySpec|None, yt_topics:dict|None, likes_cf:dict|None) -> UserTasteProfile`

**RPA**

- `place_text_search(query:str, region_code:str|None, location_bias:str|None) -> list[Candidate]`
- `place_details_batch(place_ids:list[str]) -> list[PlaceDetail]`
- `entity_relation_extract(texts:list[str]) -> list[Triple]`
- `kg_upsert(triples:list[Triple]) -> {node_ids:[], edge_ids:[]}`
- `vector_embed_and_upsert(docs:list[RestaurantDoc]) -> {embedding_ids:[]}`
- `hybrid_collect_candidates(query_spec:QuerySpec, user_profile:UserTasteProfile|None) -> list[RestaurantDoc]`

**PFA**

- `scoring_pipeline(candidates:list[RestaurantDoc], query_spec:QuerySpec, user_profile:UserTasteProfile|None) -> RankedList`

**EEA**

- `fetch_place_photos(photo_reference:str, max_w:int=800, max_h:int=800) -> {photo_url:str}`
- `opening_hours_normalize(place_detail:PlaceDetail) -> {today_is_open:bool, closes_at:str|None, next_open:str|None}`
- `build_gmaps_deeplink(place_id:str, mode:str="driving", origin:str|None=None) -> str`

> For prebuilt agent usage & tool calling semantics see LangGraph agent docs and “Call tools” how-to. [LangChain AI+1](https://langchain-ai.github.io/langgraph/reference/agents/?utm_source=chatgpt.com)

------

## 5) Handoff strategy (supervisor routing)

We use **custom handoff tools** created via `create_handoff_tool` so the supervisor can transfer control to a target agent with a **small task message** and selected state fields. The library also supports **forward_message** to send a worker’s result straight to the user. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/supervisor/?utm_source=chatgpt.com)

**Default route**

1. `delegate_to_UIA` → writes `query_spec`
2. `delegate_to_UPA` (if authorized) → writes `user_profile`
3. `delegate_to_RPA` → writes `candidates`
4. `delegate_to_PFA` → writes `ranked`
5. `delegate_to_EEA` → writes `evidence`
6. `forward_to_user` → returns final cards

**Fallbacks**

- If `candidates < 3`: route back to `UIA` to relax constraints.
- If no OAuth: skip `UPA`, pass `user_profile=None` to `PFA`.
- If any tool 429/5xx: backoff-retry (≤2); then degrade (omit photos/hours, keep core fields).

> The tutorial describes handoffs and passing (full) history; we purposely pass **only a task + needed state slice** to keep tokens small. [LangChain AI](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)

------

## 6) Wiring (code outline)

- **Workers**: build with `create_react_agent(model, tools=[...], name="...")`. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/agents/?utm_source=chatgpt.com)
- **Supervisor**: `create_supervisor(agents=[...], tools=[handoffs..., forward_message], model=..., output_mode="last_message", add_handoff_messages=True)`. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/supervisor/?utm_source=chatgpt.com)
- **Compilation & memory**: compile the graph with a checkpointer and use `thread_id` per conversation (see Agents quickstart and migration notes). [LangChain AI+1](https://langchain-ai.github.io/langgraph/agents/agents/?utm_source=chatgpt.com)

------

## 7) Coding rules for tools & agents (for Codex)

1. **I/O minimalism**: Project any third-party responses to our schemas before returning.
2. **Deterministic return shape**: No free-text in tool outputs; use fields with explicit types.
3. **Retries & timeouts**: Each network tool must set `timeout` and implement 2-try exponential backoff (catch 429/5xx).
4. **Caching**: Cache `place_details_batch` & `fetch_place_photos` by `(place_id, fields/max_w/max_h)` for 24h.
5. **Rate limits**: Add per-tool token-bucket guards (configurable).
6. **Privacy**: No raw user identifiers in logs; if logging coordinates, quantize (e.g., 3-dp).
7. **Testing**: Each tool gets a unit test with recorded or mocked responses; workers get happy-path & failure-path tests.

References for tool creation and configuration: LangChain tool guides & API docs. [LangChain Docs+3LangChain+3LangChain+3](https://python.langchain.com/docs/how_to/custom_tools/?utm_source=chatgpt.com)

------

## 8) Prompts (snippets)

**Supervisor system prompt (essentials)**

- Roles & boundaries for each worker.
- “Assign exactly one worker at a time.”
- “Never call external APIs yourself; use handoff tools.”
- “Prefer compact JSON over prose; only include fields required by the next step.”

**Worker prompts**

- What to produce (the target schema).
- Which tools exist and when to call them.
- “Stop when the schema is satisfied; avoid redundant calls.”

> The supervisor prompt quality is critical for correct delegation. [Medium](https://medium.com/@khandelwal.akansha/understanding-the-langgraph-multi-agent-supervisor-00fa1be4341b?utm_source=chatgpt.com)

------

## 9) Acceptance checklist (Done = ✅)

- ✅ **UIA** returns valid `QuerySpec` for 10+ diverse utterances (missing fields → clarifying follow-ups).
- ✅ **UPA** produces a non-empty profile with OAuth and a session-only fallback without OAuth.
- ✅ **RPA** returns 20–50 candidates with <10 required fields each; retry logic triggers on low recall.
- ✅ **PFA** produces `RankedList` with `why[]` aligned to `QuerySpec`/`UserTasteProfile`.
- ✅ **EEA** enriches Top-K with ≤3 photos, normalized hours, and a working Maps deeplink.
- ✅ Supervisor route adheres to the default pipeline; fallbacks exercised in tests.
- ✅ Tool calls are visible in traces; each step is auditable.

------

## 10) Quickstart (scaffold expectations)

- Workers defined in `apps/whatseat/agents/*.py`, tools in `apps/whatseat/tools/*.py`, supervisor in `apps/whatseat/supervisor/`.
- Use **prebuilt agent** helpers for workers; confirm via reference docs & quickstart. [LangChain AI+1](https://langchain-ai.github.io/langgraph/reference/agents/?utm_source=chatgpt.com)
- Handoffs built with `create_handoff_tool` and (optionally) `create_forward_message_tool` from the supervisor library. [LangChain AI](https://langchain-ai.github.io/langgraph/reference/supervisor/?utm_source=chatgpt.com)

------

### Appendix: Why this stack?

- **Supervisor + handoff tools**: idiomatic LangGraph approach for hierarchical multi-agent orchestration. [LangChain AI](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/?utm_source=chatgpt.com)
- **Prebuilt ReAct workers**: faster, more reliable than legacy LangChain agents; the docs recommend LangGraph’s implementation for production. [LangChain+1](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html?utm_source=chatgpt.com)
- **`@tool` with schemas**: first-class support for tool I/O contracts; improves model grounding and robustness. [LangChain+1](https://python.langchain.com/docs/how_to/custom_tools/?utm_source=chatgpt.com)

> For additional background and examples, see the official supervisor reference and community write-ups. [LangChain AI+1](https://langchain-ai.github.io/langgraph/reference/supervisor/?utm_source=chatgpt.com)

------

**End of AGENTS.md**