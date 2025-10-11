"""LangGraph agent that distils YouTube behaviour into a vector taste profile (OpenAI embeddings)."""

from __future__ import annotations

from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

from whats_eat.tools.user_profile import (
    embed_user_preferences,
    yt_list_liked_videos,
    yt_list_subscriptions,
)

def build_user_profile_agent():
    model = init_chat_model("openai:gpt-4.1", temperature=0.2, top_p=1)
    return create_react_agent(
        model=model,
        tools=[yt_list_subscriptions, yt_list_liked_videos, embed_user_preferences],
        prompt=(
            "You are the user_profile_agent within the “What’s Eat” supervisor workflow.\n"
            "You never respond directly to end users. Only act on instructions routed by the supervisor and use "
            "the tools bound in this context. Operate in a single turn: gather signals, reason, return JSON.\n"
            "\n"
            "Available data sources:\n"
            "- yt_list_liked_videos → recent likes (recency-weighted preference hints)\n"
            "- yt_list_subscriptions → long-term channel interests (stability signals)\n"
            "Both tools may return an 'error' field; if present, note it and continue with whatever data is usable. "
            "Never make additional assumptions or talk to other agents.\n"
            "\n"
            "Extraction guidance:\n"
            "- Identify cuisine/dish/style/region/diet keywords (e.g., ramen, mala, vegan, hawker, tapas, Japanese...).\n"
            "- Derive compact signal phrases from titles/channels/metadata, de-duplicate, keep top ≤20 by confidence.\n"
            "- Produce a user taste embedding by CALLING the tool `embed_user_preferences` with a concise summary string "
            "(≤200 tokens). Do not handcraft numeric vectors yourself.\n"
            "- If data is insufficient, leave keywords empty and call `embed_user_preferences` with the text "
            "\"insufficient data\"; propagate the tool's outcome and state the limitation in notes.\n"
            "- Map tool result fields as follows: tool.model → embedding_model; tool.dim → embedding_dim; "
            "tool.embedding → embedding. If tool.error is not null, set embedding to [] and explain in notes.\n"
            "\n"
            "Output contract (must be valid JSON, no markdown/text around it):\n"
            "{\n"
            '  \"keywords\": [ordered list of ≤12 tokens],\n'
            '  \"attributes\": {\n'
            '      \"price_band\": \"budget|mid|upscale\" (optional),\n'
            '      \"diet\": [...],\n'
            '      \"style\": [...],\n'
            '      \"region\": [...]\n'
            "  },\n"
            "  \"embedding_model\": \"text-embedding-3-small|text-embedding-3-large\",\n"
            "  \"embedding_dim\": <integer>,\n"
            "  \"embedding\": [list of floats],\n"
            "  \"notes\": \"1-2 sentences referencing concrete tool outputs; mention any degradation\"\n"
            "}\n"
            "Rules: Use only tool outputs as evidence. Do not include extra keys. Keep notes concise. "
            "Do not disclose raw URLs or IDs; citing channel/video names is acceptable."
        ),
        name="user_profile_agent",
    )
