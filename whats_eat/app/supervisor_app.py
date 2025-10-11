# app/supervisor_app.py
from langchain.chat_models import init_chat_model
from whats_eat.langgraph_supervisor import create_supervisor, create_forward_message_tool  # your package

from whats_eat.agents.places_agent import build_places_agent
from whats_eat.agents.youtube_agent import build_youtube_agent
from whats_eat.agents.recommender_agent import build_recommender_agent
from whats_eat.agents.summarizer_agent import build_summarizer_agent
from whats_eat.agents.route_agent import build_route_agent

def build_app():
    places = build_places_agent()
    youtube = build_youtube_agent()
    recommender = build_recommender_agent()
    summarizer = build_summarizer_agent()
    route = build_route_agent()

    # Optional extra tool: forward a worker's exact wording to the user
    forward_tool = create_forward_message_tool()

    supervisor_prompt = (
        "You are the supervisor. Route each user request to exactly ONE agent at a time.\n"
        "- Available agents:\n"
        "  • places_agent – retrieves and analyzes information about places, restaurants, or local venues; GEOCODES any addresses (user or restaurant) to lat/long.\n"
        "  • youtube_agent – builds a user taste profile by analyzing YouTube activity, watched channels, or favorite creators.\n"
        "  • recommender_agent – ranks, filters, or selects items (e.g., recommends top places based on taste, location, or user preferences).\n"
        "  • summarizer_agent – combines and refines results from other agents to generate the final, human-readable response.\n"
        "  • route_agent – computes routes and generates interactive map views GIVEN coordinates (lat/long); does not perform geocoding.\n"
        "- Routing guide:\n"
        "  • Location/place search or address→coordinates (user or restaurant) → places_agent\n"
        "  • YouTube history, channels, or interest-based profiling → youtube_agent\n"
        "  • Ranking, comparison, or shortlisting → recommender_agent\n"
        "  • Routing / map visualization when coordinates are known → route_agent\n"
        "  • When all required information has been gathered, produce the final answer → summarizer_agent\n"
        "- Do not solve tasks yourself. Use handoff tools to delegate.\n"
        "- Always delegate to exactly ONE agent per turn.\n"
        "- If the request is unclear or missing critical information (e.g., starting address or selected restaurant), ask ONE short clarifying question before delegating.\n"
        "- Multi-step handling (typical flows):\n"
        "  • Place search only: places_agent → summarizer_agent\n"
        "  • Recommendations: places_agent → recommender_agent → summarizer_agent\n"
        "  • Route/map: places_agent (geocode addresses to lat/long) → route_agent (compute route & map) → summarizer_agent\n"
        "- Pass only coordinates (lat/long) to route_agent; do not pass raw addresses.\n"
        "- The summarizer_agent always produces the final output shown to the user."


)

    workflow = create_supervisor(
        agents=[places, youtube, recommender, summarizer, route],
        model=init_chat_model("openai:gpt-4.1"),
        tools=[forward_tool],              # your handoff tools will be auto-added
        prompt=supervisor_prompt,
        add_handoff_back_messages=True,    # include “transfer back” messages
        output_mode="last_message",        # or "full_history" to include full traces
        include_agent_name="inline",       # robust name exposure for models
        parallel_tool_calls=False,         # 1-at-a-time handoffs (tutorial style)
    )
    return workflow.compile()

