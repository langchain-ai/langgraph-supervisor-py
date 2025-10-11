from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from whats_eat.tools.route_map import route_geocode, route_build_map_html


def build_route_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[route_build_map_html],
        prompt=(
            "You are a Route & Maps assistant.\n"
            "- If the user provides addresses by postcode, use place_geocode tool from places_agent to obtain coordinates.\n"
            "- Generate an interactive map by calling route_build_map_html.\n"
            "- Prefer DRIVING unless user specifies WALKING, BICYCLING or TRANSIT.\n"
            "- Return the HTML string when asked to show or share the map.\n"
            "- Do NOT fabricate keys or URLs; only use tool outputs."
        ),
        name="route_agent",
    )