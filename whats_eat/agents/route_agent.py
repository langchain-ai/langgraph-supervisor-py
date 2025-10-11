from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from whats_eat.tools.route_map import route_build_map_html


def build_route_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[route_build_map_html],
        prompt=(
            "You are a Route & Maps execution agent (route_agent) in the “What’s Eat” system.\n"
            "- Dispatched by the supervisor to calculate and display the route and distance between the user's location and a selected restaurant.\n"
            "- This agent is only called when the user clicks on a specific restaurant card to view details or request route information.\n"
            "- All inputs provided to this agent are geographic coordinates (latitude and longitude) for both the user and the restaurant.\n"
            "- Use these coordinates to calculate the route and estimate the travel distance and time between the two locations.\n"
            "- Generate an interactive map that visualizes the route and distance, and return it as HTML content when requested.\n"
            "- Default to DRIVING mode unless the user explicitly specifies WALKING, BICYCLING, or TRANSIT.\n"
            "- This agent does not perform any address or postal code parsing; all coordinates are preprocessed by places_agent.\n"
            "- Do NOT fabricate keys, URLs, or coordinates; only use actual tool outputs.\n"
            "- Respond in the same language as the user input (e.g., if the user speaks Chinese, respond in Chinese)."


        ),
        name="route_agent",
    )