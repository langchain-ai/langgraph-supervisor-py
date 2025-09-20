# agents/places_agent.py
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from tools.google_places import places_text_search, places_fetch_photos

def build_places_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[places_text_search, places_fetch_photos],
        prompt=(
            "You are a Places expert.\n"
            "- Use text search to find the best-matching place.\n"
            "- If the user asks for pictures, fetch photos.\n"
            "- Do NOT make up fields; only return what you fetched."
        ),
        name="places_agent",
    )

