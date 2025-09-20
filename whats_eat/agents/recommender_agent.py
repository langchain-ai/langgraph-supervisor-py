
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from whats_eat.tools.ranking import rank_restaurants

def build_recommender_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[rank_restaurants],
        prompt=(
            "You are a restaurant recommender.\n"
            "- Given candidates and user context, rank options and return top N.\n"
            "- Keep the output concise and actionable."
        ),
        name="recommender_agent",
    )