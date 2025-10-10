# agents/youtube_agent.py
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from whats_eat.tools.youtube_data import yt_list_subscriptions, yt_list_liked_videos

def build_youtube_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[yt_list_subscriptions, yt_list_liked_videos],
        prompt=(
            "You are a YouTube data agent.\n"
            "- Retrieve subscriptions and liked videos.\n"
            "- Summarize user tastes briefly if asked."
        ),
        name="youtube_agent",
    )