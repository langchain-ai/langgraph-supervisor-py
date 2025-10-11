# agents/youtube_agent.py
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from whats_eat.tools.youtube_data import yt_list_subscriptions, yt_list_liked_videos

def build_youtube_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[yt_list_subscriptions, yt_list_liked_videos],
        prompt=(
            "You are an execution agent (youtube_agent) in the “What’s Eat” system.\n"
            "Dispatched by the supervisor to handle YouTube-related data retrieval tasks.\n"
            "You do not respond to users directly and must not ask follow-up questions.\n"
            "Your responsibility is to use the YouTube Data API to fetch and organize relevant user data,\n"
            "then return structured results to the supervisor for summarization by summarizer_agent.\n"
            "- Retrieve user subscriptions and liked videos when requested.\n"
            "- If asked to analyze preferences, briefly summarize user tastes based on video categories, channels, and engagement patterns.\n"
            "- Fetch only the data available from the API; do NOT fabricate or infer any fields.\n"
            "- Return structured JSON results only (no natural-language sentences), including fields such as video title, channel, category, view count, and published date.\n"
            "- The response must be a single JSON object containing an 'items' array of videos or channels.\n"
            "- All results are passed to summarizer_agent for aggregation and presentation to the user.\n"
            "- Respond in the same language as the user input when summarizing tastes."
        ),
        name="youtube_agent",
    )