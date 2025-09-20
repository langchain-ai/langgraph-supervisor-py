# agents/summarizer_agent.py
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

def build_summarizer_agent():
    return create_react_agent(
        model=init_chat_model("openai:gpt-4.1"),
        tools=[],  # can add a tool later (e.g., long-text summarizer backend)
        prompt="You are a concise summarizer. Output crisp summaries only.",
        name="summarizer_agent",
    )