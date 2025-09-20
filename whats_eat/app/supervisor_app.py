# app/supervisor_app.py
from langchain.chat_models import init_chat_model
from whats_eat.langgraph_supervisor import create_supervisor, create_forward_message_tool  # your package

from whats_eat.agents.places_agent import build_places_agent
from whats_eat.agents.youtube_agent import build_youtube_agent
from whats_eat.agents.recommender_agent import build_recommender_agent
from whats_eat.agents.summarizer_agent import build_summarizer_agent

def build_app():
    places = build_places_agent()
    youtube = build_youtube_agent()
    recommender = build_recommender_agent()
    summarizer = build_summarizer_agent()

    # Optional extra tool: forward a worker's exact wording to the user
    forward_tool = create_forward_message_tool()

    supervisor_prompt = (
        "You are the supervisor. Route requests to exactly ONE agent at a time.\n"
        "- Routing guide:\n"
        "  • Places lookups/photos → places_agent\n"
        "  • YouTube tastes/profile → youtube_agent\n"
        "  • Ranking/shortlisting → recommender_agent\n"
        "  • Condense long outputs → summarizer_agent\n"
        "- Do not solve tasks yourself. Use handoff tools to delegate.\n"
        "- When a worker finishes, respond to the user with the result."
    )

    workflow = create_supervisor(
        agents=[places, youtube, recommender, summarizer],
        model=init_chat_model("openai:gpt-4.1"),
        tools=[forward_tool],              # your handoff tools will be auto-added
        prompt=supervisor_prompt,
        add_handoff_back_messages=True,    # include “transfer back” messages
        output_mode="last_message",        # or "full_history" to include full traces
        include_agent_name="inline",       # robust name exposure for models
        parallel_tool_calls=False,         # 1-at-a-time handoffs (tutorial style)
    )
    return workflow.compile()
