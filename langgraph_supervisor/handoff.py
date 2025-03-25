import re
import uuid
from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name."""
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def _remove_non_handoff_tool_calls(
    messages: list[BaseMessage], agent_name: str
) -> list[BaseMessage]:
    """Remove tool calls that are not meant for the agent."""
    last_ai_message = cast(AIMessage, messages[-1])
    # if the supervisor is calling multiple agents/tools in parallel,
    # we need to remove tool calls that are not meant for this agent
    # to ensure that the resulting message history is valid
    if len(last_ai_message.tool_calls) > 1:
        content = last_ai_message.content
        if isinstance(content, list) and len(content) > 1 and isinstance(content[0], dict):
            content = [
                content_block
                for content_block in content
                if (content_block["type"] == "tool_use" and agent_name in content_block["name"])
                or content_block["type"] != "tool_use"
            ]

        last_ai_message = AIMessage(
            content=content,
            tool_calls=[
                tool_call
                for tool_call in last_ai_message.tool_calls
                if agent_name in tool_call["name"]
            ],
            name=last_ai_message.name,
            id=str(uuid.uuid4()),
        )

    return messages[:-1] + [last_ai_message]


def create_handoff_tool(*, agent_name: str) -> BaseTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).
    """
    tool_name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        handoff_messages = (
            _remove_non_handoff_tool_calls(state["messages"], agent_name) + [tool_message]
        )
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": handoff_messages},
        )

    return handoff_to_agent


def create_handoff_back_messages(
    agent_name: str, supervisor_name: str
) -> tuple[AIMessage, ToolMessage]:
    """Create a pair of (AIMessage, ToolMessage) to add to the message history when returning control to the supervisor."""
    tool_call_id = str(uuid.uuid4())
    tool_name = f"transfer_back_to_{_normalize_agent_name(supervisor_name)}"
    tool_calls = [ToolCall(name=tool_name, args={}, id=tool_call_id)]
    return (
        AIMessage(
            content=f"Transferring back to {supervisor_name}",
            tool_calls=tool_calls,
            name=agent_name,
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
        ),
    )
