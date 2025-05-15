import re
import uuid
from typing import TypeGuard, cast

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from typing_extensions import Annotated

WHITESPACE_RE = re.compile(r"\s+")
METADATA_KEY_HANDOFF_DESTINATION = "__handoff_destination"
METADATA_KEY_IS_HANDOFF_BACK = "__is_handoff_back"


def _normalize_agent_name(agent_name: str) -> str:
    """Normalize an agent name to be used inside the tool name."""
    return WHITESPACE_RE.sub("_", agent_name.strip()).lower()


def _has_multiple_content_blocks(content: str | list[str | dict]) -> TypeGuard[list[dict]]:
    """Check if content contains multiple content blocks."""
    return isinstance(content, list) and len(content) > 1 and isinstance(content[0], dict)


def _remove_non_handoff_tool_calls(
    last_ai_message: AIMessage, handoff_tool_call_id: str
) -> AIMessage:
    """Remove tool calls that are not meant for the agent."""
    # if the supervisor is calling multiple agents/tools in parallel,
    # we need to remove tool calls that are not meant for this agent
    # to ensure that the resulting message history is valid
    content = last_ai_message.content
    if _has_multiple_content_blocks(content):
        content = [
            content_block
            for content_block in content
            if (content_block["type"] == "tool_use" and content_block["id"] == handoff_tool_call_id)
            or content_block["type"] != "tool_use"
        ]

    last_ai_message = AIMessage(
        content=content,
        tool_calls=[
            tool_call
            for tool_call in last_ai_message.tool_calls
            if tool_call["id"] == handoff_tool_call_id
        ],
        name=last_ai_message.name,
        id=str(uuid.uuid4()),
    )
    return last_ai_message


def create_handoff_tool(
    *,
    agent_name: str,
    name: str | None = None,
    description: str | None = None,
    add_handoff_messages: bool = True,
) -> BaseTool:
    """Create a tool that can handoff control to the requested agent.

    Args:
        agent_name: The name of the agent to handoff control to, i.e.
            the name of the agent node in the multi-agent graph.
            Agent names should be simple, clear and unique, preferably in snake_case,
            although you are only limited to the names accepted by LangGraph
            nodes as well as the tool names accepted by LLM providers
            (the tool name will look like this: `transfer_to_<agent_name>`).
        name: Optional name of the tool to use for the handoff.
            If not provided, the tool name will be `transfer_to_<agent_name>`.
        description: Optional description for the handoff tool.
            If not provided, the description will be `Ask agent <agent_name> for help`.
        add_handoff_messages: Whether to add handoff messages to the message history.
            If False, the handoff messages will be omitted from the message history.
    """
    if name is None:
        name = f"transfer_to_{_normalize_agent_name(agent_name)}"

    if description is None:
        description = f"Ask agent '{agent_name}' for help"

    @tool(name, description=description)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_HANDOFF_DESTINATION: agent_name},
        )
        last_ai_message = cast(AIMessage, state["messages"][-1])
        # Handle parallel handoffs
        if len(last_ai_message.tool_calls) > 1:
            handoff_messages = state["messages"][:-1]
            if add_handoff_messages:
                handoff_messages.extend(
                    (
                        _remove_non_handoff_tool_calls(last_ai_message, tool_call_id),
                        tool_message,
                    )
                )
            return Command(
                graph=Command.PARENT,
                # NOTE: we are using Send here to allow the ToolNode in langgraph.prebuilt
                # to handle parallel handoffs by combining all Send commands into a single command
                goto=[Send(agent_name, {**state, "messages": handoff_messages})],
            )
        # Handle single handoff
        else:
            if add_handoff_messages:
                handoff_messages = state["messages"] + [tool_message]
            else:
                handoff_messages = state["messages"][:-1]
            return Command(
                goto=agent_name,
                graph=Command.PARENT,
                update={**state, "messages": handoff_messages},
            )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
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
            response_metadata={METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
        ToolMessage(
            content=f"Successfully transferred back to {supervisor_name}",
            name=tool_name,
            tool_call_id=tool_call_id,
            response_metadata={METADATA_KEY_IS_HANDOFF_BACK: True},
        ),
    )


def create_forward_message_tool(supervisor_name: str = "supervisor") -> BaseTool:
    """Create a tool the supervisor can use to forward a worker message by name.

    This helps avoid information loss any time the supervisor rewrites a worker query
    to the user and also can save some tokens.

    Args:
        supervisor_name: The name of the supervisor node (used for namespacing the tool).

    Returns:
        BaseTool: The 'forward_message' tool.
    """
    tool_name = "forward_message"
    desc = (
        "Forwards the latest message from the specified agent to the user"
        " without any changes. Use this to preserve information fidelity, avoid"
        " misinterpretation of questions or responses, and save time."
    )

    @tool(tool_name, description=desc)
    def forward_message(
        from_agent: str,
        state: Annotated[dict, InjectedState],
    ) -> str | Command:
        target_message = next(
            (
                m
                for m in reversed(state["messages"])
                if isinstance(m, AIMessage)
                and (m.name or "").lower() == from_agent.lower()
                and not m.response_metadata.get(METADATA_KEY_IS_HANDOFF_BACK)
            ),
            None,
        )
        if not target_message:
            found_names = set(
                m.name for m in state["messages"] if isinstance(m, AIMessage) and m.name
            )
            return (
                f"Could not find message from source agent {from_agent}. Found names: {found_names}"
            )
        updates = [
            AIMessage(
                content=target_message.content,
                name=supervisor_name,
                id=str(uuid.uuid4()),
            ),
        ]

        return Command(
            graph=Command.PARENT,
            # NOTE: this does nothing.
            goto="__end__",
            # we also propagate the update to make sure the handoff messages are applied
            # to the parent graph's state
            update={**state, "messages": updates},
        )

    return forward_message
