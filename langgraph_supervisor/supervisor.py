import inspect
from typing import Any, Callable, Literal, Type

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchemaType,
    create_react_agent,
)
from langgraph.pregel import Pregel
from langgraph.utils.runnable import RunnableCallable

from langgraph_supervisor.handoff import (
    create_handoff_back_messages,
    create_handoff_tool,
)

OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""


def _make_call_agent(
    agent: Pregel,
    output_mode: OutputMode,
    add_handoff_back_messages: bool,
    supervisor_name: str,
) -> Callable[[dict], dict] | RunnableCallable:
    if output_mode not in OutputMode.__args__:
        raise ValueError(
            f"Invalid agent output mode: {output_mode}. Needs to be one of {OutputMode.__args__}"
        )

    def _process_output(output: dict) -> dict:
        messages = output["messages"]
        if output_mode == "full_history":
            pass
        elif output_mode == "last_message":
            messages = messages[-1:]
        else:
            raise ValueError(
                f"Invalid agent output mode: {output_mode}. "
                f"Needs to be one of {OutputMode.__args__}"
            )

        if add_handoff_back_messages:
            messages.extend(create_handoff_back_messages(agent.name, supervisor_name))

        return {
            **output,
            "messages": messages,
        }

    def call_agent(state: dict) -> dict:
        output = agent.invoke(state)
        return _process_output(output)

    async def acall_agent(state: dict) -> dict:
        output = await agent.ainvoke(state)
        return _process_output(output)

    return RunnableCallable(call_agent, acall_agent)


def _attach_message_processors(
    model: LanguageModelLike,
    process_input_message: Callable[[BaseMessage], BaseMessage] | None = None,
    process_output_message: Callable[[AIMessage], AIMessage] | None = None,
) -> LanguageModelLike:
    if process_input_message:

        def process_input_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
            return [process_input_message(message) for message in messages]

        model = process_input_messages | model

    if process_output_message:
        model = model | process_output_message

    return model


def create_supervisor(
    agents: list[Pregel],
    *,
    model: LanguageModelLike,
    tools: list[BaseTool | Callable] | None = None,
    prompt: Prompt | None = None,
    state_schema: StateSchemaType = AgentState,
    config_schema: Type[Any] | None = None,
    output_mode: OutputMode = "last_message",
    add_handoff_back_messages: bool = True,
    supervisor_name: str = "supervisor",
    process_input_message: Callable[[BaseMessage], BaseMessage] | None = None,
    process_output_message: Callable[[AIMessage], AIMessage] | None = None,
) -> StateGraph:
    """Create a multi-agent supervisor.

    Args:
        agents: List of agents to manage
        model: Language model to use for the supervisor
        tools: Tools to use for the supervisor
        prompt: Optional prompt to use for the supervisor. Can be one of:
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        state_schema: State schema to use for the supervisor graph.
        config_schema: An optional schema for configuration.
            Use this to expose configurable parameters via supervisor.config_specs.
        output_mode: Mode for adding managed agents' outputs to the message history in the multi-agent workflow.
            Can be one of:
            - `full_history`: add the entire agent message history
            - `last_message`: add only the last message (default)
        add_handoff_back_messages: Whether to add a pair of (AIMessage, ToolMessage) to the message history
            when returning control to the supervisor to indicate that a handoff has occurred.
        supervisor_name: Name of the supervisor node.
        process_input_message: An optional function that takes a message and reformats it before passing it to the LLM.
            This is useful for injecting additional information like the name of the agent into the message content.
        process_output_message: An optional function that takes AI message from the LLM and reformats it back for the end user.
    """
    agent_names = set()
    for agent in agents:
        if agent.name is None or agent.name == "LangGraph":
            raise ValueError(
                "Please specify a name when you create your agent, either via `create_react_agent(..., name=agent_name)` "
                "or via `graph.compile(name=name)`."
            )

        if agent.name in agent_names:
            raise ValueError(
                f"Agent with name '{agent.name}' already exists. Agent names must be unique."
            )

        agent_names.add(agent.name)

    handoff_tools = [create_handoff_tool(agent_name=agent.name) for agent in agents]
    all_tools = (tools or []) + handoff_tools

    if (
        hasattr(model, "bind_tools")
        and "parallel_tool_calls" in inspect.signature(model.bind_tools).parameters
    ):
        model = model.bind_tools(all_tools, parallel_tool_calls=False)
    else:
        model = model.bind_tools(all_tools)

    model = _attach_message_processors(
        model,
        process_input_message,
        process_output_message,
    )

    supervisor_agent = create_react_agent(
        name=supervisor_name,
        model=model,
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
    )

    builder = StateGraph(state_schema, config_schema=config_schema)
    builder.add_node(supervisor_agent, destinations=tuple(agent_names) + (END,))
    builder.add_edge(START, supervisor_agent.name)
    for agent in agents:
        builder.add_node(
            agent.name,
            _make_call_agent(
                agent,
                output_mode,
                add_handoff_back_messages,
                supervisor_name,
            ),
        )
        builder.add_edge(agent.name, supervisor_agent.name)

    return builder
