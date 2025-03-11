import inspect
from typing import Any, Callable, Literal, Optional, Type, Union

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt.chat_agent_executor import (
    AgentState,
    Prompt,
    StateSchemaType,
    StructuredResponseSchema,
    create_react_agent,
)
from langgraph.pregel import Pregel
from langgraph.utils.runnable import RunnableCallable

from langgraph_supervisor.agent_name import AgentNameMode, with_agent_name
from langgraph_supervisor.handoff import (
    create_handoff_back_messages,
    create_handoff_tool,
)

OutputMode = Literal["full_history", "last_message"]
"""Mode for adding agent outputs to the message history in the multi-agent workflow

- `full_history`: add the entire agent message history
- `last_message`: add only the last message
"""


MODELS_NO_PARALLEL_TOOL_CALLS = {"o3-mini"}


def _supports_disable_parallel_tool_calls(model: LanguageModelLike) -> bool:
    if not isinstance(model, BaseChatModel):
        return False

    if hasattr(model, "model_name") and model.model_name in MODELS_NO_PARALLEL_TOOL_CALLS:
        return False

    if not hasattr(model, "bind_tools"):
        return False

    if "parallel_tool_calls" not in inspect.signature(model.bind_tools).parameters:
        return False

    return True


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


def create_supervisor(
    agents: list[Pregel],
    *,
    model: LanguageModelLike,
    tools: list[BaseTool | Callable] | None = None,
    prompt: Prompt | None = None,
    response_format: Optional[
        Union[StructuredResponseSchema, tuple[str, StructuredResponseSchema]]
    ] = None,
    state_schema: StateSchemaType = AgentState,
    config_schema: Type[Any] | None = None,
    output_mode: OutputMode = "last_message",
    add_handoff_back_messages: bool = True,
    supervisor_name: str = "supervisor",
    include_agent_name: AgentNameMode | None = None,
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
        response_format: An optional schema for the final supervisor output.

            If provided, output will be formatted to match the given schema and returned in the 'structured_response' state key.
            If not provided, `structured_response` will not be present in the output state.
            Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class,
                - or a Pydantic class.
                - a tuple (prompt, schema), where schema is one of the above.
                    The prompt will be used together with the model that is being used to generate the structured response.

            !!! Important
                `response_format` requires the model to support `.with_structured_output`

            !!! Note
                `response_format` requires `structured_response` key in your state schema.
                You can use the prebuilt `langgraph.prebuilt.chat_agent_executor.AgentStateWithStructuredResponse`.
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
        include_agent_name: Use to specify how to expose the agent name to the underlying supervisor LLM.

            - None: Relies on the LLM provider using the name attribute on the AI message. Currently, only OpenAI supports this.
            - "inline": Add the agent name directly into the content field of the AI message using XML-style tags.
                Example: "How can I help you" -> "<name>agent_name</name><content>How can I help you?</content>"
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

    if _supports_disable_parallel_tool_calls(model):
        model = model.bind_tools(all_tools, parallel_tool_calls=False)
    else:
        model = model.bind_tools(all_tools)

    if include_agent_name:
        model = with_agent_name(model, include_agent_name)

    supervisor_agent = create_react_agent(
        name=supervisor_name,
        model=model,
        tools=all_tools,
        prompt=prompt,
        state_schema=state_schema,
        response_format=response_format,
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
