"""Tests for the supervisor module."""

from typing import Callable, Optional

import pytest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

from langgraph_supervisor import create_supervisor
from langgraph_supervisor.agent_name import AgentNameMode, with_agent_name
from langgraph_supervisor.handoff import create_forward_message_tool


class FakeChatModel(BaseChatModel):
    idx: int = 0
    responses: list[BaseMessage]

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        generation = ChatGeneration(message=self.responses[self.idx])
        self.idx += 1
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: list[BaseTool]) -> "FakeChatModel":
        tool_dicts = [
            {
                "name": tool.name,
            }
            for tool in tools
        ]
        return self.bind(tools=tool_dicts)


supervisor_messages = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "transfer_to_research_expert",
                "args": {},
                "id": "call_gyQSgJQm5jJtPcF5ITe8GGGF",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "transfer_to_math_expert",
                "args": {},
                "id": "call_zCExWE54g4B4oFZcwBh3Wumg",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="The combined headcount of the FAANG companies in 2024 is 1,977,586 employees.",
    ),
]

research_agent_messages = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "FAANG headcount 2024"},
                "id": "call_4sLYp7usFcIZBFcNsOGQiFzV",
                "type": "tool_call",
            },
        ],
    ),
    AIMessage(
        content="The headcount for the FAANG companies in 2024 is as follows:\n\n1. **Facebook (Meta)**: 67,317 employees\n2. **Amazon**: 1,551,000 employees\n3. **Apple**: 164,000 employees\n4. **Netflix**: 14,000 employees\n5. **Google (Alphabet)**: 181,269 employees\n\nTo find the combined headcount, simply add these numbers together.",
    ),
]

math_agent_messages = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add",
                "args": {"a": 67317, "b": 1551000},
                "id": "call_BRvA6oAlgMA1whIkAn9gE3AS",
                "type": "tool_call",
            },
            {
                "name": "add",
                "args": {"a": 164000, "b": 14000},
                "id": "call_OLVb4v0pNDlsBsKBwDK4wb1W",
                "type": "tool_call",
            },
            {
                "name": "add",
                "args": {"a": 181269, "b": 0},
                "id": "call_5VEHaInDusJ9MU3i3tVJN6Hr",
                "type": "tool_call",
            },
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add",
                "args": {"a": 1618317, "b": 178000},
                "id": "call_FdfUz8Gm3S5OQaVq2oQpMxeN",
                "type": "tool_call",
            },
            {
                "name": "add",
                "args": {"a": 181269, "b": 0},
                "id": "call_j5nna1KwGiI60wnVHM2319r6",
                "type": "tool_call",
            },
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "add",
                "args": {"a": 1796317, "b": 181269},
                "id": "call_4fNHtFvfOvsaSPb8YK1qNAiR",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="The combined headcount of the FAANG companies in 2024 is 1,977,586 employees.",
    ),
]


@pytest.mark.parametrize(
    "include_agent_name,include_individual_agent_name",
    [
        (None, None),
        (None, "inline"),
        ("inline", None),
        ("inline", "inline"),
    ],
)
def test_supervisor_basic_workflow(
    include_agent_name: AgentNameMode | None,
    include_individual_agent_name: AgentNameMode | None,
) -> None:
    """Test basic supervisor workflow with two agents."""

    # output_mode = "last_message"
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        return (
            "Here are the headcounts for each of the FAANG companies in 2024:\n"
            "1. **Facebook (Meta)**: 67,317 employees.\n"
            "2. **Apple**: 164,000 employees.\n"
            "3. **Amazon**: 1,551,000 employees.\n"
            "4. **Netflix**: 14,000 employees.\n"
            "5. **Google (Alphabet)**: 181,269 employees."
        )

    math_model = FakeChatModel(responses=math_agent_messages)
    if include_individual_agent_name:
        math_model = with_agent_name(math_model.bind_tools([add]), include_individual_agent_name)

    math_agent = create_react_agent(
        model=math_model,
        tools=[add],
        name="math_expert",
    )

    research_model = FakeChatModel(responses=research_agent_messages)
    if include_individual_agent_name:
        research_model = with_agent_name(
            research_model.bind_tools([web_search]), include_individual_agent_name
        )

    research_agent = create_react_agent(
        model=research_model,
        tools=[web_search],
        name="research_expert",
    )

    workflow = create_supervisor(
        [math_agent, research_agent],
        model=FakeChatModel(responses=supervisor_messages),
        include_agent_name=include_agent_name,
    )

    app = workflow.compile()
    assert app is not None

    result = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="what's the combined headcount of the FAANG companies in 2024?"
                )
            ]
        }
    )

    assert len(result["messages"]) == 12
    # first supervisor handoff
    assert result["messages"][1] == supervisor_messages[0]
    # last research agent message
    assert result["messages"][3] == research_agent_messages[-1]
    # next supervisor handoff
    assert result["messages"][6] == supervisor_messages[1]
    # last math agent message
    assert result["messages"][8] == math_agent_messages[-1]
    # final supervisor message
    assert result["messages"][11] == supervisor_messages[-1]

    # output_mode = "full_history"
    math_agent = create_react_agent(
        model=FakeChatModel(responses=math_agent_messages),
        tools=[add],
        name="math_expert",
    )

    research_agent = create_react_agent(
        model=FakeChatModel(responses=research_agent_messages),
        tools=[web_search],
        name="research_expert",
    )

    workflow_full_history = create_supervisor(
        [math_agent, research_agent],
        model=FakeChatModel(responses=supervisor_messages),
        output_mode="full_history",
    )
    app_full_history = workflow_full_history.compile()
    result_full_history = app_full_history.invoke(
        {
            "messages": [
                HumanMessage(
                    content="what's the combined headcount of the FAANG companies in 2024?"
                )
            ]
        }
    )

    assert len(result_full_history["messages"]) == 23
    # first supervisor handoff
    assert result_full_history["messages"][1] == supervisor_messages[0]
    # all research agent AI messages
    assert result_full_history["messages"][3] == research_agent_messages[0]
    assert result_full_history["messages"][5] == research_agent_messages[1]
    # next supervisor handoff
    assert result_full_history["messages"][8] == supervisor_messages[1]
    # all math agent AI messages
    assert result_full_history["messages"][10] == math_agent_messages[0]
    assert result_full_history["messages"][14] == math_agent_messages[1]
    assert result_full_history["messages"][17] == math_agent_messages[2]
    # final supervisor message
    assert result_full_history["messages"][-1] == supervisor_messages[-1]


class FakeChatModelWithAssertion(FakeChatModel):
    assertion: Callable[[list[BaseMessage]], None]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        self.assertion(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def get_tool_calls(msg):
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls is None:
        return None
    return [
        {"name": tc["name"], "args": tc["args"]} for tc in tool_calls if tc["type"] == "tool_call"
    ]


def as_dict(msg):
    return {
        "name": msg.name,
        "content": msg.content,
        "tool_calls": get_tool_calls(msg),
        "type": msg.type,
    }


class Expectations:
    def __init__(self, expected: list[dict]):
        self.expected = expected.copy()

    def __call__(self, messages: list[BaseMessage]):
        expected = self.expected.pop(0)
        received = [as_dict(m) for m in messages]
        assert expected == received


def test_worker_hide_handoffs():
    """Test that the supervisor forwards a message to a specific agent and receives the correct response."""

    @tool
    def echo_tool(text: str) -> str:
        """Echo the input text."""
        return text

    expectations = [
        [
            {
                "name": None,
                "content": "Scooby-dooby-doo",
                "tool_calls": None,
                "type": "human",
            }
        ],
        [
            {
                "name": None,
                "content": "Scooby-dooby-doo",
                "tool_calls": None,
                "type": "human",
            },
            {
                "name": "echo_agent",
                "content": "Echo 1!",
                "tool_calls": [],
                "type": "ai",
            },
            {"name": "supervisor", "content": "boo", "tool_calls": [], "type": "ai"},
            {
                "name": None,
                "content": "Huh take two?",
                "tool_calls": None,
                "type": "human",
            },
        ],
    ]

    echo_model = FakeChatModelWithAssertion(
        responses=[
            AIMessage(content="Echo 1!"),
            AIMessage(content="Echo 2!"),
        ],
        assertion=Expectations(expectations),
    )
    echo_agent = create_react_agent(
        model=echo_model.bind_tools([echo_tool]),
        tools=[echo_tool],
        name="echo_agent",
    )

    supervisor_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "transfer_to_echo_agent",
                    "args": {},
                    "id": "call_gyQSgJQm5jJtPcF5ITe8GGGF",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="boo",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "transfer_to_echo_agent",
                    "args": {},
                    "id": "call_gyQSgJQm5jJtPcF5ITe8GGGG",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="END",
        ),
    ]

    workflow = create_supervisor(
        [echo_agent],
        model=FakeChatModel(responses=supervisor_messages),
        add_handoff_messages=False,
    )
    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Scooby-dooby-doo")]})
    app.invoke({"messages": result["messages"] + [HumanMessage(content="Huh take two?")]})


def test_supervisor_message_forwarding():
    """Test that the supervisor forwards a message to a specific agent and receives the correct response."""

    @tool
    def echo_tool(text: str) -> str:
        """Echo the input text."""
        return text

    # Agent that simply echoes the message
    echo_model = FakeChatModel(
        responses=[
            AIMessage(content="Echo: test forwarding!"),
        ]
    )
    echo_agent = create_react_agent(
        model=echo_model.bind_tools([echo_tool]),
        tools=[echo_tool],
        name="echo_agent",
    )

    supervisor_messages = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "transfer_to_echo_agent",
                    "args": {},
                    "id": "call_gyQSgJQm5jJtPcF5ITe8GGGF",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "forward_message",
                    "args": {"from_agent": "echo_agent"},
                    "id": "abcd123",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    forwarding = create_forward_message_tool("supervisor")
    workflow = create_supervisor(
        [echo_agent],
        model=FakeChatModel(responses=supervisor_messages),
        tools=[forwarding],
    )
    app = workflow.compile()

    result = app.invoke({"messages": [HumanMessage(content="Scooby-dooby-doo")]})

    def get_tool_calls(msg):
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None:
            return None
        return [
            {"name": tc["name"], "args": tc["args"]}
            for tc in tool_calls
            if tc["type"] == "tool_call"
        ]

    received = [
        {
            "name": msg.name,
            "content": msg.content,
            "tool_calls": get_tool_calls(msg),
            "type": msg.type,
        }
        for msg in result["messages"]
    ]

    expected = [
        {
            "name": None,
            "content": "Scooby-dooby-doo",
            "tool_calls": None,
            "type": "human",
        },
        {
            "name": "supervisor",
            "content": "",
            "tool_calls": [
                {
                    "name": "transfer_to_echo_agent",
                    "args": {},
                }
            ],
            "type": "ai",
        },
        {
            "name": "transfer_to_echo_agent",
            "content": "Successfully transferred to echo_agent",
            "tool_calls": None,
            "type": "tool",
        },
        {
            "name": "echo_agent",
            "content": "Echo: test forwarding!",
            "tool_calls": [],
            "type": "ai",
        },
        {
            "name": "echo_agent",
            "content": "Transferring back to supervisor",
            "tool_calls": [
                {
                    "name": "transfer_back_to_supervisor",
                    "args": {},
                }
            ],
            "type": "ai",
        },
        {
            "name": "transfer_back_to_supervisor",
            "content": "Successfully transferred back to supervisor",
            "tool_calls": None,
            "type": "tool",
        },
        {
            "name": "supervisor",
            "content": "Echo: test forwarding!",
            "tool_calls": [],
            "type": "ai",
        },
    ]
    assert received == expected
