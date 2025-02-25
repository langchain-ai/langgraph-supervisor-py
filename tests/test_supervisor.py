"""Tests for the supervisor module."""

from typing import Self

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langgraph_supervisor import create_supervisor


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, *args, **kwargs) -> Self:
        """Do nothing for now."""
        return self


def test_supervisor_basic_workflow() -> None:
    """Test basic supervisor workflow with a math agent."""
    model = FakeModel(
        messages=iter([AIMessage(content="Mocked response")]),
    )

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    math_agent = create_react_agent(
        model=model,
        tools=[add],
        name="math_expert",
        prompt="You are a math expert. Always use one tool at a time.",
    )

    workflow = create_supervisor(
        [math_agent], model=model, prompt="You are a supervisor managing a math expert."
    )

    app = workflow.compile()
    assert app is not None

    result = app.invoke({"messages": [HumanMessage(content="what's 2 + 2?")]})

    assert "messages" in result
    assert len(result["messages"]) > 0
