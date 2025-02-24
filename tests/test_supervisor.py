"""Tests for the supervisor module."""
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool


def create_mock_model() -> ChatOpenAI:
    """Create a mock ChatOpenAI model for testing."""
    model = MagicMock(spec=ChatOpenAI)
    model.invoke.return_value = AIMessage(content="Mocked response")
    model.bind_tools.return_value = model
    model.config_specs = []
    model.steps = []
    return model


def test_supervisor_basic_workflow() -> None:
    """Test basic supervisor workflow with a math agent."""
    model = create_mock_model()

    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    math_agent = create_react_agent(
        model=model,
        tools=[add],
        name="math_expert",
        prompt="You are a math expert. Always use one tool at a time."
    )

    workflow = create_supervisor(
        [math_agent],
        model=model,
        prompt="You are a supervisor managing a math expert."
    )

    app = workflow.compile()
    assert app is not None

    result = app.invoke({
        "messages": [
            HumanMessage(content="what's 2 + 2?")
        ]
    })

    assert "messages" in result
    assert len(result["messages"]) > 0
