"""Tests for the supervisor module using functional API."""
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage


def create_mock_model() -> ChatOpenAI:
    """Create a mock ChatOpenAI model for testing."""
    model = MagicMock(spec=ChatOpenAI)
    model.invoke.return_value = AIMessage(content="Mocked joke response")
    model.bind_tools.return_value = model
    model.config_specs = []
    model.steps = []
    return model


def test_supervisor_functional_workflow() -> None:
    """Test supervisor workflow with a functional API agent."""
    model = create_mock_model()

    # Create a joke agent using functional API
    @task
    def generate_joke(messages: List[BaseMessage]) -> AIMessage:
        """Generate a joke using the model."""
        return model.invoke([SystemMessage(content="Write a short joke")] + messages)

    @entrypoint()
    def joke_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Joke agent entrypoint."""
        joke = generate_joke(state['messages']).result()
        messages = add_messages(state["messages"], [joke])
        return {"messages": messages}

    # Set agent name
    joke_agent.name = "joke_agent"

    # Create supervisor workflow
    workflow = create_supervisor(
        [joke_agent],
        model=model,
        prompt="You are a supervisor managing a joke expert."
    )

    # Compile and test
    app = workflow.compile()
    assert app is not None

    result = app.invoke({
        "messages": [
            HumanMessage(content="Tell me a joke!")
        ]
    })

    # Verify results
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert any("joke" in msg.content.lower() for msg in result["messages"]) 