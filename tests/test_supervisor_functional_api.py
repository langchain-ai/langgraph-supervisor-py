"""Tests for the supervisor module using functional API."""

from typing import Any, Dict, List

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.func import entrypoint, task
from langgraph.graph import add_messages

from langgraph_supervisor import create_supervisor


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, *args, **kwargs) -> "FakeModel":
        """Do nothing for now."""
        return self


def test_supervisor_functional_workflow() -> None:
    """Test supervisor workflow with a functional API agent."""
    model = FakeModel(
        messages=iter([AIMessage(content="Mocked response")]),
    )

    # Create a joke agent using functional API
    @task
    def generate_joke(messages: List[BaseMessage]) -> AIMessage:
        """Generate a joke using the model."""
        return model.invoke([SystemMessage(content="Write a short joke")] + messages)

    @entrypoint()
    def joke_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        """Joke agent entrypoint."""
        joke = generate_joke(state["messages"]).result()
        messages = add_messages(state["messages"], [joke])
        return {"messages": messages}

    # Set agent name
    joke_agent.name = "joke_agent"

    # Create supervisor workflow
    workflow = create_supervisor(
        [joke_agent], model=model, prompt="You are a supervisor managing a joke expert."
    )

    # Compile and test
    app = workflow.compile()
    assert app is not None

    result = app.invoke({"messages": [HumanMessage(content="Tell me a joke!")]})

    # Verify results
    assert "messages" in result
    assert len(result["messages"]) > 0
    assert any("joke" in msg.content.lower() for msg in result["messages"])
