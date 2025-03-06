from langchain_core.messages import AIMessage, HumanMessage

from langgraph_supervisor.utils import (
    process_input_message,
    process_output_message,
)


def test_process_input_message():
    # Test that non-AI messages are returned unchanged.
    human_message = HumanMessage(content="Hello")
    result = process_input_message(human_message)
    assert result == human_message

    # Test that AI messages with no name are returned unchanged.
    ai_message = AIMessage(content="Hello world")
    result = process_input_message(ai_message)
    assert result == ai_message

    # Test that AI messages get formatted with name and content tags.
    ai_message = AIMessage(content="Hello world", name="assistant")
    result = process_input_message(ai_message)
    assert result.content == "<name>assistant</name><content>Hello world</content>"
    assert result.name == "assistant"


def test_process_output_message():
    # Test that messages with empty content are returned unchanged.
    ai_message = AIMessage(content="", name="assistant")
    result = process_output_message(ai_message)
    assert result == ai_message

    # Test that messages without name/content tags are returned unchanged.
    ai_message = AIMessage(content="Hello world", name="assistant")
    result = process_output_message(ai_message)
    assert result == ai_message

    # Test that messages with mismatched name are returned unchanged.
    ai_message = AIMessage(
        content="<name>different_name</name><content>Hello world</content>", name="assistant"
    )
    result = process_output_message(ai_message)
    assert result == ai_message

    # Test that content is correctly extracted from tags.
    ai_message = AIMessage(
        content="<name>assistant</name><content>Hello world</content>", name="assistant"
    )
    result = process_output_message(ai_message)
    assert result.content == "Hello world"
    assert result.name == "assistant"


def test_process_output_message_content_blocks():
    content_blocks = [
        {"type": "text", "text": "<name>assistant</name><content>Hello world</content>"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
    ]
    ai_message = AIMessage(content=content_blocks, name="assistant")
    result = process_output_message(ai_message)

    expected_content = [
        {"type": "image", "image_url": "http://example.com/image.jpg"},
        {"type": "text", "text": "Hello world"},
    ]
    assert result.content == expected_content
    assert result.name == "assistant"


def test_process_output_message_multiline_content():
    multiline_content = """<name>assistant</name><content>This is
a multiline
message</content>"""
    ai_message = AIMessage(content=multiline_content, name="assistant")
    result = process_output_message(ai_message)
    assert result.content == "This is\na multiline\nmessage"
