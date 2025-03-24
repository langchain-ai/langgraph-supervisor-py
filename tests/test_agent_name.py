from langchain_core.messages import AIMessage, HumanMessage

from langgraph_supervisor.agent_name import (
    add_inline_agent_name,
    remove_inline_agent_name,
)


def test_add_inline_agent_name():
    # Test that non-AI messages are returned unchanged.
    human_message = HumanMessage(content="Hello")
    result = add_inline_agent_name(human_message)
    assert result == human_message

    # Test that AI messages with no name are returned unchanged.
    ai_message = AIMessage(content="Hello world")
    result = add_inline_agent_name(ai_message)
    assert result == ai_message

    # Test that AI messages get formatted with name and content tags.
    ai_message = AIMessage(content="Hello world", name="assistant")
    result = add_inline_agent_name(ai_message)
    assert result.content == "<name>assistant</name><content>Hello world</content>"
    assert result.name == "assistant"


def test_add_inline_agent_name_content_blocks():
    content_blocks = [
        {"type": "text", "text": "Hello world"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
    ]
    ai_message = AIMessage(content=content_blocks, name="assistant")
    result = add_inline_agent_name(ai_message)
    assert result.content == [
        {"type": "text", "text": "<name>assistant</name><content>Hello world</content>"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
    ]

    # Test that content blocks without text blocks are returned unchanged
    content_blocks = [
        {"type": "image", "image_url": "http://example.com/image.jpg"},
        {"type": "file", "file_url": "http://example.com/document.pdf"},
    ]
    expected_content_blocks = [
        {"type": "text", "text": "<name>assistant</name><content></content>"}
    ] + content_blocks
    ai_message = AIMessage(content=content_blocks, name="assistant")
    result = add_inline_agent_name(ai_message)

    # The message should be returned unchanged
    assert result.content == expected_content_blocks


def test_remove_inline_agent_name():
    # Test that non-AI messages are returned unchanged.
    human_message = HumanMessage(content="Hello")
    result = remove_inline_agent_name(human_message)
    assert result == human_message

    # Test that messages with empty content are returned unchanged.
    ai_message = AIMessage(content="", name="assistant")
    result = remove_inline_agent_name(ai_message)
    assert result == ai_message

    # Test that messages without name/content tags are returned unchanged.
    ai_message = AIMessage(content="Hello world", name="assistant")
    result = remove_inline_agent_name(ai_message)
    assert result == ai_message

    # Test that content is correctly extracted from tags.
    ai_message = AIMessage(
        content="<name>assistant</name><content>Hello world</content>", name="assistant"
    )
    result = remove_inline_agent_name(ai_message)
    assert result.content == "Hello world"
    assert result.name == "assistant"


def test_remove_inline_agent_name_content_blocks():
    content_blocks = [
        {"type": "text", "text": "<name>assistant</name><content>Hello world</content>"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
    ]
    ai_message = AIMessage(content=content_blocks, name="assistant")
    result = remove_inline_agent_name(ai_message)

    expected_content = [
        {"type": "text", "text": "Hello world"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
    ]
    assert result.content == expected_content
    assert result.name == "assistant"

    # Test that content blocks without text blocks are returned unchanged
    content_blocks = [
        {"type": "text", "text": "<name>assistant</name><content></content>"},
        {"type": "image", "image_url": "http://example.com/image.jpg"},
        {"type": "file", "file_url": "http://example.com/document.pdf"},
    ]
    expected_content_blocks = content_blocks[1:]
    ai_message = AIMessage(content=content_blocks, name="assistant")
    result = remove_inline_agent_name(ai_message)
    assert result.content == expected_content_blocks


def test_remove_inline_agent_name_multiline_content():
    multiline_content = """<name>assistant</name><content>This is
a multiline
message</content>"""
    ai_message = AIMessage(content=multiline_content, name="assistant")
    result = remove_inline_agent_name(ai_message)
    assert result.content == "This is\na multiline\nmessage"
