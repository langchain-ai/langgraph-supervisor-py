import re

from langchain_core.messages import AIMessage, BaseMessage

NAME_PATTERN = re.compile(r"<name>(.*?)</name>", re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.DOTALL)


def process_input_message(message: BaseMessage) -> BaseMessage:
    """Process message content by adding explicit name and content tags.

    This is useful for injecting additional information like the name of the agent into the message content.
    """
    if not isinstance(message, AIMessage):
        return message

    if not message.name:
        return message

    formatted_message = message.model_copy()
    formatted_message.content = f"<name>{message.name}</name><content>{message.content}</content>"
    return formatted_message


def process_output_message(message: AIMessage) -> AIMessage:
    """Process message content in AI message by removing explicit name and content tags."""
    if not message.content:
        return message

    is_content_blocks_content = False
    if (
        isinstance(message.content, list)
        and len(message.content) > 0
        and isinstance(message.content[0], dict)
        and "type" in message.content[0]
    ):
        text_blocks = [block for block in message.content if block["type"] == "text"]
        non_text_blocks = [block for block in message.content if block["type"] != "text"]
        content = text_blocks[0]["text"]
        is_content_blocks_content = True
    else:
        content = message.content

    name_match: re.Match | None = NAME_PATTERN.search(content)
    content_match: re.Match | None = CONTENT_PATTERN.search(content)
    if not name_match or not content_match:
        return message

    if name_match.group(1) != message.name:
        return message

    parsed_message = message.model_copy()
    if is_content_blocks_content:
        parsed_message.content = non_text_blocks + [
            {"type": "text", "text": content_match.group(1)}
        ]
    else:
        parsed_message.content = content_match.group(1)
    return parsed_message
