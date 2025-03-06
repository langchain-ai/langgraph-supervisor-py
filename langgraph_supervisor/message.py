import re
from typing import Literal

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, BaseMessage

NAME_PATTERN = re.compile(r"<name>(.*?)</name>", re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.DOTALL)

MessageFormat = Literal["xml_tags"]


def _is_content_blocks_content(content: list[dict] | str) -> bool:
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and "type" in content[0]
    )


def add_xml_tags_to_message_content(message: BaseMessage) -> BaseMessage:
    """Add name and content XML tags to the message content.

    This is useful for injecting additional information like the name of the agent into the message content.

    Examples:

        >>> add_xml_tags_to_message_content(AIMessage(content="Hello", name="assistant"))
        AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant")

        >>> add_xml_tags_to_message_content(AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant")
    """
    if not isinstance(message, AIMessage) or not message.name:
        return message

    formatted_message = message.model_copy()
    if _is_content_blocks_content(formatted_message.content):
        text_blocks = [block for block in message.content if block["type"] == "text"]
        non_text_blocks = [block for block in message.content if block["type"] != "text"]
        content = text_blocks[0]["text"] if text_blocks else ""
        formatted_content = f"<name>{message.name}</name><content>{content}</content>"
        formatted_message.content = non_text_blocks + [{"type": "text", "text": formatted_content}]
    else:
        formatted_message.content = (
            f"<name>{message.name}</name><content>{formatted_message.content}</content>"
        )
    return formatted_message


def remove_xml_tags_from_message_content(message: BaseMessage) -> BaseMessage:
    """Removing explicit name and content XML tags from the AI message content.

    Examples:

        >>> remove_xml_tags_from_message_content(AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant"))
        AIMessage(content="Hello", name="assistant")

        >>> remove_xml_tags_from_message_content(AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant")
    """
    if not isinstance(message, AIMessage) or not message.name:
        return message

    is_content_blocks_content = _is_content_blocks_content(message.content)
    if is_content_blocks_content:
        text_blocks = [block for block in message.content if block["type"] == "text"]
        if not text_blocks:
            return message

        non_text_blocks = [block for block in message.content if block["type"] != "text"]
        content = text_blocks[0]["text"]
    else:
        content = message.content

    name_match: re.Match | None = NAME_PATTERN.search(content)
    content_match: re.Match | None = CONTENT_PATTERN.search(content)
    if not name_match or not content_match:
        return message

    if name_match.group(1) != message.name:
        return message

    parsed_content = content_match.group(1)
    parsed_message = message.model_copy()
    if is_content_blocks_content:
        content_blocks = non_text_blocks
        if parsed_content:
            content_blocks.append({"type": "text", "text": parsed_content})

        parsed_message.content = content_blocks
    else:
        parsed_message.content = parsed_content
    return parsed_message


def with_message_format(
    model: LanguageModelLike,
    message_format: MessageFormat,
) -> LanguageModelLike:
    """Attach message processors to a language model.

    Args:
        model: Language model to attach message processors to.
        message_processor: The type of message processor to attach.
            - "xml_tags": Add name and content XML tags to the message content before passing to the LLM
                and remove them after receiving the response.
    """
    if message_format == "xml_tags":
        process_input_message = add_xml_tags_to_message_content
        process_output_message = remove_xml_tags_from_message_content

    else:
        raise ValueError(
            f"Invalid message format: {message_format}. Needs to be one of: {MessageFormat.__args__}"
        )

    def process_input_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        return [process_input_message(message) for message in messages]

    model = process_input_messages | model | process_output_message
    return model
