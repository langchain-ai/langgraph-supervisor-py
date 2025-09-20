import re
from typing import Literal, Sequence, TypeGuard, cast

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    MessageLikeRepresentation,
    convert_to_messages,
)
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

NAME_PATTERN = re.compile(r"<name>(.*?)</name>", re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.DOTALL)

AgentNameMode = Literal["inline"]


def _is_content_blocks_content(content: list[dict | str] | str) -> TypeGuard[list[dict]]:
    return (
        isinstance(content, list)
        and len(content) > 0
        and isinstance(content[0], dict)
        and "type" in content[0]
    )


def add_inline_agent_name(message: BaseMessage) -> BaseMessage:
    """Add name and content XML tags to the message content.

    Examples:

        >>> add_inline_agent_name(AIMessage(content="Hello", name="assistant"))
        AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant")

        >>> add_inline_agent_name(AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant")
    """
    if not isinstance(message, AIMessage) or not message.name:
        return message

    formatted_message = message.model_copy()
    if _is_content_blocks_content(message.content):
        text_blocks = [block for block in message.content if block["type"] == "text"]
        non_text_blocks = [block for block in message.content if block["type"] != "text"]
        content = text_blocks[0]["text"] if text_blocks else ""
        formatted_content = f"<name>{message.name}</name><content>{content}</content>"
        formatted_message_content = [{"type": "text", "text": formatted_content}] + non_text_blocks
        formatted_message.content = formatted_message_content
    else:
        formatted_message.content = (
            f"<name>{message.name}</name><content>{formatted_message.content}</content>"
        )
    return formatted_message


def remove_inline_agent_name(message: BaseMessage) -> BaseMessage:
    """Remove explicit name and content XML tags from the AI message content.

    Examples:

        >>> remove_inline_agent_name(AIMessage(content="<name>assistant</name><content>Hello</content>", name="assistant"))
        AIMessage(content="Hello", name="assistant")

        >>> remove_inline_agent_name(AIMessage(content=[{"type": "text", "text": "<name>assistant</name><content>Hello</content>"}], name="assistant"))
        AIMessage(content=[{"type": "text", "text": "Hello"}], name="assistant")
    """
    if not isinstance(message, AIMessage) or not message.content:
        return message

    if is_content_blocks_content := _is_content_blocks_content(message.content):
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

    parsed_content = content_match.group(1)
    parsed_message = message.model_copy()
    if is_content_blocks_content:
        content_blocks = non_text_blocks
        if parsed_content:
            content_blocks = [{"type": "text", "text": parsed_content}] + content_blocks

        parsed_message.content = cast(list[str | dict], content_blocks)
    else:
        parsed_message.content = parsed_content
    return parsed_message


def with_agent_name(
    model: LanguageModelLike,
    agent_name_mode: AgentNameMode,
) -> LanguageModelLike:
    """Attach formatted agent names to the messages passed to and from a language model.

    This is useful for making a message history with multiple agents more coherent.

    NOTE: agent name is consumed from the message.name field.
        If you're using an agent built with create_react_agent, name is automatically set.
        If you're building a custom agent, make sure to set the name on the AI message returned by the LLM.

    Args:
        model: Language model to add agent name formatting to.
        agent_name_mode: Use to specify how to expose the agent name to the LLM.
            - "inline": Add the agent name directly into the content field of the AI message using XML-style tags.
                Example: "How can I help you" -> "<name>agent_name</name><content>How can I help you?</content>".
    """
    if agent_name_mode == "inline":
        process_input_message = add_inline_agent_name
        process_output_message = remove_inline_agent_name
    else:
        raise ValueError(
            f"Invalid agent name mode: {agent_name_mode}. Needs to be one of: {AgentNameMode.__args__}"
        )

    def process_input_messages(
        input: Sequence[MessageLikeRepresentation] | PromptValue,
    ) -> list[BaseMessage]:
        messages = convert_to_messages(input)
        return [process_input_message(message) for message in messages]

    chain = (
        process_input_messages
        | model
        | RunnableLambda(process_output_message, name="process_output_message")
    )

    return cast(LanguageModelLike, chain)
