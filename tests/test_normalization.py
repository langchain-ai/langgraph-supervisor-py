"""Tests for normalization functionality."""

import unittest
from langchain_core.messages import AIMessage, BaseMessage

# Assuming _normalize_agent_name is available and works correctly
from langgraph_supervisor.handoff import _normalize_agent_name

# Import the preferred function
from langgraph_supervisor.supervisor import _normalize_tool_calls_in_message


class TestNormalization(unittest.TestCase):
    def test_normalize_agent_name(self):
        """Test that agent names are properly normalized."""
        # Test diacritics removal using unicodedata method
        self.assertEqual(_normalize_agent_name("adhésif_expert"), "adhesif_expert")
        self.assertEqual(_normalize_agent_name("café"), "cafe")
        self.assertEqual(_normalize_agent_name("crème brûlée"), "creme_brulee")
        self.assertEqual(_normalize_agent_name("françois"), "francois")
        self.assertEqual(_normalize_agent_name("tschüss"), "tschuss")
        self.assertEqual(_normalize_agent_name("Čeština"), "cestina")

        # Test spaces and stripping
        self.assertEqual(_normalize_agent_name("adhésif expert "), "adhesif_expert")
        self.assertEqual(_normalize_agent_name(" multiple spaces "), "multiple_spaces")

        # Test mixed case
        self.assertEqual(_normalize_agent_name("Adhésif_Expert"), "adhesif_expert")

        # Test already normalized
        self.assertEqual(_normalize_agent_name("already_normal"), "already_normal")

        # Test underscore preservation
        self.assertEqual(
            _normalize_agent_name("name_with_underscores"), "name_with_underscores"
        )

        # Test non-alphanumeric (assuming _normalize_agent_name doesn't strip them, only whitespace/diacritics)
        # Adjust based on actual _normalize_agent_name implementation if it handles more chars
        self.assertEqual(
            _normalize_agent_name("münich-guide"), "munich-guide"
        )  # Hyphen preserved
        self.assertEqual(
            _normalize_agent_name("test&fun"), "test&fun"
        )  # Ampersand preserved

    def test_normalize_tool_calls_in_message(self):
        """Test that tool calls with diacritics are properly normalized in AIMessage."""
        # Test case 1: Single tool call needing normalization
        original_name_1 = "transfer_to_adhésif_expert"
        expected_name_1 = "transfer_to_adhesif_expert"
        message1 = AIMessage(
            content="Need adhesive help",
            tool_calls=[
                {
                    "name": original_name_1,
                    "args": {"query": "best glue"},
                    "id": "call_1",
                }
            ],
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 25,
                "total_tokens": 35,
            },
            response_metadata={"model": "gpt-4"},
            id="msg_1",
            name="supervisor",
        )

        normalized_message1 = _normalize_tool_calls_in_message(message1)

        # Check name normalization
        self.assertEqual(len(normalized_message1.tool_calls), 1)
        self.assertEqual(normalized_message1.tool_calls[0]["name"], expected_name_1)
        # Check other fields are preserved
        self.assertEqual(
            normalized_message1.tool_calls[0]["args"], {"query": "best glue"}
        )
        self.assertEqual(normalized_message1.tool_calls[0]["id"], "call_1")
        self.assertEqual(normalized_message1.content, "Need adhesive help")
        self.assertEqual(normalized_message1.id, "msg_1")
        self.assertEqual(normalized_message1.name, "supervisor")
        # Crucially check metadata
        self.assertEqual(
            normalized_message1.usage_metadata,
            {"input_tokens": 10, "output_tokens": 25, "total_tokens": 35},
        )
        self.assertEqual(normalized_message1.response_metadata, {"model": "gpt-4"})
        # Check it's a new object because modification happened
        self.assertIsNot(normalized_message1, message1)

        # Test case 2: Tool call not needing normalization (not starting with transfer_to_)
        message2 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "some_other_tool",
                    "args": {},
                    "id": "call_2",
                }
            ],
        )
        normalized_message2 = _normalize_tool_calls_in_message(message2)
        self.assertEqual(normalized_message2.tool_calls[0]["name"], "some_other_tool")
        # Check it's the *same* object because no modification happened
        self.assertIs(normalized_message2, message2)

        # Test case 3: Multiple tool calls, some needing normalization
        original_name_3a = "transfer_to_crème_brûlée_expert"
        expected_name_3a = "transfer_to_creme_brulee_expert"
        original_name_3b = "transfer_to_sommelier"  # Already normalized
        message3 = AIMessage(
            content="Dinner plans",
            tool_calls=[
                {
                    "name": original_name_3a,
                    "args": {"recipe": "classic"},
                    "id": "call_3a",
                },
                {
                    "name": "search_web",
                    "args": {"query": "restaurants"},
                    "id": "call_3b",
                },
                {
                    "name": original_name_3b,
                    "args": {"region": "Bordeaux"},
                    "id": "call_3c",
                },
            ],
        )
        normalized_message3 = _normalize_tool_calls_in_message(message3)
        self.assertEqual(len(normalized_message3.tool_calls), 3)
        self.assertEqual(normalized_message3.tool_calls[0]["name"], expected_name_3a)
        self.assertEqual(normalized_message3.tool_calls[1]["name"], "search_web")
        self.assertEqual(normalized_message3.tool_calls[2]["name"], original_name_3b)
        # Check it's a new object because modification happened
        self.assertIsNot(normalized_message3, message3)

        # Test case 4: Message with no tool calls
        message4 = AIMessage(content="Hello world", id="msg_4")
        normalized_message4 = _normalize_tool_calls_in_message(message4)
        # Check it's the *same* object
        self.assertIs(normalized_message4, message4)
        self.assertEqual(normalized_message4.content, "Hello world")

        # Test case 5: Non-AIMessage input
        message5 = BaseMessage(content="I am not an AI", type="human")
        normalized_message5 = _normalize_tool_calls_in_message(message5)
        # Check it's the *same* object
        self.assertIs(normalized_message5, message5)

        # Test case 6: Tool call already normalized but matches prefix
        message6 = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "transfer_to_adhesive_expert",
                    "args": {},
                    "id": "call_6",
                }
            ],
            id="msg_6",
        )
        normalized_message6 = _normalize_tool_calls_in_message(message6)
        self.assertEqual(
            normalized_message6.tool_calls[0]["name"], "transfer_to_adhesive_expert"
        )
        # Check it's the *same* object because no modification happened
        self.assertIs(normalized_message6, message6)


# Add this to run tests if the file is executed directly
if __name__ == "__main__":
    unittest.main()
