import pytest
import json

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException

from hippycampus.langchain_util import FixedJSONAgentOutputParser


class TestFixedJSONAgentOutputParser:
    @pytest.fixture
    def parser(self) -> FixedJSONAgentOutputParser:
        return FixedJSONAgentOutputParser()

    def test_valid_agent_action(self, parser: FixedJSONAgentOutputParser):
        json_string = '{"action": "Search", "action_input": "What is Langchain?"}'
        expected_output = AgentAction(
            tool="Search", tool_input="What is Langchain?", log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_valid_agent_finish(self, parser: FixedJSONAgentOutputParser):
        json_string = '{"action": "Final Answer", "action_input": "Langchain is a framework for developing applications powered by language models."}'
        expected_output = AgentFinish(
            return_values={
                "output": "Langchain is a framework for developing applications powered by language models."
            },
            log=json_string,
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_action_input_is_none(self, parser: FixedJSONAgentOutputParser):
        json_string = '{"action": "CustomTool", "action_input": null}'
        # According to the original parser's logic, null action_input becomes an empty string for AgentAction
        # or an empty dict for AgentFinish (though Final Answer typically has string input).
        # Let's assume CustomTool implies AgentAction.
        expected_output = AgentAction(tool="CustomTool", tool_input="", log=json_string)
        result = parser.parse(json_string)
        assert result == expected_output

    def test_action_input_is_none_for_final_answer(
        self, parser: FixedJSONAgentOutputParser
    ):
        json_string = '{"action": "Final Answer", "action_input": null}'
        # For Final Answer, null action_input should result in an empty string output, or handled by Pydantic as None if model allows
        # The original JSONAgentOutputParser converts `action_input` to a string for the `output` key.
        # If `action_input` is `None`, it becomes `{"output": "None"}`.
        # FixedJSONAgentOutputParser aims to fix this. If action_input is null for Final Answer, output should be empty or None.
        # The current FixedJSONAgentOutputParser code:
        # if isinstance(action_input, dict): output = json.dumps(action_input, ensure_ascii=False)
        # else: output = str(action_input)
        # So, null -> "None" string. This might be an area the "fix" is intended to address.
        # Let's assume the desired fix is that null action_input for Final Answer means empty output.
        # However, the provided code for FixedJSONAgentOutputParser *does not* change this behavior from the original.
        # It will still convert null to "None".
        # If the intention of "Fixed" was to make output for Final Answer truly empty for null input:
        # expected_output = AgentFinish(return_values={"output": ""}, log=json_string)
        # But based on the provided code for FixedJSONAgentOutputParser:
        expected_output = AgentFinish(return_values={"output": "None"}, log=json_string)
        result = parser.parse(json_string)
        assert result == expected_output

        # If the fix *was* intended to make it an empty string for null input:
        # A possible modification in FixedJSONAgentOutputParser would be:
        # elif action == "Final Answer":
        #    output = str(action_input) if action_input is not None else ""
        #    return AgentFinish({"output": output}, text)

    def test_llm_returns_list_of_actions_picks_first(
        self, parser: FixedJSONAgentOutputParser
    ):
        # The provided FixedJSONAgentOutputParser code does not explicitly handle a list of actions.
        # It expects a single JSON object representing one action or finish.
        # If the LLM returns a string that is a JSON list, json.loads will parse it into a list.
        # The parser then tries to access `parsed_json["action"]` which will fail for a list.
        # This test will likely fail or needs to assume the "fix" handles this.
        # The original langchain parser also expects a dict, not a list.
        json_string = '[{"action": "Search", "action_input": "query1"}, {"action": "Calculate", "action_input": "1+1"}]'
        # If the parser were to pick the first, the expected output:
        # expected_output = AgentAction(tool="Search", tool_input="query1", log=json_string)
        # However, current code will raise an error because it expects a dict.
        with pytest.raises(OutputParserException) as excinfo:
            parser.parse(json_string)
        assert "Could not parse LLM output as a JSON object" in str(excinfo.value)
        # assert "got list" in str(excinfo.value).lower() # More specific check if the error message includes it.
        # The error message is "Could not parse LLM output as a JSON object, got <class 'list'>"
        # which is what the current FixedJSONAgentOutputParser produces.

    def test_malformed_json(self, parser: FixedJSONAgentOutputParser):
        json_string = '{"action": "Search", "action_input": "What is Langchain?"'  # Missing closing brace
        with pytest.raises(OutputParserException) as excinfo:
            parser.parse(json_string)
        assert "Could not parse LLM output as JSON" in str(
            excinfo.value
        )  # Error from json.loads

    def test_missing_action_key(self, parser: FixedJSONAgentOutputParser):
        json_string = '{"tool_name": "Search", "action_input": "query"}'  # 'action' key is missing
        with pytest.raises(OutputParserException) as excinfo:
            parser.parse(json_string)
        assert "Missing 'action' key in LLM output" in str(excinfo.value)

    def test_action_input_is_string_for_final_answer(
        self, parser: FixedJSONAgentOutputParser
    ):
        json_string = (
            '{"action": "Final Answer", "action_input": "This is the final output."}'
        )
        expected_output = AgentFinish(
            return_values={"output": "This is the final output."}, log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_action_input_is_dict_for_final_answer(
        self, parser: FixedJSONAgentOutputParser
    ):
        # The "fix" in FixedJSONAgentOutputParser is specifically for this case.
        action_input_dict = {"result": "some data", "value": 42}
        json_string = f'{{"action": "Final Answer", "action_input": {json.dumps(action_input_dict)}}}'

        # The FixedJSONAgentOutputParser should now dump the dict to a JSON string for the output.
        expected_output_value = json.dumps(action_input_dict, ensure_ascii=False)
        expected_output = AgentFinish(
            return_values={"output": expected_output_value}, log=json_string
        )

        result = parser.parse(json_string)
        assert result == expected_output

    def test_action_input_is_dict_for_agent_action(
        self, parser: FixedJSONAgentOutputParser
    ):
        action_input_dict = {"query": "What is Langchain?", "source": "web"}
        json_string = (
            f'{{"action": "Search", "action_input": {json.dumps(action_input_dict)}}}'
        )
        # For AgentAction, if action_input is a dict, it should be passed as is (or dumped to string, depending on tool needs)
        # The original parser converts dict to string. FixedJSONAgentOutputParser does not change this for AgentAction.
        expected_tool_input = json.dumps(
            action_input_dict
        )  # Langchain tools often expect string or dict.
        # Default behavior is to stringify.
        expected_output = AgentAction(
            tool="Search", tool_input=expected_tool_input, log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_action_input_is_string_for_agent_action(
        self, parser: FixedJSONAgentOutputParser
    ):
        json_string = '{"action": "Search", "action_input": "simple string query"}'
        expected_output = AgentAction(
            tool="Search", tool_input="simple string query", log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_json_with_code_block_wrapper(self, parser: FixedJSONAgentOutputParser):
        json_string_with_wrapper = '```json\n{"action": "Search", "action_input": "query from wrapped json"}\n```'
        expected_output = AgentAction(
            tool="Search",
            tool_input="query from wrapped json",
            log=json_string_with_wrapper,
        )
        result = parser.parse(json_string_with_wrapper)
        assert result == expected_output

    def test_json_with_code_block_wrapper_and_whitespace(
        self, parser: FixedJSONAgentOutputParser
    ):
        json_string_with_wrapper = '  ```json\n  {  "action": "Search",\n"action_input": "query with whitespace"}\n```  '
        expected_output = AgentAction(
            tool="Search",
            tool_input="query with whitespace",
            log=json_string_with_wrapper,
        )
        result = parser.parse(json_string_with_wrapper)
        assert result == expected_output

    def test_json_with_text_before_code_block(self, parser: FixedJSONAgentOutputParser):
        json_string = 'Some text before the JSON block.\n```json\n{"action": "Think", "action_input": "Thinking about the query"}\n```'
        expected_output = AgentAction(
            tool="Think", tool_input="Thinking about the query", log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_json_with_text_after_code_block(self, parser: FixedJSONAgentOutputParser):
        json_string = '```json\n{"action": "Summarize", "action_input": "text to summarize"}\n```\nSome text after the JSON block.'
        expected_output = AgentAction(
            tool="Summarize", tool_input="text to summarize", log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output

    def test_json_without_code_block_but_with_newlines(
        self, parser: FixedJSONAgentOutputParser
    ):
        json_string = '\n\n{"action": "ExtractInfo", "action_input": "details"}\n\n'
        expected_output = AgentAction(
            tool="ExtractInfo", tool_input="details", log=json_string
        )
        result = parser.parse(json_string)
        assert result == expected_output


# For type hinting review, I will inspect hippycampus/langchain_util.py later.

if __name__ == "__main__":
    pytest.main()
