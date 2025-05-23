import pytest
from unittest.mock import patch, MagicMock, ANY
from click.testing import CliRunner

from rich.console import Console
from rich.markdown import Markdown
from rich.json import JSON as RichJSON
from rich.pretty import Pretty
from langchain_core.tools import BaseTool

# Module to test
from hippycampus import cli


class TestRenderAgentResponse:
    @patch("rich.console.Console.print")
    def test_render_python_dict(self, mock_rich_print: MagicMock):
        console = Console()
        data = {"key": "value", "number": 123}
        cli.render_agent_response(
            data, console_instance=console
        )  # Pass console instance

        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Pretty)
        assert args[0].object == data

    @patch("rich.console.Console.print")
    def test_render_python_list(self, mock_rich_print: MagicMock):
        console = Console()
        data = [1, "string", {"key": "value"}]
        cli.render_agent_response(data, console_instance=console)

        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Pretty)
        assert args[0].object == data

    @patch("rich.console.Console.print")
    def test_render_valid_json_string(self, mock_rich_print: MagicMock):
        console = Console()
        json_string = '{"name": "Test User", "age": 30, "city": "New York"}'
        cli.render_agent_response(json_string, console_instance=console)

        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], RichJSON)
        # RichJSON parses the string internally. We can check if the parsed data matches.
        # This requires json.loads on the original string for comparison if checking RichJSON.from_data
        # If checking the `json` attribute of RichJSON, it's the original string.
        assert args[0].json == json_string

    @patch("rich.console.Console.print")
    def test_render_markdown_string(self, mock_rich_print: MagicMock):
        console = Console()
        markdown_string = "# Title\n\nThis is **bold** and *italic*."
        cli.render_agent_response(markdown_string, console_instance=console)

        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Markdown)
        assert args[0].markup == markdown_string

    @patch("rich.console.Console.print")
    def test_render_markdown_string_with_code_block(self, mock_rich_print: MagicMock):
        console = Console()
        markdown_string = '```json\n{"key": "value"}\n```'
        cli.render_agent_response(markdown_string, console_instance=console)

        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Markdown)
        assert args[0].markup == markdown_string

    @patch("rich.console.Console.print")
    def test_render_plain_text_string(self, mock_rich_print: MagicMock):
        console = Console()
        plain_string = (
            "This is a plain text string without special JSON or Markdown markers."
        )
        cli.render_agent_response(plain_string, console_instance=console)

        mock_rich_print.assert_called_once_with(plain_string)

    @patch("rich.console.Console.print")
    def test_render_empty_string(self, mock_rich_print: MagicMock):
        console = Console()
        empty_string = ""
        cli.render_agent_response(empty_string, console_instance=console)

        mock_rich_print.assert_called_once_with(empty_string)

    @patch("rich.console.Console.print")
    def test_render_invalid_json_string_looks_like_json(
        self, mock_rich_print: MagicMock
    ):
        console = Console()
        invalid_json_object_like = "{'key': 'value',}"
        cli.render_agent_response(invalid_json_object_like, console_instance=console)
        mock_rich_print.assert_called_with(
            invalid_json_object_like
        )  # Falls back to plain text

        invalid_json_array_like = "[1, 2, 'oops]"
        cli.render_agent_response(invalid_json_array_like, console_instance=console)
        mock_rich_print.assert_called_with(invalid_json_array_like)

    @patch("rich.console.Console.print")
    def test_render_string_that_is_just_a_number_or_bool(
        self, mock_rich_print: MagicMock
    ):
        console = Console()
        cli.render_agent_response("123.45", console_instance=console)
        mock_rich_print.assert_called_with("123.45")

        cli.render_agent_response("true", console_instance=console)
        mock_rich_print.assert_called_with("true")

    @patch("rich.console.Console.print")
    def test_render_integer_input(self, mock_rich_print: MagicMock):
        console = Console()
        data = 12345
        cli.render_agent_response(data, console_instance=console)
        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Pretty)
        assert args[0].object == data

    @patch("rich.console.Console.print")
    def test_render_float_input(self, mock_rich_print: MagicMock):
        console = Console()
        data = 123.45
        cli.render_agent_response(data, console_instance=console)
        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Pretty)
        assert args[0].object == data

    @patch("rich.console.Console.print")
    def test_render_boolean_input(self, mock_rich_print: MagicMock):
        console = Console()
        cli.render_agent_response(True, console_instance=console)
        # Check if called with Pretty(True)
        # The exact call might be Pretty(True) or just True depending on how console.print handles it
        # For robustness, check the type and value of the first arg's object if it's Pretty
        call_args = mock_rich_print.call_args_list
        assert (
            isinstance(call_args[0][0][0], Pretty) and call_args[0][0][0].object is True
        )

        cli.render_agent_response(False, console_instance=console)
        assert (
            isinstance(call_args[1][0][0], Pretty)
            and call_args[1][0][0].object is False
        )

    @patch("rich.console.Console.print")
    def test_render_none_input(self, mock_rich_print: MagicMock):
        console = Console()
        data = None
        cli.render_agent_response(data, console_instance=console)
        mock_rich_print.assert_called_once()
        args, _ = mock_rich_print.call_args
        assert isinstance(args[0], Pretty)
        assert args[0].object is None


class TestCliMain:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch(
        "hippycampus.cli.ChatGoogleGenerativeAI"
    )  # Mock the specific LLM class used by default
    @patch("langchain.hub.pull")
    @patch("langchain.agents.AgentExecutor.invoke")
    def test_main_default_options(
        self,
        mock_agent_invoke: MagicMock,
        mock_hub_pull: MagicMock,
        mock_chat_google: MagicMock,
        mock_load_tools: MagicMock,
        runner: CliRunner,
    ):
        # Setup mocks
        mock_load_tools.return_value = [
            MagicMock(spec=BaseTool, name="test_tool", description="A tool")
        ]
        mock_llm_instance = MagicMock()
        mock_chat_google.return_value = mock_llm_instance
        mock_prompt_instance = MagicMock()
        mock_hub_pull.return_value = mock_prompt_instance
        mock_agent_invoke.return_value = {"output": "Default agent response"}

        result = runner.invoke(cli.main)  # Run with default options

        assert result.exit_code == 0
        mock_load_tools.assert_called_with(
            "./test/xkcd.com/1.0.0/openapi2.yaml", auth=None
        )
        mock_chat_google.assert_called_with(
            model="gemini-1.0-pro", streaming=True, callbacks=ANY
        )
        mock_hub_pull.assert_called_with("hwchase17/structured-chat-agent")
        # Default query is "hi" if console is not interactive (like in tests)
        mock_agent_invoke.assert_called_with({"input": "hi"})

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch("langchain_openai.ChatOpenAI")  # Mock OpenAI if testing that path
    @patch("langchain.hub.pull")
    @patch("langchain.agents.AgentExecutor.invoke")
    def test_main_custom_options_openai(
        self,
        mock_agent_invoke: MagicMock,
        mock_hub_pull: MagicMock,
        mock_chat_openai: MagicMock,  # Changed from google
        mock_load_tools: MagicMock,
        runner: CliRunner,
    ):
        mock_load_tools.return_value = [
            MagicMock(spec=BaseTool, name="custom_tool", description="Custom")
        ]
        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance
        mock_prompt_instance = MagicMock()
        mock_hub_pull.return_value = mock_prompt_instance
        mock_agent_invoke.return_value = {"output": "Custom agent response"}

        cli_args = [
            "--openapi-spec",
            "http://custom.com/spec.yaml",
            "--model-name",
            "gpt-3.5-turbo",
            "--prompt-hub-repo",
            "custom/prompt",
            "--query",
            "My custom query",
            "--auth-token",
            "secret_token",
            "--verbose",
        ]
        result = runner.invoke(cli.main, cli_args)

        assert result.exit_code == 0, result.output  # Print output if error
        mock_load_tools.assert_called_with(
            "http://custom.com/spec.yaml", auth="secret_token"
        )
        mock_chat_openai.assert_called_with(
            model="gpt-3.5-turbo", streaming=True, callbacks=ANY
        )
        mock_hub_pull.assert_called_with("custom/prompt")
        mock_agent_invoke.assert_called_with({"input": "My custom query"})
        assert "Total tools loaded: 1" in result.output
        assert "Generated Tools:" in result.output  # Due to verbose
        assert "Custom agent response" in result.output  # Rendered output

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch("hippycampus.cli.ChatGoogleGenerativeAI")
    @patch("langchain.hub.pull")
    @patch("langchain.agents.AgentExecutor.invoke")
    @patch(
        "rich.console.Console.input", return_value="interactive query"
    )  # Mock console input
    def test_main_interactive_query(
        self,
        mock_console_input: MagicMock,
        mock_agent_invoke: MagicMock,
        mock_hub_pull: MagicMock,
        mock_chat_google: MagicMock,
        mock_load_tools: MagicMock,
        runner: CliRunner,
    ):
        mock_load_tools.return_value = [MagicMock(spec=BaseTool)]
        mock_agent_invoke.return_value = {"output": "Interactive response"}

        # Run without --query to trigger interactive input
        result = runner.invoke(cli.main, ["--openapi-spec", "dummy.yaml"])

        assert result.exit_code == 0
        mock_console_input.assert_called_once()
        mock_agent_invoke.assert_called_with({"input": "interactive query"})
        assert "Interactive response" in result.output

    @patch(
        "hippycampus.cli.load_tools_from_openapi",
        side_effect=Exception("Spec load failed!"),
    )
    def test_main_spec_load_failure(
        self, mock_load_tools: MagicMock, runner: CliRunner
    ):
        result = runner.invoke(cli.main, ["--openapi-spec", "bad_spec.yaml"])
        assert (
            result.exit_code == 0
        )  # main() currently catches and prints, then exits gracefully
        assert (
            "Error loading tools from bad_spec.yaml: Spec load failed!" in result.output
        )
        assert "No tools were successfully loaded. Exiting." in result.output

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch(
        "hippycampus.cli.ChatGoogleGenerativeAI",
        side_effect=Exception("LLM init failed!"),
    )
    def test_main_llm_init_failure(
        self, mock_chat_google: MagicMock, mock_load_tools: MagicMock, runner: CliRunner
    ):
        mock_load_tools.return_value = [MagicMock(spec=BaseTool)]  # Ensure tools load
        result = runner.invoke(cli.main, ["--openapi-spec", "dummy.yaml"])
        assert result.exit_code == 0  # main() catches and prints
        assert "Error initializing LLM model" in result.output
        assert "LLM init failed!" in result.output

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch("hippycampus.cli.ChatGoogleGenerativeAI")
    @patch("langchain.hub.pull", side_effect=Exception("Hub pull failed!"))
    def test_main_hub_pull_failure(
        self,
        mock_hub_pull: MagicMock,
        mock_chat_google: MagicMock,
        mock_load_tools: MagicMock,
        runner: CliRunner,
    ):
        mock_load_tools.return_value = [MagicMock(spec=BaseTool)]
        result = runner.invoke(cli.main, ["--openapi-spec", "dummy.yaml"])
        assert result.exit_code == 0
        assert "Error pulling prompt from Langchain Hub" in result.output
        assert "Hub pull failed!" in result.output

    @patch("hippycampus.cli.load_tools_from_openapi")
    @patch("hippycampus.cli.ChatGoogleGenerativeAI")
    @patch("langchain.hub.pull")
    @patch(
        "langchain.agents.AgentExecutor.invoke",
        side_effect=Exception("Agent execution error!"),
    )
    def test_main_agent_execution_failure(
        self,
        mock_agent_invoke: MagicMock,
        mock_hub_pull: MagicMock,
        mock_chat_google: MagicMock,
        mock_load_tools: MagicMock,
        runner: CliRunner,
    ):
        mock_load_tools.return_value = [MagicMock(spec=BaseTool)]
        result = runner.invoke(
            cli.main, ["--openapi-spec", "dummy.yaml", "--query", "test query"]
        )
        assert result.exit_code == 0
        assert "Error during agent execution: Agent execution error!" in result.output


if __name__ == "__main__":
    pytest.main()
