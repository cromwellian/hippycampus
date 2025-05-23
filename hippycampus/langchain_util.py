# -*- coding: utf-8 -*-
"""
Langchain utility functions and custom classes.

This module provides helper functions and custom classes to integrate with or extend
Langchain functionalities, particularly for creating and parsing agent responses.
"""
from typing import Sequence, Union, List, Callable, Any, Dict 
import json 

from langchain.agents import AgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str # type: ignore
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool, ToolsRenderer, render_text_description_and_args


def fixed_create_structured_chat_agent(
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        tools_renderer: ToolsRenderer = render_text_description_and_args, # type: ignore
        *,
        stop_sequence: Union[bool, List[str]] = True,
) -> Runnable[Dict[str, Any], Union[AgentAction, AgentFinish]]: 
    """
    Create an agent tailored for structured chat environments with multi-input tools.

    This function adapts Langchain's standard agent creation to better support
    tools that require multiple inputs, often represented as a JSON blob or dictionary.
    It ensures that the prompt is correctly formatted and that the output parser
    can handle the structured JSON output from the LLM.

    Args:
        llm: The language model (LLM) to be used as the core of the agent.
        tools: A sequence of `BaseTool` instances that the agent can use.
        prompt: The `ChatPromptTemplate` to guide the agent's reasoning and responses.
                It must include 'tools', 'tool_names', and 'agent_scratchpad'
                input variables.
        tools_renderer: A function responsible for rendering the provided tools
                        into a string format suitable for inclusion in the prompt.
                        Defaults to `render_text_description_and_args`.
        stop_sequence: Determines the stop sequence for the LLM.
                       If True (default), uses "\\nObservation:" to prevent hallucination.
                       If False, no stop sequence is added.
                       If a list of strings, those strings are used as stop sequences.

    Returns:
        A `Runnable` object representing the configured agent. This runnable takes
        a dictionary as input (matching the prompt's input variables) and
        outputs either an `AgentAction` (if a tool is to be called) or an
        `AgentFinish` (if the agent provides a final answer).

    Raises:
        ValueError: If the provided `prompt` is missing any of the required
                    input variables (`tools`, `tool_names`, `agent_scratchpad`).

    Example (from Langchain documentation, adapted for this function):
        ```python
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import AgentExecutor
        from langchain_community.chat_models import ChatOpenAI # Or other LLM
        from hippycampus.langchain_util import fixed_create_structured_chat_agent

        # Assume 'tools' are defined elsewhere (e.g., loaded via OpenAPI)
        # prompt = hub.pull("hwchase17/structured-chat-agent") # Example prompt
        # For a custom prompt, ensure it has 'tools', 'tool_names', 'agent_scratchpad'
        prompt = ChatPromptTemplate.from_messages(...) # Define your prompt
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        agent = fixed_create_structured_chat_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        response = agent_executor.invoke({"input": "What is the weather in Paris?"})
        print(response)
        ```
    """
    missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )
    if stop_sequence:
        stop = ["\nObservation"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)
    else:
        llm_with_stop = llm

    agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            )
            | prompt
            | llm_with_stop
            | FixedJSONAgentOutputParser()
    )
    return agent


class FixedJSONAgentOutputParser(AgentOutputParser):
    """
    Parses the JSON output of an LLM to determine agent actions or final answers.

    This parser is designed to handle JSON output that specifies a tool invocation
    (an `AgentAction`) or a final response to the user (an `AgentFinish`).
    It includes fixes for common issues, such as:
    - LLMs returning a list of actions instead of a single action.
    - Handling `null` or empty `action_input`.
    - Ensuring dictionary `action_input` for "Final Answer" is correctly stringified.

    Expected JSON structure for an `AgentAction`:
    ```json
    {
      "action": "tool_name",
      "action_input": {"arg1": "value1", "arg2": "value2"}
    }
    ```
    or
    ```json
    {
      "action": "tool_name",
      "action_input": "simple string input"
    }
    ```

    Expected JSON structure for an `AgentFinish`:
    ```json
    {
      "action": "Final Answer",
      "action_input": "The final response to the user."
    }
    ```
    or
    ```json
    {
      "action": "Final Answer",
      "action_input": {"key": "structured response"}
    }
    ```
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parses the LLM's JSON output string.

        Args:
            text: The JSON string output from the LLM.

        Returns:
            An `AgentAction` if the LLM indicates a tool should be used,
            or an `AgentFinish` if the LLM provides a final answer.

        Raises:
            OutputParserException: If the JSON cannot be parsed, or if essential
                                   keys like "action" are missing.
        """
        try:
            # Use Langchain's utility to parse JSON that might be in a markdown code block
            response = parse_json_markdown(text)
            
            if isinstance(response, list):
                # Handle cases where the LLM might (incorrectly) return a list of actions.
                # The current implementation takes the first valid action.
                if not response: 
                    raise OutputParserException(f"LLM output was an empty list: {text}")
                response = response[0] # Take the first action
                if not isinstance(response, dict): # Ensure the first item is a dict
                     raise OutputParserException(f"First item in LLM output list is not a JSON object: {response}")

            if not isinstance(response, dict): # Ensure final response is a dict
                raise OutputParserException(f"LLM output, after potential list processing, is not a JSON object: {response}")

            action = response.get("action")
            action_input = response.get("action_input")

            if action is None:
                raise OutputParserException(f"Missing 'action' key in LLM output dictionary: {response}")

            if action == "Final Answer":
                if action_input is None:
                    output = ""  # Treat null input for Final Answer as empty string
                elif isinstance(action_input, dict):
                    # Stringify dictionary inputs for Final Answer as per common agent expectations
                    output = json.dumps(action_input, ensure_ascii=False)
                else:
                    output = str(action_input)
                return AgentFinish({"output": output}, text)
            else: # It's an AgentAction
                tool_input: Any 
                if action_input is None:
                    tool_input = "" 
                elif isinstance(action_input, dict) and not action_input: 
                    tool_input = "" # Empty dict also becomes empty string for some tool inputs
                elif isinstance(action_input, (str, bool, int, float)): # Primitives are passed as is
                    tool_input = action_input
                elif isinstance(action_input, dict):
                     # For structured tools, the dict is often the expected input.
                     # For older tools, it might be stringified. Langchain's StructuredTool handles dicts.
                    tool_input = action_input # Pass dicts as is
                else: # Other types (e.g. list if not handled by schema) stringified
                    tool_input = str(action_input)

                return AgentAction(action, tool_input, text)
        except json.JSONDecodeError as e: 
            raise OutputParserException(
                f"Could not parse LLM output as JSON: {text}. \nParsing error: {e}"
            ) from e
        except OutputParserException: # Re-raise specific OutputParserExceptions
            raise
        except Exception as e: # Catch any other unexpected errors during parsing
            raise OutputParserException(f"Could not parse LLM output: {text}. \nUnexpected error: {e}") from e

    @property
    def _type(self) -> str:
        """Identifies the type of this output parser for Langchain."""
        return "json-agent-fixed" # Distinguish from standard JSONAgentOutputParser if needed
