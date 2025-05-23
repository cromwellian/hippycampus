from langchain import hub
from langchain_core.callbacks import StreamingStdOutCallbackHandler # CallbackManager is deprecated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.language_models import BaseLanguageModel 
from langchain_core.tools import BaseTool 
from langchain.agents import AgentExecutor 


from hippycampus.langchain_util import fixed_create_structured_chat_agent
from hippycampus.openapi_builder import load_tools_from_openapi

import json
from typing import Any, List, Dict, Union, Optional # Added Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax # Corrected: Syntax is used, not RichJSON directly for this purpose
from rich.pretty import Pretty
from rich.json import JSON as RichJSON # Correct import for RichJSON

# --- Click CLI ---
import click # Moved to top with other imports


console = Console()


def render_agent_response(response: Any, console_instance: Console = console):
    """
    Render an arbitrary agent response using Rich.

    The function attempts to:
      1. Parse the response as JSON and pretty-print it if possible.
      2. Render as Markdown if it looks like markdown.
      3. Fall back to a pretty print of the object or plain text.
    Args:
        response (Any): The agent's response to render.
        console_instance (Console, optional): The Rich Console instance to use. 
                                             Defaults to a global console instance.
    """
    if isinstance(response, (dict, list)):
        console_instance.print(Pretty(response))
        return

    if not isinstance(response, str):
        console_instance.print(Pretty(response)) 
        return

    try:
        if response.strip().startswith(("{", "[")) and response.strip().endswith(("}", "]")):
            parsed_json = json.loads(response) 
            console_instance.print(RichJSON.from_data(parsed_json)) 
            return
    except json.JSONDecodeError: 
        pass 

    markdown_markers = ['#', '*', '_', '`', '\n- ', '\n1. '] 
    if any(marker in response for marker in markdown_markers) or ("\n" in response and len(response) > 80): 
        try:
            md = Markdown(response)
            console_instance.print(md)
            return
        except Exception: 
            pass

    console_instance.print(response)


@click.command()
@click.option(
    "--openapi-spec",
    "openapi_specs", 
    multiple=True,
    help="Path or URL to an OpenAPI spec file. Can be specified multiple times.",
    type=str,
)
@click.option(
    "--model-name",
    default="gemini-1.0-pro", 
    help="Name of the LLM model to use (e.g., 'gemini-1.0-pro', 'gpt-4').",
    type=str,
)
@click.option(
    "--prompt-hub-repo",
    default="hwchase17/structured-chat-agent",
    help="Langchain Hub repository for the agent prompt.",
    type=str,
)
@click.option(
    "--query",
    "query_to_run", 
    help="The input query for the agent. If not provided, runs a default 'hi' query.",
    type=str,
    default=None, 
)
@click.option(
    "--auth-token",
    help="Authentication token for loading OpenAPI specs that require it.",
    type=str,
    default=None,
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output for the AgentExecutor.",
)
def main(
    openapi_specs: List[str], 
    model_name: str, 
    prompt_hub_repo: str, 
    query_to_run: Optional[str], 
    auth_token: Optional[str],
    verbose: bool
):
    """
    HippyCampus CLI: Load OpenAPI tools and run queries against an LLM agent.
    """
    console.print("Starting HippyCampus CLI...", style="bold magenta")

    if not openapi_specs:
        openapi_specs = ["./test/xkcd.com/1.0.0/openapi2.yaml"]
        console.print(f"No OpenAPI spec provided, using default: {openapi_specs[0]}", style="yellow")

    all_tools: List[BaseTool] = []
    for spec_path_or_url in openapi_specs:
        try:
            current_tools = load_tools_from_openapi(spec_path_or_url, auth=auth_token) 
            all_tools.extend(current_tools)
            console.print(f"Loaded {len(current_tools)} tools from: {spec_path_or_url}", style="green")
        except Exception as e:
            console.print(f"Error loading tools from {spec_path_or_url}: {e}", style="red")
            

    if not all_tools:
        console.print("No tools were successfully loaded. Exiting.", style="bold red")
        return

    console.print(f"\nTotal tools loaded: {len(all_tools)}", style="bold blue")
    if verbose:
        console.print("Generated Tools:", style="bold blue")
        for tool in all_tools:
            console.print(f"- {tool.name}: {tool.description[:100] if tool.description else ''}...")

    llm: BaseLanguageModel
    try:
        callbacks_list = [StreamingStdOutCallbackHandler()] if verbose else [] 
        if "gemini" in model_name.lower():
            llm = ChatGoogleGenerativeAI(model=model_name, streaming=True, callbacks=callbacks_list)
        elif "gpt" in model_name.lower() or "openai" in model_name.lower() or model_name.startswith("o1"):
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=model_name, streaming=True, callbacks=callbacks_list)
        else:
            console.print(f"Attempting to use ChatGoogleGenerativeAI as a fallback for unrecognized model: {model_name}", style="yellow")
            llm = ChatGoogleGenerativeAI(model=model_name, streaming=True, callbacks=callbacks_list)
    except ModuleNotFoundError as e:
         console.print(f"Missing module for model '{model_name}': {e}. Please install required packages (e.g., 'langchain-openai', 'langchain-google-genai').", style="bold red")
         return
    except Exception as e:
        console.print(f"Error initializing LLM model '{model_name}': {e}", style="bold red")
        console.print("Ensure API keys (e.g., GOOGLE_API_KEY, OPENAI_API_KEY) are set as environment variables.", style="yellow")
        return

    try:
        prompt: ChatPromptTemplate = hub.pull(prompt_hub_repo)
    except Exception as e:
        console.print(f"Error pulling prompt from Langchain Hub ('{prompt_hub_repo}'): {e}", style="bold red")
        return

    try:
        agent = fixed_create_structured_chat_agent(llm, all_tools, prompt)
        agent_executor: AgentExecutor = AgentExecutor(
            agent=agent, 
            tools=all_tools, 
            verbose=verbose, 
            handle_parsing_errors=True 
        )
    except Exception as e:
        console.print(f"Error creating agent executor: {e}", style="bold red")
        return

    if query_to_run is None:
        if console.is_interactive:
            try:
                query_to_run = console.input("[bold green]Enter your query for the agent (or press Enter for 'hi'):[/] ")
                if not query_to_run: # Handles empty input
                    query_to_run = "hi"
            except KeyboardInterrupt:
                console.print("\nExiting on user request.", style="yellow")
                return
            except Exception as e: 
                console.print(f"Error during input: {e}. Defaulting to 'hi' query.", style="red")
                query_to_run = "hi"
        else: 
            query_to_run = "hi"


    console.print(f"\nRunning query: \"{query_to_run}\" using model: {model_name}", style="bold blue")
    
    try:
        response: Dict[str, Any] = agent_executor.invoke({"input": query_to_run})
        console.print("\nAgent Response:", style="bold blue")
        render_agent_response(response.get('output', "No output from agent."), console_instance=console)
    except Exception as e:
        console.print(f"\nError during agent execution: {e}", style="bold red",_stderr=True)
        if verbose:
            console.print_exception(show_locals=True) 
        else:
            console.print("Enable --verbose for more detailed error information.", style="yellow")


if __name__ == "__main__":
    main()
