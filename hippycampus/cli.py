from langchain import hub
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI

from hippycampus.langchain_util import fixed_create_structured_chat_agent
from hippycampus.openapi_builder import load_tools_from_openapi

import json
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.pretty import Pretty

console = Console()


def render_agent_response(response):
    """
    Render an arbitrary agent response using Rich.

    The function attempts to:
      1. Parse the response as JSON and pretty-print it if possible.
      2. Render as Markdown if it looks like markdown.
      3. Fall back to a pretty print of the object.
    """
    # If response is a dict or list, pretty print it
    if isinstance(response, (dict, list)):
        console.print(Pretty(response))
        return

    # Try to parse the response as JSON
    try:
        parsed = json.loads(response)
        pretty_json = json.dumps(parsed, indent=2)
        syntax = Syntax(pretty_json, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
        return
    except Exception:
        pass

    # Optionally, check if it contains markdown markers (e.g., '#', '*', etc.)
    if any(marker in response for marker in ['#', '*', '_', '-', '`']):
        md = Markdown(response)
        console.print(md)
        return

    # Fall back to printing as plain text
    console.print(response)


def main():
    # github_tools = load_tools_from_openapi("../test/api.github.com/1.1.4/openapi.yaml", "<your token here>")
    xkcd_tools = load_tools_from_openapi("./test/xkcd.com/1.0.0/openapi2.yaml")
    tools = xkcd_tools
    print("Generated Tools:")
    for tool in tools:
        print(f"- {tool.name}")

    # Import ChatOpenAI from the updated package.
    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "The module 'langchain_openai' is missing. Please install it via 'pip install langchain-openai'."
        ) from e

    from langchain.agents import AgentExecutor

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # Depending on the LLM used, make sure your API key is set, e.g. OPENAI_API_KEY or GOOGLE_API_KEY
    model = "gemini-2.0-flash-exp"
    # model = "gemini-2.0-flash-thinking-exp-01-21"
    # Initialize the LLM model (adjust parameters as needed).
    llm = ChatGoogleGenerativeAI(model=model, streaming=True, callback_manager=callback_manager)

    # Uncomment to use ChatGPT
    # model = "chatgpt-4o-latest"
    # model = "o1"
    # llm = ChatOpenAI(model=model, streaming=True, callback_manager=callback_manager)

    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent = fixed_create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    agent_executor.invoke({"input": "hi"})
    query = "What is the XKCD that is 1 before the current XKCD?"
    # GitHub API example
    # query = "List my five my recently updated github repositories."

    # query = "Tell me my list of vercel deployments. Pick the first one, and shut it down."
    # query = "Please update the README.md file in the hippy-next-js repo with the current date and commit it to github."
    # query = "Please get the README.md from the hippy-next-js repo with owner cromwellian and print it."
    print(f"\nRunning query {query} against {model}...\n")
    response = agent_executor.invoke({"input": query})
    print("\nAgent Response:")
    render_agent_response(response['output'])


if __name__ == "__main__":
    main()
