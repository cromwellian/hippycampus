import asyncio
import base64
import json
import os
import hashlib
import inspect
from typing import Union, List, Dict, Any, Optional

import httpx
import logging
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from langchain_core.tools import BaseTool

from hippycampus.openapi_builder import OpenAPIToolBuilder
from hippycampus.tool_auth.authentication import AbstractAuth

logger = logging.getLogger(__name__)


# --- Tool Registry Class ---
class ToolRegistry:
    """Encapsulates tool registration and management."""

    def __init__(self):
        self.langchain_tools: Dict[str, BaseTool] = {}
        self.external_docs_for_tools: Dict[str, str] = {}
        self.external_metadata_for_tools: Dict[str, Dict[str, Any]] = {}
        logger.info("ToolRegistry initialized.")

    def register_tool(
        self,
        name: str,
        tool_instance: BaseTool,
        docs_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registers a tool with its documentation URL and metadata."""
        if name in self.langchain_tools:
            logger.warning(f"Tool '{name}' is being replaced in registry.")
        self.langchain_tools[name] = tool_instance
        if docs_url:
            self.external_docs_for_tools[name] = docs_url
        if metadata:
            self.external_metadata_for_tools[name] = metadata
        logger.info(
            f"Tool '{name}' registered. Docs: {docs_url is not None}, Meta: {metadata is not None}"
        )

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self.langchain_tools.get(name)

    def get_external_doc_url(self, name: str) -> Optional[str]:
        return self.external_docs_for_tools.get(name)

    def get_external_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        return self.external_metadata_for_tools.get(name)

    def list_tools_with_details(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered tools with their details."""
        tool_list = []
        for name, tool_instance in self.langchain_tools.items():
            tool_info: Dict[str, Any] = {
                "name": name,
                "description": tool_instance.description,
                "args_schema": tool_instance.args_schema.model_json_schema()
                if tool_instance.args_schema
                and hasattr(tool_instance.args_schema, "model_json_schema")
                else None,
            }
            if name in self.external_docs_for_tools:
                tool_info["external_doc_url"] = self.external_docs_for_tools[name]
            if name in self.external_metadata_for_tools:
                tool_info["external_metadata"] = self.external_metadata_for_tools[name]
            tool_list.append(tool_info)
        return tool_list


# --- Helper Functions ---
async def fetch_website(url: str) -> str:
    """Fetches a website and returns its content as a string."""
    headers = {
        "User-Agent": "MCP Hippycampus Server (github.com/hippycampus/hippycampus)"
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPStatusError as e:
        error_message = f"Error fetching website: Client error '{e.response.status_code} {e.response.reason_phrase}' for url '{e.request.url}'."
        # Limit response preview to avoid overly long error messages
        response_text_preview = e.response.text[:500] if e.response.text else ""
        if response_text_preview:
            error_message += f" Response: {response_text_preview}"
        logger.error(error_message)
        return error_message
    except httpx.RequestError as e:
        error_message = f"Request error for url '{e.request.url}': {e}"
        logger.error(error_message)
        return error_message


def encode_as_base64(data: Union[str, bytes]) -> str:
    """Encode string or bytes content as a base64 string."""
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data
    return base64.b64encode(data_bytes).decode("utf-8")


def decode_base64(encoded_content: str) -> str:
    """Decode base64 encoded content to string."""
    try:
        return base64.b64decode(encoded_content).decode("utf-8")
    except (TypeError, base64.binascii.Error) as e:
        logger.error(
            f"Invalid base64 string for decoding: {encoded_content[:50]}... Error: {e}"
        )  # Log part of content for context
        raise ValueError("Invalid base64 string provided.")


def compute_git_commit_sha(content: Union[str, bytes]) -> str:
    """Compute SHA-1 hash for content."""
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    else:
        content_bytes = content
    sha1_hash = hashlib.sha1(content_bytes)
    return sha1_hash.hexdigest()


def fetch_documentation_for_tool(
    tool_name: str,
    langchain_tools_dict: Dict[str, BaseTool],
    external_docs_dict: Dict[str, str],
    external_metadata_dict: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fetches documentation and metadata for a given tool.
    """
    tool_instance = langchain_tools_dict.get(tool_name)
    if not tool_instance:
        raise ValueError(f"Tool '{tool_name}' not found for documentation retrieval.")

    tool_info: Dict[str, Any] = {
        "name": tool_instance.name,
        "description": tool_instance.description,
        "args_schema": tool_instance.args_schema.model_json_schema()
        if tool_instance.args_schema
        and hasattr(tool_instance.args_schema, "model_json_schema")
        else None,
    }
    doc_url = external_docs_dict.get(tool_name)
    if doc_url:
        tool_info["external_doc_url"] = doc_url

    metadata = external_metadata_dict.get(tool_name)
    if metadata:
        tool_info["external_metadata"] = metadata
        examples_text_parts = []
        request_examples = metadata.get("requestExamples", {})
        if isinstance(request_examples, dict):  # Ensure it's a dict before iterating
            for ex_name, ex_data in request_examples.items():
                if isinstance(ex_data, dict):
                    examples_text_parts.append(
                        f"Example Input ({ex_name}): {json.dumps(ex_data.get('value'))}"
                    )

        response_examples = metadata.get("responseExamples", {})
        if isinstance(response_examples, dict):
            for status, resp_ex_map in response_examples.items():
                if isinstance(resp_ex_map, dict):
                    for ex_name, ex_data in resp_ex_map.items():
                        if isinstance(ex_data, dict):
                            examples_text_parts.append(
                                f"Example Response ({status} - {ex_name}): {json.dumps(ex_data.get('value'))}"
                            )
        if examples_text_parts:
            tool_info["examples_text"] = "\n".join(examples_text_parts)

    return tool_info


# --- Tool Registration and Loading (Refactored to use ToolRegistry via app.state) ---
async def load_openapi(request: Request) -> JSONResponse:
    """
    Endpoint to load an OpenAPI specification from a URL (http/https or file)
    and register the derived tools into the app's ToolRegistry.
    """
    url = request.query_params.get("url")
    if not url:
        return JSONResponse(
            {
                "error_type": "MissingParameter",
                "message": "URL query parameter is required.",
            },
            status_code=400,
        )

    if not hasattr(request.app.state, "tool_registry") or not isinstance(
        request.app.state.tool_registry, ToolRegistry
    ):
        logger.critical(
            "ToolRegistry not found in app.state. This indicates a server misconfiguration."
        )
        return JSONResponse(
            {
                "error_type": "ServerConfigurationError",
                "message": "ToolRegistry not available.",
            },
            status_code=500,
        )

    tool_registry: ToolRegistry = request.app.state.tool_registry
    spec_content: Optional[str] = None
    file_path_for_error: str = url  # Use URL in error messages unless it's a file path

    try:
        if url.startswith("file://"):
            file_path = url[7:]
            file_path_for_error = os.path.basename(
                file_path
            )  # For user-friendly error messages
            if ".." in file_path or not os.path.isabs(file_path):
                return JSONResponse(
                    {
                        "error_type": "InvalidFilePath",
                        "message": "Invalid local file path provided.",
                    },
                    status_code=400,
                )
            if not os.path.exists(file_path):  # Check existence before opening
                raise FileNotFoundError(
                    f"Local OpenAPI spec file not found: {file_path_for_error}"
                )
            with open(file_path, "r", encoding="utf-8") as f:
                spec_content = f.read()
            logger.info(f"Loaded OpenAPI spec from local file: {file_path}")
        else:
            spec_content = await fetch_website(url)
            if spec_content.startswith(
                "Error fetching website"
            ) or spec_content.startswith("Request error"):
                raise httpx.RequestError(
                    spec_content
                )  # Let the specific exception handler below catch this
            logger.info(f"Fetched OpenAPI spec from URL: {url}")

        if not spec_content:
            return JSONResponse(
                {
                    "error_type": "NoContent",
                    "message": "Failed to retrieve OpenAPI spec content.",
                },
                status_code=400,
            )

        builder = OpenAPIToolBuilder(spec=spec_content)

        auth_obj: Optional[AbstractAuth] = None
        # Placeholder for auth resolution based on request.headers or other context
        # Example: if hasattr(request.app.state, 'auth_resolver'):
        # auth_header = request.headers.get("Authorization")
        # if auth_header: auth_obj = request.app.state.auth_resolver.resolve(auth_header, url_being_loaded=url)

        loaded_lc_tools = builder.build_tools(auth=auth_obj)

        tools_loaded_info = []
        for tool_instance in loaded_lc_tools:
            docs_url = (
                tool_instance.metadata.get("externalDocs", {}).get("url")
                if tool_instance.metadata
                else None
            )
            tool_registry.register_tool(
                name=tool_instance.name,
                tool_instance=tool_instance,
                docs_url=docs_url,
                metadata=tool_instance.metadata,
            )
            tools_loaded_info.append(
                {
                    "name": tool_instance.name,
                    "description": tool_instance.description,
                }
            )

        logger.info(
            f"Successfully loaded and registered {len(tools_loaded_info)} tools from '{url}'."
        )
        return JSONResponse(
            {
                "message": "OpenAPI spec loaded and tools registered successfully.",
                "tools_loaded": tools_loaded_info,
            }
        )

    except FileNotFoundError as e:
        logger.error(
            f"FileNotFoundError in load_openapi for '{file_path_for_error}': {e}"
        )
        return JSONResponse(
            {
                "error_type": "FileNotFound",
                "message": f"File not found: {file_path_for_error}.",
            },
            status_code=400,
        )
    except PermissionError as e:
        logger.error(
            f"PermissionError in load_openapi for '{file_path_for_error}': {e}"
        )
        return JSONResponse(
            {
                "error_type": "PermissionError",
                "message": f"Permission denied for file operation: {file_path_for_error}.",
            },
            status_code=403,
        )
    except (
        httpx.RequestError
    ) as e:  # This will catch the re-raised error from fetch_website
        logger.error(f"HTTP RequestError in load_openapi for URL '{url}': {e}")
        return JSONResponse(
            {
                "error_type": "RequestError",
                "message": f"Failed to fetch OpenAPI spec from URL: {url}. Detail: {str(e)}",
            },
            status_code=400,
        )
    except ValueError as e:
        logger.error(
            f"ValueError in load_openapi (spec parsing/tool building for '{url}'): {e}"
        )
        return JSONResponse(
            {
                "error_type": "SpecProcessingError",
                "message": f"Failed to parse OpenAPI spec or build tools: {str(e)}",
            },
            status_code=400,
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in load_openapi for URL '{url}': {e}", exc_info=True
        )
        return JSONResponse(
            {
                "error_type": "InternalServerError",
                "message": "An unexpected error occurred while processing the OpenAPI specification.",
            },
            status_code=500,
        )


# --- Core Tool Dispatch Logic (Refactored to use ToolRegistry) ---
async def fetch_tool_from_registry(
    tool_name: str, arguments: Dict[str, Any], tool_registry: ToolRegistry
) -> Any:
    """
    Fetches and executes a tool using the provided ToolRegistry.
    Default tools are checked first, then dynamically loaded LangChain tools from the registry.
    """
    logger.info(f"Fetching tool: '{tool_name}' with arguments: {arguments}")

    # Default tools (not in registry, handled by name)
    if tool_name == "fetch":
        url_to_fetch = arguments.get("url")
        if not url_to_fetch or not isinstance(url_to_fetch, str):
            raise ValueError(
                "URL argument is required for fetch tool and must be a string."
            )
        return await fetch_website(url_to_fetch)
    elif tool_name == "encode_as_base64":
        data_to_encode = arguments.get("data")
        if data_to_encode is None:  # Allow empty string, but None is invalid.
            raise ValueError("Data argument is required for encode_as_base64 tool.")
        return encode_as_base64(data_to_encode)
    elif tool_name == "decode_base64":
        data_to_decode = arguments.get("data")
        if not data_to_decode or not isinstance(data_to_decode, str):
            raise ValueError(
                "Data argument is required for decode_base64 tool and must be a non-empty string."
            )
        return decode_base64(data_to_decode)  # Can raise ValueError if invalid base64
    elif tool_name == "compute_git_commit_sha":
        data_to_hash = arguments.get("data")
        if data_to_hash is None:
            raise ValueError(
                "Data argument is required for compute_git_commit_sha tool."
            )
        return compute_git_commit_sha(data_to_hash)
    elif tool_name == "fetch_documentation_for_tool":
        doc_tool_name = arguments.get("tool_name")
        if not doc_tool_name or not isinstance(doc_tool_name, str):
            raise ValueError(
                "tool_name argument is required for fetch_documentation_for_tool."
            )

        # Create temporary dicts for the helper as it expects them
        temp_langchain_tools = (
            {doc_tool_name: tool_registry.get_tool(doc_tool_name)}
            if tool_registry.get_tool(doc_tool_name)
            else {}
        )
        temp_docs = (
            {doc_tool_name: tool_registry.get_external_doc_url(doc_tool_name)}
            if tool_registry.get_external_doc_url(doc_tool_name)
            else {}
        )
        temp_meta = (
            {doc_tool_name: tool_registry.get_external_metadata(doc_tool_name)}
            if tool_registry.get_external_metadata(doc_tool_name)
            else {}
        )
        # fetch_documentation_for_tool itself raises ValueError if tool not in temp_langchain_tools
        return fetch_documentation_for_tool(
            doc_tool_name, temp_langchain_tools, temp_docs, temp_meta
        )

    tool_instance = tool_registry.get_tool(tool_name)
    if not tool_instance:
        raise ValueError(f"Tool '{tool_name}' not found in registry.")

    logger.info(
        f"Found dynamic tool in registry: '{tool_name}', type: {type(tool_instance)}"
    )

    try:
        # Langchain tools handle input parsing via Pydantic args_schema when invoke/ainvoke is called.
        # The `arguments` dict should contain keys matching the tool's args_schema.
        if inspect.iscoroutinefunction(getattr(tool_instance, "ainvoke", None)):
            logger.info(f"A-Invoking tool '{tool_name}' with input: {arguments}")
            return await tool_instance.ainvoke(input=arguments)  # type: ignore
        elif hasattr(tool_instance, "invoke"):
            logger.info(
                f"Invoking tool '{tool_name}' with input: {arguments} (sync in executor)"
            )
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: tool_instance.invoke(input=arguments)
            )  # type: ignore
        # Fallback for older/simpler tools if invoke/ainvoke are not standard.
        elif inspect.iscoroutinefunction(tool_instance._arun):
            logger.info(f"A-Running tool '{tool_name}' with kwargs: {arguments}")
            return await tool_instance._arun(**arguments)
        elif hasattr(tool_instance, "_run"):
            logger.info(
                f"Running tool '{tool_name}' with kwargs: {arguments} (sync in executor)"
            )
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, tool_instance._run, **arguments)
        else:
            raise NotImplementedError(
                f"Tool '{tool_name}' does not have a standard execution method (invoke, ainvoke, _run, _arun)."
            )

    except Exception as e:  # Catch execution errors from the tool itself
        logger.error(
            f"Error during execution of tool '{tool_name}': {e}", exc_info=True
        )
        # Re-raise to be caught by call_tool_endpoint for a 500 response.
        # Or, could return a specific error structure if tools are expected to fail gracefully.
        raise  # Let call_tool_endpoint handle the final JSON response


# --- Starlette Application Setup ---
async def on_startup():
    """Initialize resources or load configurations when the app starts."""
    logger.info("MCP Server starting up...")
    app.state.tool_registry = ToolRegistry()
    # Example: Register additional default tools or perform other setup
    # from langchain_community.tools import WikipediaQueryRun
    # from langchain_community.utilities import WikipediaAPIWrapper
    # wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    # app.state.tool_registry.register_tool(wikipedia_tool.name, wikipedia_tool)


async def on_shutdown():
    """Clean up resources when the app shuts down."""
    logger.info("MCP Server shutting down...")


async def call_tool_endpoint(request: Request) -> JSONResponse:
    """Endpoint to call a registered tool using the ToolRegistry from app.state."""
    tool_name = request.path_params["tool_name"]
    try:
        payload = await request.json()
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            return JSONResponse(
                {
                    "error_type": "InvalidPayload",
                    "message": "Arguments must be a JSON object if provided.",
                },
                status_code=400,
            )
    except json.JSONDecodeError:
        return JSONResponse(
            {"error_type": "InvalidJSON", "message": "Invalid JSON payload."},
            status_code=400,
        )

    try:
        if not hasattr(request.app.state, "tool_registry"):
            logger.critical(
                "ToolRegistry not found in app.state during call_tool_endpoint. Server not configured correctly."
            )
            return JSONResponse(
                {
                    "error_type": "ServerConfigurationError",
                    "message": "ToolRegistry not initialized.",
                },
                status_code=500,
            )

        tool_registry: ToolRegistry = request.app.state.tool_registry
        result = await fetch_tool_from_registry(tool_name, arguments, tool_registry)
        return JSONResponse({"result": result})
    except ValueError as e:
        logger.warning(f"ValueError in call_tool_endpoint for '{tool_name}': {e}")
        error_type = (
            "ToolNotFound" if "not found" in str(e).lower() else "InvalidArguments"
        )
        status_code = 404 if error_type == "ToolNotFound" else 400
        return JSONResponse(
            {"error_type": error_type, "message": str(e)}, status_code=status_code
        )
    except TypeError as e:
        logger.warning(
            f"TypeError in call_tool_endpoint for '{tool_name}' with args {arguments}: {e}"
        )
        return JSONResponse(
            {
                "error_type": "ArgumentTypeError",
                "message": f"Invalid argument type or structure for tool '{tool_name}': {e}",
            },
            status_code=400,
        )
    except NotImplementedError as e:  # If a tool doesn't have a runnable method
        logger.error(f"NotImplementedError for tool '{tool_name}': {e}", exc_info=True)
        return JSONResponse(
            {"error_type": "ToolNotRunnable", "message": str(e)}, status_code=501
        )  # Not Implemented
    except Exception as e:
        logger.error(
            f"Unexpected error executing tool '{tool_name}': {e}", exc_info=True
        )
        return JSONResponse(
            {
                "error_type": "ToolExecutionError",
                "message": f"An internal error occurred while executing tool '{tool_name}'.",
            },
            status_code=500,
        )


async def list_tools_endpoint(request: Request) -> JSONResponse:
    """Endpoint to list all available tools with their details from the ToolRegistry."""
    if not hasattr(request.app.state, "tool_registry"):
        logger.critical(
            "ToolRegistry not found in app.state for listing tools. Server not configured correctly."
        )
        return JSONResponse(
            {
                "error_type": "ServerConfigurationError",
                "message": "ToolRegistry not available.",
            },
            status_code=500,
        )

    tool_registry: ToolRegistry = request.app.state.tool_registry
    tools_details = tool_registry.list_tools_with_details()
    return JSONResponse({"tools": tools_details})


routes = [
    Route("/load_openapi", endpoint=load_openapi, methods=["POST"]),
    Route("/call_tool/{tool_name}", endpoint=call_tool_endpoint, methods=["POST"]),
    Route("/list_tools", endpoint=list_tools_endpoint, methods=["GET"]),
]

app = Starlette(routes=routes, on_startup=[on_startup], on_shutdown=[on_shutdown])


if __name__ == "__main__":
    import uvicorn

    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    port_str = os.getenv("MCP_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        logger.error(f"Invalid MCP_PORT value: '{port_str}'. Defaulting to 8000.")
        port = 8000

    uvicorn.run(app, host="0.0.0.0", port=port)
