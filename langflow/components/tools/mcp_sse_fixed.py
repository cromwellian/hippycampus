import asyncio
from contextlib import AsyncExitStack
from typing import Any, Type, Optional, Dict, List
from typing import Union

import httpx
from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from langchain_core.tools import StructuredTool

from langflow.custom import Component
from langflow.inputs import HandleInput
from langflow.io import MessageTextInput, Output
from mcp import ClientSession, types, Tool
from mcp.client.sse import sse_client
from pydantic import Field, create_model

from hippycampus.openapi_builder import create_input_schema_from_json_schema


def coerce_value_to_type(value: Any, target_type: Type) -> Any:
    """Coerce a value to match the target type."""
    if value is None:
        return None

    # Get the base type (handling Optional[Type])
    if hasattr(target_type, "__origin__") and target_type.__origin__ is Union:
        # Handle Optional (Union[Type, NoneType])
        types = [t for t in target_type.__args__ if t is not type(None)]
        if types:
            target_type = types[0]

    try:
        # Handle basic types
        if target_type is int:
            return int(float(value)) if isinstance(value, (str, float)) else int(value)
        elif target_type is float:
            return float(value) if isinstance(value, (str, int)) else value
        elif target_type is bool:
            if isinstance(value, str):
                return value.lower() in ("yes", "true", "t", "1")
            return bool(value)
        elif target_type is str:
            return str(value)
        elif target_type is list:
            if isinstance(value, str):
                import json
                try:
                    return json.loads(value) if value.startswith("[") else [value]
                except:
                    return [value]
            return list(value) if hasattr(value, "__iter__") and not isinstance(value, str) else [value]
        return value
    except (ValueError, TypeError):
        return value  # Return original value if conversion fails


# Create custom versions of these functions to handle unexpected arguments
def create_tool_coroutine(name: str, args_schema, session):
    """Create a coroutine for a tool that handles unexpected keyword arguments."""

    async def tool_coroutine(**kwargs):
        # Remove callbacks from kwargs if present
        kwargs.pop('callbacks', None)
        kwargs.pop('run_manager', None)
        # Filter out unexpected keyword arguments
        expected_args = {}
        if args_schema:
            schema = args_schema.schema()
            schema_props = schema.get('properties', {})
            field_types = {field: args_schema.__annotations__[field]
                           for field in args_schema.__annotations__
                           if field in schema_props}

            # Only include arguments that are in the schema
            for arg_name, arg_value in kwargs.items():
                # Coerce value to match expected type if needed
                if arg_name in field_types:
                    target_type = field_types[arg_name]
                    arg_value = coerce_value_to_type(arg_value, target_type)

                if arg_name in schema_props:
                    expected_args[arg_name] = arg_value
        else:
            # Filter out non-serializable objects
            expected_args = {k: v for k, v in kwargs.items()
                             if not k.startswith('_') and
                             not callable(v) and
                             not isinstance(v, (AsyncCallbackManagerForToolRun, type))}

        # Call the tool with the filtered arguments
        response = await session.call_tool(name, expected_args)
        response = response.content
        # Extract text content from response if it's a list
        if isinstance(response, list) and len(response) > 0:
            return response[0].text if hasattr(response[0], 'text') else str(response[0])
        return str(response.text) if hasattr(response, 'text') else str(response)

    return tool_coroutine


def create_tool_func(name: str, session):
    """Create a function for a tool that handles unexpected keyword arguments."""

    def tool_func(**kwargs):
        # Remove callbacks from kwargs if present
        kwargs.pop('callbacks', None)
        kwargs.pop('run_manager', None)

        # Filter out unexpected keyword arguments
        expected_args = {}

        # Get the tool schema from the session if possible
        tool_schema = None
        if hasattr(session, 'tools_info'):
            tool_info = next((t for t in session.tools_info if t.name == name), None)
            if tool_info and hasattr(tool_info, 'args_schema'):
                tool_schema = tool_info.args_schema

        if tool_schema:
            schema_props = tool_schema.schema().get('properties', {})
            field_types = {field: tool_schema.__annotations__[field]
                           for field in tool_schema.__annotations__
                           if field in schema_props}
            for arg_name, arg_value in kwargs.items():
                if arg_name in field_types:
                    expected_args[arg_name] = coerce_value_to_type(arg_value, field_types[arg_name])
        else:
            # Filter out non-serializable objects
            expected_args = {k: v for k, v in kwargs.items()
                             if not k.startswith('_') and
                             not callable(v) and
                             not isinstance(v, (AsyncCallbackManagerForToolRun, type))}

        # Use asyncio to run the coroutine
        response = asyncio.run(session.call_tool(name, expected_args))
        # Extract text content from response if it's a list
        if isinstance(response, list) and len(response) > 0:
            return response[0].text if hasattr(response[0], 'text') else str(response[0])
        return str(response)

    return tool_func


class MCPSseClient:
    def __init__(self):
        self.session = None
        self.write = None
        self.sse = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, url: str, headers: dict[str, str] | None = None):
        if headers is None:
            headers = {}

        # Close existing session if it exists
        if self.session is not None:
            try:
                await self.session.aclose()
            except:
                pass

        # Create new SSE transport and session without timeout parameters
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers)
        )
        self.sse, self.write = sse_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.sse, self.write))
        await self.session.initialize()

        # Get the list of available tools
        response = await self.session.list_tools()
        return response.tools


class MCPSse(Component):
    client = MCPSseClient()
    tools = types.ListToolsResult
    tool_names = [str]
    display_name = "Hippycampus MCP Server"
    description = "Connects to an MCP server over SSE and exposes its tools as langflow tools to be used by an Agent."
    documentation: str = "http://docs.langflow.org/components/custom"
    icon = "code"
    name = "MCPSseFixed"

    inputs = [
        MessageTextInput(
            name="url",
            display_name="Hippycampus MCP URL",
            info="sse url",
            value="http://localhost:8000/sse",
            tool_mode=True,
        ),
        HandleInput(
            name="openapi_definitions",
            display_name="OpenAPI YAML URLs",
            input_types=["OpenApiSpec"],
            is_list=True,
            required=True,
            info="List of OpenAPI YAML URLs to load tools from",
        ),
    ]

    outputs = [
        Output(display_name="Tools", name="tools", method="build_output"),
    ]

    def __init__(self, **data):
        super().__init__(**data)
        self.client = MCPSseClient()  # This is just a container class, not the actual ClientSession
        self.tools = []
        self.tool_names = []

    async def build_output(self) -> list[Tool]:
        # Connect to server if not already connected
        if not hasattr(self.client, 'session') or self.client.session is None:
            self.tools = await self.client.connect_to_server(self.url, {})

        tool_list = []

        # Load OpenAPI definitions sequentially to avoid session conflicts
        for openapi_definition in self.openapi_definitions:
            print(f"Loading tools from {openapi_definition.url}")

            # Construct the load_openapi URL
            base_url = self.url
            if base_url.endswith("/sse"):
                base_url = base_url[:-4]  # Remove "/sse" from the end

            load_url = f"{base_url}/load_openapi?url={openapi_definition.url}"
            if openapi_definition.token:
                load_url += f"&token={openapi_definition.token}"

            # Load each spec in a separate, isolated client
            try:
                # Create a completely separate client for each request
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(load_url)
                    response.raise_for_status()
                    result = response.json()

                # Add a small delay to ensure the server has time to process
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"Error loading OpenAPI spec: url={load_url},{str(e)}")

        # Explicitly close and reconnect to get a fresh session
        try:
            if hasattr(self.client, 'session') and self.client.session is not None:
                if hasattr(self.client.session, 'aclose'):
                    await self.client.session.aclose()
                elif hasattr(self.client.session, 'close'):
                    self.client.session.close()
        except Exception as e:
            print(f"Error closing session: {str(e)}")

        # Create a fresh connection
        self.tools = await self.client.connect_to_server(self.url, {})

        # Now build the tool list
        for tool in self.tools:
            args_schema = create_input_schema_from_json_schema(tool.inputSchema)
            tool_list.append(
                StructuredTool(
                    name=tool.name,
                    description=tool.description,
                    args_schema=args_schema,
                    coroutine=create_tool_coroutine(name=tool.name, args_schema=args_schema,
                                                    session=self.client.session),
                    func=create_tool_func(name=tool.name, session=self.client.session),
                )
            )

        self.tool_names = [tool.name for tool in self.tools]
        print(f"Tool list: {tool_list}")
        return tool_list

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected"""
        if hasattr(self, 'client') and self.client is not None:
            if hasattr(self.client, 'session') and self.client.session is not None:
                try:
                    # Try to close synchronously if possible
                    if hasattr(self.client.session, 'close'):
                        self.client.session.close()
                except Exception:
                    # Just log and continue if there's an error during cleanup
                    import sys
                    print("Error during cleanup in __del__", file=sys.stderr)
