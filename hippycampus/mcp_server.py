import asyncio
import base64
import json
import os
import tempfile
import traceback
from typing import Union

import anyio
import click
import httpx
import mcp.types as types
from hippycampus.openapi_builder import load_tools_from_openapi, create_input_schema_from_json_schema
from mcp.server.lowlevel import Server
from starlette.responses import JSONResponse


async def fetch_website(
        url: str,
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    headers = {
        "User-Agent": "MCP Hippycampus Server (github.com/hippycampus/hippycampus)"
    }
    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()
        return [types.TextContent(type="text", text=response.text)]


external_docs_for_tools = {}
external_metadata_for_tools = {}


async def fetch_documentation_for_tool(tool_name: str) -> list[
    Union[types.TextContent, types.ImageContent, types.EmbeddedResource]]:
    tool_name = json.loads(tool_name)['tool_name']
    url = external_docs_for_tools.get(tool_name, None)
    metadata = external_metadata_for_tools.get(tool_name, None)
    docs = []
    if metadata is not None:
        docs.append("Here are some examples of how to use this tool:\n")
        for example in metadata.get('requestExamples', {}).values():
            docs.append(f"Example Input: {example['value']}")
        for status_code, examples in metadata.get('responseExamples', {}).items():
            for example in examples.values():
                docs.append(f"Example Response: {example['value']}")
    else:
        docs.append("No external documentation available for this tool.")
    # else:
    #     content = await fetch_website(url)
    #     docs.append(content[0].text)
    return [types.TextContent(type="text", text="\n".join(docs))]


async def load_openapi(request):
    """
    Handle loading OpenAPI specifications from a URL or local file and register the tools.
    The URL is specified as a query parameter 'url'.
    If the URL starts with '/', it's treated as a local file path.
    """
    url = request.query_params.get('url')
    print(f"Loading OpenAPI from {url}")
    token = request.query_params.get('token')  # Optional auth token

    if not url:
        return JSONResponse(
            {"error": "Missing required query parameter 'url'"},
            status_code=400
        )

    try:
        # Check if the URL is a local file path
        if url.startswith('/'):
            print(f"Loading local file: {url}")
            # Load from local file system
            try:
                with open(url, 'r') as file:
                    spec_content = file.read()

                # Create a temporary file for processing
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                    temp_file.write(spec_content)
                    temp_path = temp_file.name
            except FileNotFoundError:
                return JSONResponse(
                    {"error": f"Local file not found: {url}"},
                    status_code=404
                )
            except PermissionError:
                return JSONResponse(
                    {"error": f"Permission denied when accessing file: {url}"},
                    status_code=403
                )
        else:
            # Fetch the OpenAPI specification from URL
            headers = {
                "User-Agent": "MCP Hippycampus Server (github.com/hippycampus/hippycampus)"
            }
            async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()
                spec_content = response.text

            # Save the spec to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
                temp_file.write(spec_content)
                temp_path = temp_file.name

        try:
            # Load tools from the OpenAPI spec
            tools = load_tools_from_openapi(temp_path, token, url)

            # Register each tool with the server
            registered_tools = []
            for tool in tools:
                # Convert LangChain tool to MCP tool
                mcp_tool = types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.args_schema.schema() if hasattr(tool, 'args_schema') else {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"]
                    }
                )
                print(f"Tool metadata: {tool.metadata}")
                if tool.metadata is not None and tool.metadata.get('externalDocs', None) is not None:
                    externalDocs = tool.metadata.get('externalDocs', {})
                    external_docs_for_tools[tool.name] = externalDocs.get('url', None) if isinstance(externalDocs,
                                                                                                     dict) else externalDocs
                    external_metadata_for_tools[tool.name] = tool.metadata

                # Register the tool
                request.app.state.register_tool(tool, mcp_tool)
                registered_tools.append(mcp_tool.name)

            return JSONResponse({
                "status": "success",
                "message": f"Successfully loaded {len(registered_tools)} tools from OpenAPI specification",
                "tools": registered_tools
            })
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to load OpenAPI specification: {str(e)}"},
            status_code=500
        )


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    print("Starting MCP server...")
    app = Server("mcp-website-fetcher")

    # Create a list to store dynamically registered tools
    dynamic_tools = []
    langchain_tools = {}  # Store LangChain tools by name

    # Add the default fetch tool
    default_tools = [
        types.Tool(
            name="encode_as_base64",
            description="Encode content as a base64 string. Returns the encoded string as text.",
            inputSchema={
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "String content to encode (will be UTF-8 encoded)"
                            },
                            {
                                "type": "object",
                                "description": "Bytes-like object to encode"
                            }
                        ],
                        "description": "the string or bytes content to encode",
                    }
                },
            },
        ),
        types.Tool(
            name="decode_base64",
            description="Decode base64 encoded content to string.",
            inputSchema={
                "type": "object",
                "required": ["encoded_content"],
                "properties": {
                    "encoded_content": {
                        "type": "string",
                        "description": "The string to decode",
                    }
                },
            },
        ),
        types.Tool(
            name="compute_git_commit_sha",
            description="Compute Github SHA-1 hash for content.",
            inputSchema={
                "type": "object",
                "required": ["content"],
                "properties": {
                    "content": {
                        "oneOf": [
                            {
                                "type": "string",
                                "description": "String content to hash (will be UTF-8 encoded)"
                            },
                            {
                                "type": "object",
                                "description": "Bytes-like object to hash"
                            }
                        ],
                        "description": "the string or bytes content to compute SHA-1 hash for",
                    }
                },
            },
        ),

        types.Tool(
            name="fetch_documentation_for_tool",
            description="Fetch additional external documentation for a tool. Returns result as string.",
            inputSchema={
                "type": "object",
                "required": ["tool_name"],
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Name of the tool to fetch external documentation for",
                    }
                },
            },
        ),
        # types.Tool(
        #     name="fetch",
        #     description="Fetches a website and returns its content",
        #     inputSchema={
        #         "type": "object",
        #         "required": ["url"],
        #         "properties": {
        #             "url": {
        #                 "type": "string",
        #                 "description": "URL to fetch",
        #             }
        #         },
        #     },
        # )
    ]

    # Function to register new tools
    def register_tool(langchain_tool, mcp_tool):
        """Register a new tool dynamically"""
        # Check if a tool with the same name already exists
        for existing_tool in dynamic_tools:
            if existing_tool.name == mcp_tool.name:
                # Replace the existing tool
                dynamic_tools.remove(existing_tool)
                break
        dynamic_tools.append(mcp_tool)
        langchain_tools[mcp_tool.name] = langchain_tool

    def encode_as_base64(content: Union[bytes, str]) -> str:
        """Encode content as a base64 string."""
        content = json.loads(content)['content']
        print(f"Encoding content: {content}")
        if isinstance(content, str):
            content = content.encode('utf-8')
        result = {}
        result['base64result'] = base64.b64encode(content).decode("utf-8")
        return json.dumps(result)

    # Function to decode base64 to string
    def decode_base64(encoded_content: str) -> str:
        encoded_content = json.loads(encoded_content)['encoded_content']
        print(f"Decoding content: {encoded_content}")

        """Decode base64 encoded content to string."""
        return base64.b64decode(encoded_content).decode("utf-8")

    import hashlib

    def compute_git_commit_sha(content: Union[bytes, str]) -> str:
        """Compute Github SHA-1 hash for content."""
        content = json.loads(content)['content']

        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        prefix = f"blob {len(content)}\0".encode()
        sha1_hash = hashlib.sha1(prefix + content_bytes)
        return sha1_hash.hexdigest()

    # Create LangChain tools for the default tools
    from langchain.tools import StructuredTool

    # Create and register the encode_as_base64 tool, these are needed by GitHub repo operations.
    encode_tool = StructuredTool.from_function(
        func=encode_as_base64,
        name="encode_as_base64",
        description="Encode content as a base64 string.",
        args_schema=create_input_schema_from_json_schema(default_tools[0].inputSchema)
    )
    register_tool(encode_tool, default_tools[0])

    decode_tool = StructuredTool.from_function(
        func=decode_base64,
        name="decode_base64",
        description="Decode base64 encoded content to string.",
        args_schema=create_input_schema_from_json_schema(default_tools[1].inputSchema)
    )
    register_tool(decode_tool, default_tools[1])

    # Create and register the compute_git_commit_sha tool
    sha_tool = StructuredTool.from_function(
        func=compute_git_commit_sha,
        name="compute_git_commit_sha",
        description="Compute Github SHA-1 hash for content.",
        args_schema=create_input_schema_from_json_schema(default_tools[0].inputSchema)
    )
    register_tool(sha_tool, default_tools[2])

    # Create and register the fetch_documentation_for_tool tool
    docs_tool = StructuredTool.from_function(
        coroutine=fetch_documentation_for_tool,
        name="fetch_documentation_for_tool",
        description="Fetch additional external documentation for a tool if available.",
        args_schema=create_input_schema_from_json_schema(default_tools[1].inputSchema)
    )
    register_tool(docs_tool, default_tools[3])

    @app.call_tool()
    async def fetch_tool(
            name: str, arguments: dict
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name == "fetch":
            if "url" not in arguments:
                raise ValueError("Missing required argument 'url'")
            return await fetch_website(arguments["url"])
        elif name in langchain_tools:
            # Handle dynamically registered tools
            tool = langchain_tools[name]
            try:
                # Different approach: Use the tool's _parse_input method to handle the arguments
                # This should properly format the arguments according to what the tool expects
                parsed_input = None

                # If the tool has a _parse_input method, use it
                if hasattr(tool, '_parse_input'):
                    try:
                        # Some tools expect a string input, others expect a dict
                        if hasattr(tool, 'args_schema'):
                            parsed_input = tool._parse_input(arguments)
                        else:
                            # For tools without args_schema, try passing as string
                            parsed_input = tool._parse_input(str(arguments))
                    except Exception:
                        # If parsing fails, try a different approach
                        parsed_input = arguments
                else:
                    # No _parse_input method, use arguments directly
                    parsed_input = arguments

                # Use the proper tool execution methods
                if hasattr(tool, 'arun'):
                    # Handle async tools
                    if isinstance(parsed_input, dict):
                        if hasattr(tool, 'args_schema'):
                            schema_props = tool.args_schema.schema().get('properties', {})
                            filtered_args = {k: v for k, v in parsed_input.items() if k in schema_props}
                            # Convert dict to string for tool_input
                            tool_input = json.dumps(filtered_args)
                            result = await tool.arun(tool_input=tool_input)
                        else:
                            result = await tool.arun(tool_input=str(parsed_input))
                    else:
                        result = await tool.arun(tool_input=parsed_input)
                elif hasattr(tool, 'run'):
                    # Handle sync tools, although we should deprecate and remove this
                    if isinstance(parsed_input, dict):
                        if hasattr(tool, 'args_schema'):
                            schema_props = tool.args_schema.schema().get('properties', {})
                            filtered_args = {k: v for k, v in parsed_input.items() if k in schema_props}
                            # Convert dict to string for tool_input
                            tool_input = json.dumps(filtered_args)
                            result = tool.run(tool_input=tool_input)
                        else:
                            result = tool.run(tool_input=str(parsed_input))
                    else:
                        result = tool.run(tool_input=parsed_input)
                elif hasattr(tool, 'func'):
                    # For function-based tools
                    if asyncio.iscoroutinefunction(tool.func):
                        result = await tool.func(parsed_input)
                    else:
                        result = tool.func(parsed_input)
                else:
                    # Last resort
                    result = str(tool(parsed_input))

                # Return the result as text content
                return [types.TextContent(type="text", text=str(result))]
            except Exception as e:
                print(f"Error executing tool {name}: {str(e)}")
                traceback.print_exc()
                raise ValueError(f"Error executing tool {name}: {str(e)}")
        else:
            raise ValueError(f"Unknown tool: {name}")

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        # Return both default and dynamically registered tools
        return dynamic_tools

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                    request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/load_openapi", endpoint=load_openapi),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        # Store the register_tool function in the app state for access in endpoints
        starlette_app.state.register_tool = register_tool

        import uvicorn
        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0
