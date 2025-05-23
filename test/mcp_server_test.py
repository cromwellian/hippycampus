import pytest
import asyncio
import base64
import hashlib
import json
import os
import tempfile
from unittest.mock import patch, AsyncMock, MagicMock, call, mock_open
from typing import Any, Dict, List, Optional, Union, Type # Added Type

import httpx # For mocking AsyncClient
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

# Modules to test
from hippycampus import mcp_server 
from hippycampus.mcp_server import (
    fetch_website,
    encode_as_base64,
    decode_base64,
    compute_git_commit_sha,
    fetch_documentation_for_tool,
    load_openapi, 
    # fetch_tool, # Original name, now fetch_tool_from_registry
    register_tool, # Original name, now part of ToolRegistry
    ToolRegistry, 
    fetch_tool_from_registry, 
)
from hippycampus.tool_auth.authentication import AbstractAuth 
from hippycampus.openapi_builder import OpenAPIToolBuilder 

# --- Fixtures ---

@pytest.fixture
def temp_openapi_file_content() -> str:
    return """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
servers:
  - url: http://localhost:8080/api
paths:
  /test_endpoint:
    get:
      operationId: testOperation
      summary: A simple test operation
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
"""

@pytest.fixture
def temp_openapi_file(temp_openapi_file_content: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
        tmp_file.write(temp_openapi_file_content)
        path = tmp_file.name
    yield path
    os.remove(path)


class MockToolArgs(BaseModel):
    arg1: str = Field(description="Argument 1")
    arg2: Optional[int] = Field(default=None, description="Argument 2")

class MockToolWithArgs(BaseTool):
    name: str = "mock_tool_with_args"
    description: str = "A mock tool that requires arguments"
    args_schema: Type[BaseModel] = MockToolArgs # type: ignore # Pydantic v1/v2 compatibility for args_schema

    def _run(self, arg1: str, arg2: Optional[int] = None) -> Any:
        return f"Ran with {arg1} and {arg2}"

    async def _arun(self, arg1: str, arg2: Optional[int] = None) -> Any:
        return f"Async ran with {arg1} and {arg2}"

class MockToolNoArgs(BaseTool):
    name: str = "mock_tool_no_args"
    description: str = "A mock tool with no arguments"
    # No args_schema needed for tools that take no arguments (or string input for BaseTool)

    def _run(self) -> Any: # type: ignore
        return "Ran no_args tool"
    async def _arun(self) -> Any: # type: ignore
        return "Async ran no_args tool"


@pytest.fixture
def mock_tool_with_args_instance():
    return MockToolWithArgs()

@pytest.fixture
def mock_tool_no_args_instance():
    return MockToolNoArgs()


@pytest.fixture
def tool_registry_instance() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def test_app(tool_registry_instance: ToolRegistry) -> Starlette:
    async def startup_event():
        app.state.tool_registry = tool_registry_instance
        # For testing original global functions if needed during transition
        # This assumes mcp_server might still have global fallbacks or that tests might target older states.
        # If fully refactored, these lines might not be necessary.
        if hasattr(mcp_server, 'langchain_tools'):
            mcp_server.langchain_tools = tool_registry_instance.langchain_tools
        if hasattr(mcp_server, 'external_docs_for_tools'):
            mcp_server.external_docs_for_tools = tool_registry_instance.external_docs_for_tools
        if hasattr(mcp_server, 'external_metadata_for_tools'):
            mcp_server.external_metadata_for_tools = tool_registry_instance.external_metadata_for_tools


    async def call_tool_for_testing(request: Request) -> JSONResponse:
        tool_name = request.path_params["tool_name"]
        try:
            payload = await request.json()
            arguments = payload.get("arguments", {})
        except json.JSONDecodeError:
            return JSONResponse({"error_type": "InvalidJSON", "message": "Invalid JSON payload"}, status_code=400)
        try:
            result = await fetch_tool_from_registry(tool_name, arguments, app.state.tool_registry)
            return JSONResponse({"result": result})
        except ValueError as e: 
            status_code = 404 if "not found" in str(e).lower() else 400
            return JSONResponse({"error_type": "ToolError", "message": str(e)}, status_code=status_code)
        except Exception as e:
            return JSONResponse({"error_type": "ToolExecutionError", "message": f"Tool execution failed: {str(e)}"}, status_code=500)

    app = Starlette(
        routes=[
            Route("/load_openapi", load_openapi, methods=["POST"]),
            Route("/call_tool/{tool_name}", call_tool_for_testing, methods=["POST"])
        ],
        on_startup=[startup_event]
    )
    return app


@pytest.fixture
def client(test_app: Starlette) -> TestClient:
    return TestClient(test_app)


# --- Helper Function Tests ---
@pytest.mark.asyncio
class TestHelperFunctions:
    @patch('httpx.AsyncClient.get', new_callable=AsyncMock)
    async def test_fetch_website_success(self, mock_get: AsyncMock):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "<html><body>Hello World</body></html>"
        mock_get.return_value = mock_response

        result = await fetch_website("http://example.com")
        assert result == "<html><body>Hello World</body></html>"
        mock_get.assert_called_once_with("http://example.com", follow_redirects=True)

    @patch('httpx.AsyncClient.get', new_callable=AsyncMock)
    async def test_fetch_website_http_error(self, mock_get: AsyncMock):
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.request = MagicMock(spec=httpx.Request) 
        mock_response.request.url = "http://example.com/notfound" # type: ignore
        mock_get.return_value = mock_response
        mock_response.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("Not Found", request=mock_response.request, response=mock_response))

        result = await fetch_website("http://example.com/notfound")
        assert "Error fetching website: Client error '404 Not Found' for url 'http://example.com/notfound'." in result
        assert "Response: Not Found" in result


    @patch('httpx.AsyncClient.get', new_callable=AsyncMock)
    async def test_fetch_website_request_error(self, mock_get: AsyncMock):
        request_obj = MagicMock(spec=httpx.Request)
        request_obj.url = "http://example.com/failed" # type: ignore
        mock_get.side_effect = httpx.RequestError("Connection failed", request=request_obj)
        
        result = await fetch_website("http://example.com/failed")
        assert "Request error for url 'http://example.com/failed': Connection failed" in result

    def test_encode_as_base64(self):
        assert encode_as_base64("hello") == base64.b64encode(b"hello").decode()
        assert encode_as_base64(b"world") == base64.b64encode(b"world").decode()

    def test_decode_base64(self):
        encoded_str = base64.b64encode(b"hello").decode()
        assert decode_base64(encoded_str) == "hello"
        
        with pytest.raises(ValueError, match="Invalid base64 string provided."): # Match updated error message
            decode_base64("not base64")

    def test_compute_git_commit_sha(self):
        content_str = "hello\n"
        expected_sha_str = hashlib.sha1(content_str.encode('utf-8')).hexdigest()
        assert compute_git_commit_sha(content_str) == expected_sha_str

        content_bytes = b"world\n"
        expected_sha_bytes = hashlib.sha1(content_bytes).hexdigest()
        assert compute_git_commit_sha(content_bytes) == expected_sha_bytes

    def test_fetch_documentation_for_tool(self, mock_tool_with_args_instance: MockToolWithArgs, tool_registry_instance: ToolRegistry):
        tool_registry_instance.register_tool("mock_tool_with_args", mock_tool_with_args_instance, docs_url="http://docs.example.com", metadata={"version": "1.0"})
        
        doc_info = fetch_documentation_for_tool(
            "mock_tool_with_args", 
            tool_registry_instance.langchain_tools, 
            tool_registry_instance.external_docs_for_tools, 
            tool_registry_instance.external_metadata_for_tools
        )
        assert doc_info["name"] == "mock_tool_with_args"
        assert doc_info["description"] == "A mock tool that requires arguments"
        assert doc_info["external_doc_url"] == "http://docs.example.com"
        assert doc_info["external_metadata"]["version"] == "1.0" # type: ignore
        assert doc_info["args_schema"] is not None 

        no_extra_tool = MockToolNoArgs(name="mock_tool_no_extras_desc", description="No extras")
        tool_registry_instance.register_tool("mock_tool_no_extras_desc", no_extra_tool)
        doc_info_no_extras = fetch_documentation_for_tool(
            "mock_tool_no_extras_desc", 
            tool_registry_instance.langchain_tools,
            tool_registry_instance.external_docs_for_tools, 
            tool_registry_instance.external_metadata_for_tools
        )
        assert doc_info_no_extras["name"] == "mock_tool_no_extras_desc"
        assert "external_doc_url" not in doc_info_no_extras
        assert "external_metadata" not in doc_info_no_extras
        assert doc_info_no_extras["args_schema"] is None 
        
        with pytest.raises(ValueError, match="Tool 'not_found_tool' not found for documentation retrieval."): # Match updated error
            fetch_documentation_for_tool("not_found_tool", tool_registry_instance.langchain_tools, {}, {})


# --- Endpoint Logic Tests (`load_openapi`) ---
@pytest.mark.asyncio
class TestLoadOpenAPI:
    @patch('hippycampus.mcp_server.OpenAPIToolBuilder') 
    @patch('builtins.open', new_callable=mock_open) 
    async def test_load_openapi_local_file_success(
        self, mock_file_open: MagicMock, 
        mock_builder_class: MagicMock, client: TestClient, temp_openapi_file_content: str,
        tool_registry_instance: ToolRegistry 
    ):
        mock_file_open.return_value.read.return_value = temp_openapi_file_content
        
        mock_builder_instance = MagicMock()
        mock_tool1 = MockToolNoArgs(name="tool1", description="Tool 1")
        mock_builder_instance.build_tools.return_value = [mock_tool1]
        mock_builder_class.return_value = mock_builder_instance
        
        # No need to mock tempfile if OpenAPIToolBuilder takes content string

        response = client.post("/load_openapi", params={"url": "file:///some/local/spec.yaml"}) # Use client.post for sync test client

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["message"] == "OpenAPI spec loaded and tools registered successfully."
        assert len(response_json["tools_loaded"]) == 1
        assert response_json["tools_loaded"][0]["name"] == "tool1"

        mock_file_open.assert_called_once_with("/some/local/spec.yaml", "r", encoding='utf-8')
        mock_builder_class.assert_called_once_with(spec=temp_openapi_file_content)
        mock_builder_instance.build_tools.assert_called_once()
        
        assert tool_registry_instance.get_tool("tool1") is mock_tool1


    @patch('hippycampus.mcp_server.fetch_website', new_callable=AsyncMock)
    @patch('hippycampus.mcp_server.OpenAPIToolBuilder')
    async def test_load_openapi_url_success(
        self, mock_builder_class: MagicMock, 
        mock_fetch_website: AsyncMock, client: TestClient, temp_openapi_file_content: str,
        tool_registry_instance: ToolRegistry
    ):
        mock_fetch_website.return_value = temp_openapi_file_content # fetch_website returns string
        
        mock_builder_instance = MagicMock()
        mock_tool2 = MockToolNoArgs(name="tool2", description="Tool 2 from URL")
        mock_builder_instance.build_tools.return_value = [mock_tool2]
        mock_builder_class.return_value = mock_builder_instance

        response = client.post("/load_openapi", params={"url": "http://example.com/spec.yaml"})

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["tools_loaded"][0]["name"] == "tool2"

        mock_fetch_website.assert_called_once_with("http://example.com/spec.yaml")
        mock_builder_class.assert_called_once_with(spec=temp_openapi_file_content) 
        assert tool_registry_instance.get_tool("tool2") is mock_tool2


    def test_load_openapi_missing_url_param(self, client: TestClient): # No async for TestClient direct calls
        response = client.post("/load_openapi") 
        assert response.status_code == 400 
        assert response.json()["message"] == "URL query parameter is required." # Match new error structure

    @patch('builtins.open', new_callable=mock_open)
    def test_load_openapi_local_file_not_found(self, mock_file_open: MagicMock, client: TestClient):
        mock_file_open.side_effect = FileNotFoundError("File not found")
        response = client.post("/load_openapi", params={"url": "file:///nonexistent.yaml"})
        assert response.status_code == 400
        assert "File not found" in response.json()["message"] # Match new error structure

    @patch('hippycampus.mcp_server.fetch_website', new_callable=AsyncMock) # Needs to be async for the endpoint
    async def test_load_openapi_url_fetch_error(self, mock_fetch_website: AsyncMock, client: TestClient): # TestClient can call async endpoint
        mock_fetch_website.return_value = "Error fetching website: Client error '404 Not Found'" 
        # TestClient calls are sync, but endpoint is async. TestClient handles the loop.
        response = client.post("/load_openapi", params={"url": "http://example.com/error_spec.yaml"})
        assert response.status_code == 400
        assert "Failed to fetch OpenAPI spec from URL" in response.json()["message"]
        assert "Client error '404 Not Found'" in response.json()["message"]

    @patch('hippycampus.mcp_server.OpenAPIToolBuilder')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_openapi_invalid_spec_content(
        self, mock_file_open: MagicMock, mock_builder_class: MagicMock, client: TestClient
    ):
        mock_file_open.return_value.read.return_value = "invalid: yaml: content"
        mock_builder_instance = mock_builder_class.return_value
        mock_builder_instance.build_tools.side_effect = ValueError("Invalid OpenAPI content")
        
        response = client.post("/load_openapi", params={"url": "file:///invalid_spec.yaml"})
        assert response.status_code == 400
        assert "Failed to parse OpenAPI spec or build tools" in response.json()["message"]
        assert "Invalid OpenAPI content" in response.json()["message"]


# --- Core Dispatch Logic Tests (`fetch_tool_from_registry`) ---
@pytest.mark.asyncio
class TestFetchTool:
    @patch('hippycampus.mcp_server.fetch_website', new_callable=AsyncMock)
    async def test_fetch_tool_default_fetch(self, mock_fetch_website: AsyncMock, tool_registry_instance: ToolRegistry):
        mock_fetch_website.return_value = "Website content"
        result = await fetch_tool_from_registry("fetch", {"url": "http://example.com"}, tool_registry_instance)
        assert result == "Website content"
        mock_fetch_website.assert_called_once_with("http://example.com")

    async def test_fetch_tool_default_encode_base64(self, tool_registry_instance: ToolRegistry):
        result = await fetch_tool_from_registry("encode_as_base64", {"data": "hello"}, tool_registry_instance)
        assert result == base64.b64encode(b"hello").decode()

    async def test_fetch_tool_default_decode_base64(self, tool_registry_instance: ToolRegistry):
        encoded = base64.b64encode(b"hello").decode()
        result = await fetch_tool_from_registry("decode_base64", {"data": encoded}, tool_registry_instance)
        assert result == "hello"

    async def test_fetch_tool_default_compute_sha(self, tool_registry_instance: ToolRegistry):
        result = await fetch_tool_from_registry("compute_git_commit_sha", {"data": "content"}, tool_registry_instance)
        assert result == hashlib.sha1(b"content").hexdigest()

    async def test_fetch_tool_default_fetch_documentation(
        self, mock_tool_with_args_instance: MockToolWithArgs, tool_registry_instance: ToolRegistry
    ):
        tool_registry_instance.register_tool("documented_tool", mock_tool_with_args_instance, docs_url="http://docs.tool.com")
        result = await fetch_tool_from_registry("fetch_documentation_for_tool", {"tool_name": "documented_tool"}, tool_registry_instance)
        assert result["name"] == "documented_tool"
        assert result["external_doc_url"] == "http://docs.tool.com"

    async def test_fetch_tool_dynamic_tool_ainvoke( # Test ainvoke path
        self, mock_tool_with_args_instance: MockToolWithArgs, tool_registry_instance: ToolRegistry
    ):
        # Ensure the mock tool has an ainvoke if that's what we want to test
        mock_tool_with_args_instance.ainvoke = AsyncMock(return_value="Async success via ainvoke") # type: ignore
        mock_tool_with_args_instance._arun = AsyncMock(return_value="Should not be called if ainvoke exists") # type: ignore
        tool_registry_instance.register_tool("dynamic_async_tool", mock_tool_with_args_instance)
        
        result = await fetch_tool_from_registry("dynamic_async_tool", {"arg1": "val1", "arg2": 2}, tool_registry_instance)
        assert result == "Async success via ainvoke"
        mock_tool_with_args_instance.ainvoke.assert_called_once_with(input={"arg1": "val1", "arg2": 2})
        mock_tool_with_args_instance._arun.assert_not_called()


    async def test_fetch_tool_dynamic_tool_invoke_sync_fallback( # Test invoke path
        self, mock_tool_no_args_instance: MockToolNoArgs, tool_registry_instance: ToolRegistry
    ):
        mock_tool_no_args_instance.invoke = MagicMock(return_value="Sync success via invoke") # type: ignore
        mock_tool_no_args_instance._run = MagicMock(return_value="Should not be called if invoke exists") # type: ignore
        tool_registry_instance.register_tool("dynamic_sync_tool", mock_tool_no_args_instance)

        result = await fetch_tool_from_registry("dynamic_sync_tool", {}, tool_registry_instance)
        assert result == "Sync success via invoke"
        mock_tool_no_args_instance.invoke.assert_called_once_with(input={})
        mock_tool_no_args_instance._run.assert_not_called()


    async def test_fetch_tool_unknown_tool(self, tool_registry_instance: ToolRegistry):
        with pytest.raises(ValueError, match="Tool 'unknown_tool_name' not found in registry."): # Match updated error
            await fetch_tool_from_registry("unknown_tool_name", {}, tool_registry_instance)

    async def test_fetch_tool_execution_error_ainvoke(
        self, mock_tool_with_args_instance: MockToolWithArgs, tool_registry_instance: ToolRegistry
    ):
        mock_tool_with_args_instance.ainvoke = AsyncMock(side_effect=Exception("Tool failed via ainvoke!")) # type: ignore
        tool_registry_instance.register_tool("failing_tool_ainvoke", mock_tool_with_args_instance)
        
        with pytest.raises(Exception, match="Tool failed via ainvoke!"):
            await fetch_tool_from_registry("failing_tool_ainvoke", {"arg1": "test"}, tool_registry_instance)


# --- `ToolRegistry.register_tool` Function Tests ---
class TestRegisterTool: # Class name changed to reflect testing ToolRegistry method
    def test_register_new_tool(self, tool_registry_instance: ToolRegistry, mock_tool_no_args_instance: MockToolNoArgs):
        tool_registry_instance.register_tool("new_tool", mock_tool_no_args_instance, "http://docs.new.com", {"meta": "data"})
        assert tool_registry_instance.get_tool("new_tool") is mock_tool_no_args_instance
        assert tool_registry_instance.get_external_doc_url("new_tool") == "http://docs.new.com"
        assert tool_registry_instance.get_external_metadata("new_tool") == {"meta": "data"}

    def test_replace_existing_tool(self, tool_registry_instance: ToolRegistry): # Removed mock_tool_no_args to avoid unused
        tool1 = MockToolNoArgs(name="tool_to_replace_1")
        tool2 = MockToolNoArgs(name="tool_to_replace_2") 
        
        tool_registry_instance.register_tool("replaceable", tool1, "http://docs1.com")
        assert tool_registry_instance.get_tool("replaceable") is tool1
        assert tool_registry_instance.get_external_doc_url("replaceable") == "http://docs1.com"

        tool_registry_instance.register_tool("replaceable", tool2, "http://docs2.com", {"v": 2})
        assert tool_registry_instance.get_tool("replaceable") is tool2 
        assert tool_registry_instance.get_external_doc_url("replaceable") == "http://docs2.com"
        assert tool_registry_instance.get_external_metadata("replaceable") == {"v": 2}


# --- Integration Tests (Using TestClient) ---
@pytest.mark.asyncio # Mark class for async tests if any test methods are async (client calls often are)
class TestIntegration:
    def test_load_openapi_integration_local_file(self, client: TestClient, temp_openapi_file: str, tool_registry_instance: ToolRegistry): # Not async if client calls are sync
        file_url = f"file://{temp_openapi_file}" 
        response = client.post(f"/load_openapi?url={file_url}") # Use ? for query params with TestClient

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["message"] == "OpenAPI spec loaded and tools registered successfully."
        assert len(response_json["tools_loaded"]) == 1 
        
        loaded_tool_name = response_json["tools_loaded"][0]["name"] 
        assert tool_registry_instance.get_tool(loaded_tool_name) is not None
        assert tool_registry_instance.get_tool(loaded_tool_name).name == loaded_tool_name # type: ignore
        assert tool_registry_instance.get_tool(loaded_tool_name).description == "A simple test operation" # type: ignore

    @patch('hippycampus.mcp_server.fetch_website', new_callable=AsyncMock)
    def test_load_openapi_integration_url( # Not async if client calls are sync
        self, mock_fetch_website: AsyncMock, client: TestClient, 
        temp_openapi_file_content: str, tool_registry_instance: ToolRegistry
    ):
        mock_fetch_website.return_value = temp_openapi_file_content # fetch_website is async, but client.post is sync
        
        response = client.post("/load_openapi", params={"url": "http://mock.com/spec.yaml"})
        assert response.status_code == 200
        response_json = response.json()
        assert len(response_json["tools_loaded"]) == 1
        loaded_tool_name = response_json["tools_loaded"][0]["name"]
        assert tool_registry_instance.get_tool(loaded_tool_name) is not None
        mock_fetch_website.assert_called_once_with("http://mock.com/spec.yaml")

    def test_call_tool_integration(self, client: TestClient, tool_registry_instance: ToolRegistry, mock_tool_with_args_instance: MockToolWithArgs):
        # Mock the arun method to be an AsyncMock for the test
        mock_tool_with_args_instance.ainvoke = AsyncMock(return_value="Called via endpoint") # type: ignore
        tool_registry_instance.register_tool("callable_tool", mock_tool_with_args_instance)

        response = client.post( # TestClient.post is synchronous
            "/call_tool/callable_tool", 
            json={"arguments": {"arg1": "hello", "arg2": 123}}
        )
        assert response.status_code == 200
        assert response.json()["result"] == "Called via endpoint"
        # The assertion on ainvoke needs to happen on the mock attached to the instance in the registry
        registered_tool_mock = tool_registry_instance.get_tool("callable_tool")
        assert isinstance(registered_tool_mock, MockToolWithArgs) # Type check for safety
        registered_tool_mock.ainvoke.assert_called_once_with(input={"arg1": "hello", "arg2": 123}) # type: ignore


    def test_call_tool_integration_unknown_tool(self, client: TestClient):
        response = client.post("/call_tool/nonexistent_tool", json={"arguments": {}})
        assert response.status_code == 404
        assert "Tool 'nonexistent_tool' not found" in response.json()["message"]


if __name__ == "__main__":
    pytest.main()
