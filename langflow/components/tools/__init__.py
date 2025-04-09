"""Tool components for Langflow."""
from typing import Any, Dict, List, Type

from langflow.interface.base import LangflowComponent
from langflow.interface.tools.base import ToolComponent
from langflow.components.tools.mcp_sse_fixed import MCPSse
from langflow.components.tools.mcp_sse_fixed2 import MCPSse2
from langflow.components.tools.openapi_service import OpenApiService

# List of all tool components
TOOLS_COMPONENTS: List[Type[LangflowComponent]] = [
    MCPSse,
    MCPSse2,
    OpenApiService,
]

# Dictionary mapping component names to their classes
tool_type_to_cls_dict: Dict[str, Any] = {
    tool.__name__: tool for tool in TOOLS_COMPONENTS
}

__all__ = [
    "TOOLS_COMPONENTS",
    "tool_type_to_cls_dict",
    "MCPSse",
    "MCPSse2",
    "OpenApiService",
    "ToolComponent",
]