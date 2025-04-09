"""Hippycampus - A LangChain-based CLI and MCP server."""

import sys

# Ensure the correct package name
if 'hippycampus.hippycampus' in sys.modules:
    del sys.modules['hippycampus.hippycampus']

__package__ = "hippycampus"

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

# Export commonly used modules
from hippycampus.cli import main
from hippycampus.langchain_util import fixed_create_structured_chat_agent
from hippycampus.openapi_builder import load_tools_from_openapi

__all__ = [
    "main",
    "fixed_create_structured_chat_agent",
    "load_tools_from_openapi",
]