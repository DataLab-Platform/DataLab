# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Exposure policy of the MCP server: which tools external assistants may call.
"""

from __future__ import annotations

from datalab.aiassistant.tools.builtin import build_default_registry
from datalab.aiassistant.tools.registry import ToolRegistry

#: Tools of the AI Assistant that are not published to external assistants:
#: macros run arbitrary code, files are read from any path, and plugin actions
#: open dialogs and run code that DataLab does not control.
EXCLUDED_TOOLS = (
    "create_and_run_macro",
    "get_macro_console_output",
    "load_file",
    "list_plugin_actions",
    "trigger_plugin_action",
)


def build_mcp_registry() -> ToolRegistry:
    """Return the registry of the tools published by the MCP server."""
    return build_default_registry(expose_macro_tool=False).without(EXCLUDED_TOOLS)
