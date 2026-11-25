# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Native MCP server
=================

The MCP (Model Context Protocol) server lets external assistants (e.g. VS Code
Copilot, Claude Desktop) discover and call the tools of the AI Assistant on the
live DataLab session.

It listens on the loopback interface only, requires a token and asks the user to
confirm the actions modifying the workspace, as the AI Assistant panel does.

It needs the optional ``mcp`` dependency (``pip install datalab-platform[mcp]``).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

#: True if the MCP SDK is installed
MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


def mcp_unavailable_message(_: Callable[[str], str] = str) -> str:
    """Return the message explaining how to enable the MCP server.

    Args:
        _: Translation function. Defaults to leaving the message in English.
    """
    return _(
        "The MCP server needs the 'mcp' package: install it with "
        "'pip install datalab-platform[mcp]'."
    )
