# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
ASGI application of the MCP server (Streamable HTTP transport).

The application publishes the tools of :mod:`datalab.mcpserver.policy` at the
``/mcp`` endpoint. Every request must come from the loopback interface (``Host``
and ``Origin`` headers) and carry the server token (``Authorization: Bearer``).
Refused requests get a 403 response, including those with a missing or invalid
token.
"""

from __future__ import annotations

import contextlib
import json
import secrets
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from datalab import __version__
from datalab.aiassistant.tools.builtin import json_safe

if TYPE_CHECKING:
    from datalab.aiassistant.tools.registry import Tool, ToolResult
    from datalab.mcpserver.executor import McpToolExecutor

MCP_PATH = "/mcp"

INSTRUCTIONS = (
    "Tools of the DataLab session running on this computer (signal and image "
    "processing). Call 'list_objects' to find the UUIDs of the objects, "
    "'list_available_operations' to find the operations and their parameters, "
    "then 'apply_operation' with explicit 'target_uuids'. Analyses store their "
    "results on the objects: read them with 'get_results'. The user confirms the "
    "actions modifying the workspace in DataLab."
)


class ToolCallError(Exception):
    """Error of a tool call, reported to the client as an error result.

    The message starts with the kind of error (e.g. ``refused: ...``).
    """

    def __init__(self, kind: str | None, message: str | None) -> None:
        super().__init__(f"{kind or 'failed'}: {message or 'unknown error'}")
        self.kind = kind


def tool_definition(tool: Tool) -> types.Tool:
    """Return the MCP definition of a tool."""
    return types.Tool(
        name=tool.name,
        description=tool.description,
        inputSchema=tool.parameters,
        annotations=types.ToolAnnotations(
            readOnlyHint=tool.readonly, openWorldHint=False
        ),
    )


def tool_result_content(
    result: ToolResult,
) -> tuple[list[types.TextContent | types.ImageContent], dict[str, Any]]:
    """Return the MCP content of a successful result.

    Args:
        result: Result of the tool.

    Returns:
        The content blocks (JSON text, then images) and the structured content.

    Raises:
        ToolCallError: The tool failed.
    """
    if not result.ok:
        raise ToolCallError(result.error_kind, result.error)
    data = json_safe(result.data)
    if result.images and isinstance(data, dict):
        # The note tells the AI Assistant panel where to find the image
        data.pop("note", None)
    structured = data if isinstance(data, dict) else {"result": data}
    content: list[types.TextContent | types.ImageContent] = [
        types.TextContent(type="text", text=json.dumps(data, indent=2))
    ]
    for mime_type, b64_data in result.images:
        content.append(
            types.ImageContent(type="image", data=b64_data, mimeType=mime_type)
        )
    return content, structured


def _client_name(server: Server) -> str | None:
    """Return the name of the client of the current request, if known."""
    try:
        params = server.request_context.session.client_params
    except LookupError:
        return None
    if params is None:
        return None
    return params.clientInfo.name


def build_mcp_server(executor: McpToolExecutor) -> Server:
    """Return the MCP server publishing the tools of ``executor``."""
    server = Server("DataLab", version=__version__, instructions=INSTRUCTIONS)
    registry = executor.registry

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [tool_definition(tool) for tool in registry.tools.values()]

    # The tool registry validates the arguments, as for the AI Assistant panel
    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]):
        result = await executor.call(name, arguments or {}, _client_name(server))
        return tool_result_content(result)

    return server


def _allowed_hosts(port: int) -> list[str]:
    return [f"127.0.0.1:{port}", f"localhost:{port}"]


def _allowed_origins(port: int) -> list[str]:
    return [f"http://{host}" for host in _allowed_hosts(port)]


class AuthMiddleware:
    """ASGI middleware checking the origin and the token of each request.

    Args:
        app: Wrapped ASGI application.
        port: Port of the server.
        get_token: Callable returning the current token (it may be renewed while
         the server runs).
    """

    def __init__(self, app, port: int, get_token: Callable[[], str]) -> None:
        self.app = app
        self.hosts = set(_allowed_hosts(port))
        self.origins = set(_allowed_origins(port))
        self.get_token = get_token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope["headers"]
        }
        response = None
        origin = headers.get("origin")
        if headers.get("host") not in self.hosts:
            response = JSONResponse({"error": "Invalid Host header"}, status_code=403)
        elif origin is not None and origin not in self.origins:
            response = JSONResponse({"error": "Origin not allowed"}, status_code=403)
        else:
            scheme, _sep, credentials = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not secrets.compare_digest(
                credentials.strip().encode(), self.get_token().encode()
            ):
                # Not 401: MCP clients answer 401 with an OAuth sign-in, which this
                # server does not support, hiding the actual cause to the user
                response = JSONResponse(
                    {
                        "error": "Missing or invalid DataLab MCP token: enter the "
                        "current token shown by DataLab"
                    },
                    status_code=403,
                )
        if response is not None:
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _McpEndpoint:
    """ASGI endpoint passing the requests to the session manager."""

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope, receive, send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


def build_app(executor: McpToolExecutor, port: int, get_token: Callable[[], str]):
    """Return the ASGI application of the MCP server.

    A session manager runs only once: build a new application at each start.

    Args:
        executor: Executor of the tool calls.
        port: Port of the server.
        get_token: Callable returning the current token.
    """
    session_manager = StreamableHTTPSessionManager(
        app=build_mcp_server(executor),
        security_settings=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_allowed_hosts(port),
            allowed_origins=_allowed_origins(port),
        ),
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app) -> AsyncIterator[None]:
        async with session_manager.run():
            yield

    app = Starlette(
        routes=[Route(MCP_PATH, endpoint=_McpEndpoint(session_manager))],
        lifespan=lifespan,
    )
    return AuthMiddleware(app, port, get_token)
