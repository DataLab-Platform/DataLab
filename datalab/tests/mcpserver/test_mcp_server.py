# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Acceptance tests of the MCP server, with the client of the official MCP SDK.

The client runs in a thread of its own while the test processes the Qt events, as
an external assistant does with a live DataLab session.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
from qtpy import QtWidgets as QW

pytest.importorskip("mcp")

# pylint: disable=wrong-import-position,wrong-import-order
import httpx  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

from datalab.aiassistant.tools.builtin import build_default_registry  # noqa: E402
from datalab.aiassistant.widgets.toolconfirmdialog import (  # noqa: E402
    ToolConfirmDialog,
)
from datalab.config import Conf  # noqa: E402
from datalab.mcpserver import auth  # noqa: E402
from datalab.mcpserver.controller import McpServerController  # noqa: E402
from datalab.mcpserver.policy import EXCLUDED_TOOLS  # noqa: E402
from datalab.objectmodel import get_uuid  # noqa: E402
from datalab.tests import datalab_test_app_context  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_server(tmp_path, monkeypatch):
    """Keep the user's token file untouched and use the preferred port."""
    monkeypatch.setattr(auth, "get_token_file_path", lambda: str(tmp_path / "token"))
    monkeypatch.delenv("DATALAB_MCP_PORT", raising=False)


class Confirmations:
    """Answers of the confirmation dialog, recorded instead of shown."""

    def __init__(self) -> None:
        self.answer = True
        self.calls: list[tuple[str, dict, str | None]] = []

    def confirm(self, tool, arguments, _parent=None, source=None) -> bool:
        """Record the confirmation request and return the answer."""
        self.calls.append((tool.name, arguments, source))
        return self.answer


@pytest.fixture(name="confirmations")
def fixture_confirmations(monkeypatch) -> Confirmations:
    """Replace the confirmation dialog."""
    confirmations = Confirmations()
    monkeypatch.setattr(ToolConfirmDialog, "confirm", confirmations.confirm)
    return confirmations


def run_in_client_thread(coro_factory, timeout: float = 60.0) -> Any:
    """Run a coroutine in a client thread, processing the Qt events meanwhile."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = asyncio.run(coro_factory())
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive():
        assert time.monotonic() < deadline, "MCP client timed out"
        QW.QApplication.processEvents()
        time.sleep(0.005)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def with_session(controller: McpServerController, func, token: str | None = None):
    """Run ``func(session)`` with an initialized MCP client session."""
    url, token = controller.url, token or controller.token

    async def run() -> Any:
        headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(headers=headers, timeout=60.0) as http:
            async with streamable_http_client(url, http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await func(session)

    return run_in_client_thread(run)


def call_tools(controller: McpServerController, *calls: tuple[str, dict]) -> list:
    """Call tools in one session and return their results."""

    async def run(session: ClientSession) -> list:
        return [await session.call_tool(name, args) for name, args in calls]

    return with_session(controller, run)


def call_tool(controller: McpServerController, name: str, args: dict | None = None):
    """Call one tool and return its result."""
    return call_tools(controller, (name, args or {}))[0]


def error_text(result) -> str:
    """Return the error message of a failed result."""
    assert result.isError
    return result.content[0].text


@contextmanager
def running_server(win, port: int | None = None) -> Iterator[McpServerController]:
    """Start the MCP server of ``win``, and stop it on exit."""
    controller = McpServerController(win)
    controller.start(port)
    try:
        yield controller
    finally:
        controller.stop()


def add_signal(win, title: str, x: np.ndarray, y: np.ndarray, xunit: str = "") -> str:
    """Add a signal and return its UUID."""
    win.add_signal(title, x, y, xunit=xunit)
    return get_uuid(win.signalpanel.objview.get_current_object())


def test_published_tools_and_list_objects(confirmations: Confirmations) -> None:
    """The client discovers the published tools and tells apart homonyms."""
    with datalab_test_app_context(console=False) as win:
        x = np.linspace(0, 1, 100)
        uuid1 = add_signal(win, "Same title", x, np.sin(x))
        uuid2 = add_signal(win, "Same title", x, np.cos(x))
        with running_server(win) as controller:

            async def run(session: ClientSession):
                tools = (await session.list_tools()).tools
                listing = await session.call_tool("list_objects", {"panel": "signal"})
                return tools, listing

            tools, listing = with_session(controller, run)
        expected = set(build_default_registry().tools) - set(EXCLUDED_TOOLS)
        assert {tool.name for tool in tools} == expected
        annotations = {tool.name: tool.annotations for tool in tools}
        assert annotations["list_objects"].readOnlyHint is True
        assert annotations["apply_operation"].readOnlyHint is False
        objects = listing.structuredContent["objects"]
        assert [obj["uuid"] for obj in objects] == [uuid1, uuid2]
        assert confirmations.calls == []  # Read-only tools are approved


def test_fft_with_explicit_target(confirmations: Confirmations) -> None:
    """An operation runs on the target, after confirmation, without dialog."""
    with datalab_test_app_context(console=False) as win:
        x = np.arange(1024) / 1024.0
        target = add_signal(win, "Sine 64 Hz", x, np.sin(2 * np.pi * 64 * x))
        add_signal(win, "Other", x, np.zeros_like(x))  # Selected object
        with running_server(win) as controller:
            result = call_tool(
                controller, "apply_operation", {"name": "fft", "target_uuids": [target]}
            )
        assert not result.isError, result.content[0].text
        output = result.structuredContent
        assert [obj["uuid"] for obj in output["targets"]] == [target]
        (created,) = output["created_objects"]
        fft = win.signalpanel.objmodel[created["uuid"]]
        assert abs(fft.x[np.argmax(np.abs(fft.y))]) == pytest.approx(64.0)
        ((tool_name, arguments, source),) = confirmations.calls
        assert tool_name == "apply_operation" and source
        assert arguments["target_uuids"] == [target]


@pytest.mark.usefixtures("confirmations")
def test_fwhm_results_and_units() -> None:
    """An analysis reports its new results, read with their units."""
    with datalab_test_app_context(console=False) as win:
        x = np.arange(-20.0, 20.0, 0.02)
        uuid = add_signal(win, "Pulse", x, np.exp(-(x**2) / 8.0), xunit="ns")
        with running_server(win) as controller:
            applied, read = call_tools(
                controller,
                (
                    "apply_operation",
                    {"name": "fwhm", "parameters": {"method": "gauss"}},
                ),
                ("get_results", {"uuids": [uuid]}),
            )
        assert not applied.isError, applied.content[0].text
        assert applied.structuredContent["new_results"]
        assert applied.structuredContent["parameters"]["method"] == "gauss"
        (obj,) = read.structuredContent["objects"]
        (entry,) = obj["results"]
        assert entry["object_units"]["x"] == "ns"
        assert entry["rows"][0]["Δx"] == pytest.approx(4.70964, abs=1e-4)
        assert entry["rows"][0]["roi"] is None


def test_refusal_and_errors(confirmations: Confirmations) -> None:
    """Refused, invalid and unpublished calls are errors with their kind."""
    with datalab_test_app_context(console=False) as win:
        x = np.linspace(0, 1, 100)
        uuid = add_signal(win, "Signal", x, np.sin(x))
        confirmations.answer = False
        with running_server(win) as controller:
            refused, invalid, unknown, wrong_param, bad_uuid = call_tools(
                controller,
                ("apply_operation", {"name": "fft", "target_uuids": [uuid]}),
                ("apply_operation", {"name": "fft", "target_uuids": "x"}),
                ("create_and_run_macro", {"code": "print(1)"}),
                ("list_objects", {"panel": "spectrum"}),
                ("apply_operation", {"name": "fft", "target_uuids": ["nope"]}),
            )
        assert error_text(refused).startswith("refused:")
        assert error_text(invalid).startswith("invalid:")
        assert error_text(unknown).startswith("unknown_tool:")
        assert error_text(wrong_param).startswith("invalid:")
        assert error_text(bad_uuid).startswith("invalid:")
        assert len(win.signalpanel.objmodel) == 1


@pytest.mark.usefixtures("confirmations")
def test_invalid_operation_parameters() -> None:
    """Unknown or invalid parameters are refused before running."""
    with datalab_test_app_context(console=False) as win:
        x = np.linspace(0, 1, 100)
        uuid = add_signal(win, "Signal", x, np.sin(x))
        with running_server(win) as controller:
            unknown, invalid = call_tools(
                controller,
                ("apply_operation", {"name": "fwhm", "parameters": {"wrong": 1}}),
                (
                    "apply_operation",
                    {"name": "fwhm", "parameters": {"method": "nope"}},
                ),
            )
        assert "Unknown parameter 'wrong'" in error_text(unknown)
        assert "Invalid value 'nope'" in error_text(invalid)
        assert not win.signalpanel.objmodel[uuid].metadata.get("Geometry_fwhm_dict")


@pytest.mark.usefixtures("confirmations")
def test_authentication_and_origin() -> None:
    """Requests without the token, or from a foreign origin, are refused."""
    with datalab_test_app_context(console=False) as win:
        with running_server(win) as controller:
            url, token = controller.url, controller.token
            port = url.split(":")[2].split("/")[0]
            body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
            headers = {"Accept": "application/json, text/event-stream"}

            def post(**extra: str) -> int:
                return httpx.post(
                    url, json=body, headers=headers | extra, timeout=10
                ).status_code

            # 403, not 401: a 401 makes VS Code start an unsupported OAuth sign-in
            assert post() == 403
            assert post(Authorization="Bearer wrong") == 403
            bearer = f"Bearer {token}"
            assert post(Authorization=bearer, Origin="https://evil.example") == 403
            assert post(Authorization=bearer, Host=f"evil.example:{port}") == 403
            old_token = token
            new_token = controller.renew_token()
            assert new_token != old_token
            assert post(Authorization=f"Bearer {old_token}") == 403
            listing = with_session(
                controller, lambda s: s.call_tool("get_current_panel", {})
            )
            assert not listing.isError


@pytest.mark.usefixtures("confirmations")
def test_lifecycle_and_ports() -> None:
    """The server takes the next free port, stops, restarts and runs along the
    Web API server."""
    with datalab_test_app_context(console=False) as win:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            busy_port = busy.getsockname()[1]
            with pytest.raises(RuntimeError, match="could not start"):
                McpServerController(win).start(busy_port)
        controller = McpServerController(win)
        url = controller.start()
        assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
        controller.stop()
        assert controller.url is None and controller.token is None
        with pytest.raises(httpx.ConnectError):
            httpx.post(url, json={}, timeout=5)
        # pylint: disable=import-outside-toplevel
        from datalab.webapi import get_webapi_controller

        webapi = get_webapi_controller()
        webapi.set_main_window(win)
        webapi.start()
        try:
            controller.start()
            try:
                result = call_tool(controller, "list_objects")
                assert not result.isError
            finally:
                controller.stop()
        finally:
            webapi.stop()
            webapi.release_main_window(win)


def test_persistent_token() -> None:
    """The persistent token is kept between starts; the session token is not."""
    first = auth.get_token(persistent=True)
    assert auth.get_token(persistent=True) == first
    renewed = auth.renew_token(persistent=True)
    assert auth.get_token(persistent=True) == renewed != first
    session = auth.get_token(persistent=False)
    assert session not in (first, renewed)
    assert auth.get_token(persistent=False) != session


@pytest.mark.usefixtures("confirmations")
def test_connection_dialog() -> None:
    """The dialog starts and stops the server and gives the VS Code
    configuration, without the token; closing DataLab stops the server."""
    # pylint: disable=protected-access
    with datalab_test_app_context(console=False) as win:
        actions = win.mcp_actions
        assert actions.action.isEnabled()
        # The AI Assistant panel opens the dialog too
        win.aiassistantpanel.mcp_button.click()
        dialog = actions._dialog
        assert dialog.isVisible()
        dialog._toggle_server()
        controller = actions.controller
        assert controller.is_running
        config = json.loads(dialog._config_edit.toPlainText())
        server = config["servers"]["datalab"]
        assert server["type"] == "http" and server["url"] == controller.url
        assert controller.token not in dialog._config_edit.toPlainText()
        assert dialog._token_edit.text() == controller.token
        result = call_tool(controller, "list_objects")
        assert not result.isError
        dialog._toggle_server()
        assert not controller.is_running
        assert dialog._config_edit.toPlainText() == ""
        dialog._toggle_server()
        url = controller.url
    assert not controller.is_running
    with pytest.raises(httpx.ConnectError):
        httpx.post(url, json={}, timeout=5)


def test_auto_approve(confirmations: Confirmations) -> None:
    """Read-only tools never need confirmation; the auto-approve option of the
    dialog runs the other tools without confirmation."""
    # pylint: disable=protected-access
    readonly_option = Conf.ai_auto_approve_readonly.get(True)
    Conf.ai_auto_approve_readonly.set(False)
    confirmations.answer = False
    try:
        with datalab_test_app_context(console=False) as win:
            x = np.linspace(0, 1, 100)
            uuid = add_signal(win, "Signal", x, np.sin(x))
            actions = win.mcp_actions
            actions.show_dialog()
            box = actions._dialog._auto_approve_box
            assert not box.isChecked()
            with running_server(win) as controller:
                listing, refused = call_tools(
                    controller,
                    ("list_objects", {"panel": "signal"}),
                    ("apply_operation", {"name": "fft", "target_uuids": [uuid]}),
                )
                box.setChecked(True)
                assert Conf.mcp_auto_approve.get(False)
                applied = call_tool(
                    controller,
                    "apply_operation",
                    {"name": "fft", "target_uuids": [uuid]},
                )
            assert not listing.isError
            assert error_text(refused).startswith("refused:")
            assert not applied.isError, applied.content[0].text
            assert [call[0] for call in confirmations.calls] == ["apply_operation"]
            assert len(win.signalpanel.objmodel) == 2
    finally:
        Conf.ai_auto_approve_readonly.set(readonly_option)
        Conf.mcp_auto_approve.set(False)
