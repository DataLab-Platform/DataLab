# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AI assistant tool registry."""

from __future__ import annotations

import json

import pytest

from datalab.aiassistant.tools.registry import Tool, ToolRegistry, ToolResult


def _ok_tool(_proxy, _mw, **kwargs):  # noqa: ARG001
    return {"echo": kwargs}


def _failing_tool(_proxy, _mw):  # noqa: ARG001
    raise RuntimeError("boom")


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="Echo arguments back.",
            parameters={"type": "object", "properties": {}},
            handler=_ok_tool,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="boom",
            description="Always fails.",
            parameters={"type": "object", "properties": {}},
            handler=_failing_tool,
        )
    )
    return reg


def test_register_duplicate_raises() -> None:
    reg = _make_registry()
    with pytest.raises(ValueError):
        reg.register(
            Tool(
                name="echo",
                description="dup",
                parameters={"type": "object"},
                handler=_ok_tool,
            )
        )


def test_get_unknown_tool_lists_available() -> None:
    reg = _make_registry()
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "echo" in str(excinfo.value)


def test_call_success_returns_payload() -> None:
    reg = _make_registry()
    res: ToolResult = reg.call("echo", {"x": 1, "y": "z"}, proxy=None, mainwindow=None)
    assert res.ok is True
    assert res.data == {"echo": {"x": 1, "y": "z"}}
    payload = json.loads(res.to_message_content())
    assert payload == {"ok": True, "result": {"echo": {"x": 1, "y": "z"}}}


def test_call_failure_returns_error() -> None:
    reg = _make_registry()
    res = reg.call("boom", {}, proxy=None, mainwindow=None)
    assert res.ok is False
    assert "boom" in (res.error or "")
    payload = json.loads(res.to_message_content())
    assert payload["ok"] is False


def test_list_schemas_format() -> None:
    reg = _make_registry()
    schemas = reg.list_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"echo", "boom"}
    for schema in schemas:
        assert "description" in schema
        assert "parameters" in schema
