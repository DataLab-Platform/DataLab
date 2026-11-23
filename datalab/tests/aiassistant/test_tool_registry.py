# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for the AI assistant tool registry."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sigima.io import RawImageImportParam

from datalab.aiassistant.tools.builtin import build_default_registry
from datalab.aiassistant.tools.registry import Tool, ToolRegistry, ToolResult
from datalab.aiassistant.tools.validation import validate_arguments


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
    """Registering a tool twice under the same name raises ValueError."""
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
    """Unknown tool lookup raises KeyError mentioning available tool names."""
    reg = _make_registry()
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "echo" in str(excinfo.value)


def test_call_success_returns_payload() -> None:
    """A successful tool call returns its data wrapped in a ToolResult."""
    reg = _make_registry()
    res: ToolResult = reg.call("echo", {"x": 1, "y": "z"}, proxy=None, mainwindow=None)
    assert res.ok is True
    assert res.data == {"echo": {"x": 1, "y": "z"}}
    payload = json.loads(res.to_message_content())
    assert payload == {"ok": True, "result": {"echo": {"x": 1, "y": "z"}}}


def test_call_failure_returns_error() -> None:
    """A failing tool call returns ToolResult with ok=False and an error."""
    reg = _make_registry()
    res = reg.call("boom", {}, proxy=None, mainwindow=None)
    assert res.ok is False
    assert "boom" in (res.error or "")
    payload = json.loads(res.to_message_content())
    assert payload["ok"] is False


def test_list_schemas_format() -> None:
    """list_schemas() returns one schema per registered tool."""
    reg = _make_registry()
    schemas = reg.list_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"echo", "boom"}
    for schema in schemas:
        assert "description" in schema
        assert "parameters" in schema


def test_default_registry_exposes_macro_tool() -> None:
    """By default, the macro execution tool is exposed to the LLM."""
    reg = build_default_registry()
    assert "create_and_run_macro" in {s["name"] for s in reg.list_schemas()}


def test_default_registry_can_hide_macro_tool() -> None:
    """When ``expose_macro_tool=False``, the macro tool is not registered."""
    reg = build_default_registry(expose_macro_tool=False)
    names = {s["name"] for s in reg.list_schemas()}
    assert "create_and_run_macro" not in names
    # Read-only macro inspection tool stays available (no code execution).
    assert "get_macro_console_output" in names


def _make_load_file_context() -> tuple[Mock, SimpleNamespace, Mock]:
    """Return a proxy, a main window and its image panel for ``load_file``."""
    imagepanel = Mock(PANEL_STR_ID="image")
    imagepanel.load_from_files.return_value = []
    mainwindow = SimpleNamespace(imagepanel=imagepanel, signalpanel=Mock())
    return Mock(), mainwindow, imagepanel


def test_load_file_forwards_raw_import_parameters() -> None:
    """``load_file`` converts RAW import parameters into a DataSet."""
    reg = build_default_registry()
    proxy, mainwindow, imagepanel = _make_load_file_context()
    arguments = {
        "filename": "image.raw",
        "panel": "image",
        "import_parameters": {"dtype": "uint8", "width": 4, "height": 3},
    }

    res = reg.call("load_file", arguments, proxy=proxy, mainwindow=mainwindow)

    assert res.ok is True
    param = imagepanel.load_from_files.call_args.kwargs["param"]
    assert isinstance(param, RawImageImportParam)
    assert (param.dtype, param.width, param.height) == ("uint8", 4, 3)


def test_load_file_without_import_parameters_keeps_default_path() -> None:
    """``load_file`` passes no import parameter when none is given."""
    reg = build_default_registry()
    proxy, mainwindow, imagepanel = _make_load_file_context()

    res = reg.call(
        "load_file",
        {"filename": "image.png", "panel": "image"},
        proxy=proxy,
        mainwindow=mainwindow,
    )

    assert res.ok is True
    imagepanel.load_from_files.assert_called_once_with(["image.png"], param=None)


def test_load_file_rejects_unknown_import_parameter() -> None:
    """``load_file`` reports invalid RAW import parameters as tool errors."""
    reg = build_default_registry()
    proxy, mainwindow, imagepanel = _make_load_file_context()

    res = reg.call(
        "load_file",
        {"filename": "a.raw", "panel": "image", "import_parameters": {"nope": 1}},
        proxy=proxy,
        mainwindow=mainwindow,
    )

    assert res.ok is False
    assert "nope" in (res.error or "")
    imagepanel.load_from_files.assert_not_called()


def test_load_file_schema_describes_raw_import_parameters() -> None:
    """The ``load_file`` schema exposes RAW import parameter fields."""
    reg = build_default_registry()
    schema = next(s for s in reg.list_schemas() if s["name"] == "load_file")
    props = schema["parameters"]["properties"]["import_parameters"]["properties"]
    assert set(props) == {item.get_name() for item in RawImageImportParam().get_items()}


def test_macro_tool_documents_autorun() -> None:
    """The model must be told how to propose a macro without running it."""
    reg = build_default_registry()
    tool = reg.get("create_and_run_macro")
    autorun = tool.parameters["properties"]["autorun"]
    assert autorun["type"] == "boolean"
    assert autorun["default"] is True
    assert "without executing" in autorun["description"]
    assert "'autorun' to false" in tool.description


def _make_strict_registry(handler: Mock) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="strict",
            description="Strict schema.",
            parameters={
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "minimum": 1},
                    "kind": {"type": "string", "enum": ["a", "b"]},
                    "uuid": {"type": ["string", "null"]},
                },
                "required": ["n"],
                "additionalProperties": False,
            },
            handler=handler,
        )
    )
    return reg


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({}, "missing required property 'n'"),
        ({"n": "3"}, "expected integer"),
        ({"n": True}, "expected integer"),
        ({"n": 0}, "lower than 1"),
        ({"n": 1, "kind": "c"}, "is not one of"),
        ({"n": 1, "other": 2}, "unexpected property 'other'"),
    ],
)
def test_call_rejects_invalid_arguments(arguments: dict, message: str) -> None:
    """Arguments not matching the schema are rejected before the handler runs."""
    handler = Mock(return_value={})
    res = _make_strict_registry(handler).call("strict", arguments, None, None)
    assert res.ok is False
    assert res.error_kind == "invalid"
    assert message in (res.error or "")
    handler.assert_not_called()


def test_call_accepts_valid_arguments() -> None:
    """Valid arguments, including integral floats and nulls, reach the handler."""
    handler = Mock(return_value={"done": True})
    res = _make_strict_registry(handler).call(
        "strict", {"n": 2.0, "kind": "a", "uuid": None}, None, None
    )
    assert res.ok is True
    handler.assert_called_once()


def test_call_error_kinds() -> None:
    """Unknown tools and handler failures are reported with their error kind."""
    reg = _make_registry()
    assert reg.call("missing", {}, None, None).error_kind == "unknown_tool"
    assert reg.call("boom", {}, None, None).error_kind == "failed"


def test_needs_confirmation() -> None:
    """Only read-only tools may skip the confirmation."""
    reg = _make_registry()
    assert reg.get("echo").needs_confirmation(True) is False
    assert reg.get("echo").needs_confirmation(False) is True
    assert reg.get("boom").needs_confirmation(True) is True


def test_without_returns_filtered_registry() -> None:
    """``without`` leaves out the given tools, without changing the original."""
    reg = _make_registry()
    filtered = reg.without(["boom", "unknown"])
    assert set(filtered.tools) == {"echo"}
    assert set(reg.tools) == {"echo", "boom"}


def test_default_registry_schemas_accept_their_defaults() -> None:
    """The defaults declared in the built-in schemas are valid arguments."""
    for tool in build_default_registry().tools.values():
        properties = tool.parameters.get("properties", {})
        for name, schema in properties.items():
            if "default" in schema:
                arguments = {name: schema["default"]}
                for required in tool.parameters.get("required", []):
                    arguments.setdefault(required, "x")
                assert validate_arguments(tool.parameters, arguments) == [], (
                    tool.name,
                    name,
                )
