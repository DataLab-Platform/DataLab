# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Built-in tools exposed to the LLM.

This module groups all tools that are registered by default in
:func:`build_default_registry`. They cover four families:

- **Inspection** (read-only, may be auto-approved): list and describe objects,
  enumerate available operations.
- **Creation** (confirmation required): synthetic signals/images, file loading.
- **Processing** (confirmation required): :meth:`LocalProxy.calc` calls.
- **Macros** (confirmation required): create + run a Python macro.
"""

# pylint: disable=unused-argument

from __future__ import annotations

import ast
import base64
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.aiassistant.providers.base import ChatMessage
from datalab.aiassistant.tools.registry import Tool, ToolRegistry, ToolResult
from datalab.gui.actionhandler import ActionCategory

if TYPE_CHECKING:
    from datalab.control.proxy import LocalProxy
    from datalab.gui.main import DLMainWindow


# =============================================================================
# Helpers
# =============================================================================


def _resolve_panel(mainwindow: DLMainWindow, panel: str | None):
    """Return the panel widget for ``panel`` (signal/image), or current panel."""
    if panel is None:
        panel = mainwindow.get_current_panel()
    panel = panel.lower()
    if panel == "signal":
        return mainwindow.signalpanel
    if panel == "image":
        return mainwindow.imagepanel
    raise ValueError(f"Invalid panel name {panel!r} (expected 'signal' or 'image').")


def _safe_metadata(metadata: dict | None) -> dict:
    """Return a JSON-serializable subset of an object's metadata."""
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = [str(v) for v in value]
        elif isinstance(value, np.ndarray):
            out[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            out[key] = str(value)
    return out


def _describe_object(obj) -> dict[str, Any]:
    """Return a JSON-serializable description of a signal or image object."""
    info: dict[str, Any] = {
        "uuid": getattr(obj, "uuid", None),
        "title": getattr(obj, "title", None),
        "type": type(obj).__name__,
    }
    if hasattr(obj, "y") and hasattr(obj, "x"):
        info["size"] = int(np.asarray(obj.x).size)
        info["x_min"] = float(np.min(obj.x))
        info["x_max"] = float(np.max(obj.x))
        info["y_min"] = float(np.min(obj.y))
        info["y_max"] = float(np.max(obj.y))
        info["xunit"] = getattr(obj, "xunit", "") or ""
        info["yunit"] = getattr(obj, "yunit", "") or ""
        info["xlabel"] = getattr(obj, "xlabel", "") or ""
        info["ylabel"] = getattr(obj, "ylabel", "") or ""
    elif hasattr(obj, "data"):
        data = np.asarray(obj.data)
        info["shape"] = list(data.shape)
        info["dtype"] = str(data.dtype)
        info["min"] = float(np.min(data))
        info["max"] = float(np.max(data))
        info["mean"] = float(np.mean(data))
    info["roi_count"] = len(obj.roi) if getattr(obj, "roi", None) else 0
    info["metadata"] = _safe_metadata(getattr(obj, "metadata", None))
    return info


# =============================================================================
# Inspection tools (read-only)
# =============================================================================


def _tool_list_objects(
    proxy: LocalProxy, mainwindow: DLMainWindow, panel: str | None = None
) -> dict[str, Any]:
    titles = proxy.get_object_titles(panel)
    uuids = proxy.get_object_uuids(panel)
    return {
        "panel": panel or proxy.get_current_panel(),
        "objects": [{"uuid": u, "title": t} for u, t in zip(uuids, titles)],
    }


def _tool_get_current_panel(
    proxy: LocalProxy, mainwindow: DLMainWindow
) -> dict[str, Any]:
    return {"panel": proxy.get_current_panel()}


def _tool_get_object_info(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    nb_id_title: str | int | None = None,
    panel: str | None = None,
) -> dict[str, Any]:
    obj = proxy.get_object(nb_id_title, panel)
    return _describe_object(obj)


def _tool_list_available_operations(
    proxy: LocalProxy, mainwindow: DLMainWindow, panel: str | None = None
) -> dict[str, Any]:
    panel_widget = _resolve_panel(mainwindow, panel)
    processor = panel_widget.processor
    ops: list[dict[str, Any]] = []
    for feature in processor.computing_registry.values():
        param_fields: list[dict[str, Any]] = []
        if feature.paramclass is not None:
            try:
                instance = feature.paramclass()
                for item in instance.get_items():
                    param_fields.append(
                        {
                            "name": item.get_name(),
                            "type": type(item).__name__,
                            "label": item.get_prop_value("display", instance, "label"),
                            "default": item.get_value(instance),
                        }
                    )
            except Exception:  # pylint: disable=broad-except
                # Some param classes require positional args; skip introspection.
                param_fields = [{"name": "?", "info": "introspection failed"}]
        ops.append(
            {
                "name": feature.name,
                "title": feature.title,
                "pattern": feature.pattern,
                "param_class": (
                    feature.paramclass.__name__ if feature.paramclass else None
                ),
                "parameters": param_fields,
            }
        )
    return {"panel": panel_widget.PANEL_STR_ID, "operations": ops}


# =============================================================================
# Plugin discovery and triggering
# =============================================================================


def _iter_plugin_actions(
    menu: QW.QMenu, path: list[str]
) -> list[tuple[list[str], QW.QAction]]:
    """Recursively collect (path, action) pairs from a plugin QMenu.

    Submenus contribute to the path; leaf QActions (without submenu) yield
    one ``(path, action)`` entry.
    """
    out: list[tuple[list[str], QW.QAction]] = []
    for action in menu.actions():
        if action.isSeparator():
            continue
        submenu = action.menu()
        if submenu is not None:
            out.extend(
                _iter_plugin_actions(submenu, path + [action.text() or submenu.title()])
            )
        else:
            title = action.text()
            if not title:
                continue
            out.append((path + [title], action))
    return out


def _collect_plugin_actions_for_panel(
    panel_widget,
) -> list[tuple[list[str], QW.QAction]]:
    """Return all plugin actions registered on a panel as (path, action) pairs.

    The first path element is the plugin top-level menu title (typically the
    plugin name).
    """
    sah = panel_widget.acthandler
    entries = sah.feature_actions.get(ActionCategory.PLUGINS, [])
    out: list[tuple[list[str], QW.QAction]] = []
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, QW.QMenu):
            out.extend(_iter_plugin_actions(entry, [entry.title()]))
        elif isinstance(entry, QW.QAction):
            submenu = entry.menu()
            title = entry.text()
            if submenu is not None and title:
                out.extend(_iter_plugin_actions(submenu, [title]))
            elif title:
                out.append(([title], entry))
    return out


def _normalise(text: str) -> str:
    """Strip Qt mnemonic markers, ellipses and surrounding whitespace."""
    return (text or "").replace("&", "").replace("…", "").replace("...", "").strip()


def _action_matches(path: list[str], query: str) -> bool:
    """Match a plugin action path against a user-provided query string.

    Accepts either the full ``"Plugin/Submenu/Action"`` path or just the
    leaf action title. Matching is case-insensitive and tolerant of Qt
    mnemonics ('&') and trailing ellipses.
    """
    norm_path = [_normalise(p).lower() for p in path]
    full = "/".join(norm_path)
    q = _normalise(query).lower()
    if not q:
        return False
    if q in (full, norm_path[-1]):
        return True
    # Allow partial path match (e.g. "Correction/Corriger le spectre")
    q_parts = [p for p in q.split("/") if p]
    return all(part in full for part in q_parts) or q in norm_path[-1]


def _tool_list_plugin_actions(
    proxy: LocalProxy, mainwindow: DLMainWindow, panel: str | None = None
) -> dict[str, Any]:
    """List menu actions contributed by registered DataLab plugins."""
    if panel is None:
        panels = [mainwindow.signalpanel, mainwindow.imagepanel]
    else:
        panels = [_resolve_panel(mainwindow, panel)]
    plugins_info: list[dict[str, Any]] = []
    for panel_widget in panels:
        for path, action in _collect_plugin_actions_for_panel(panel_widget):
            plugins_info.append(
                {
                    "panel": panel_widget.PANEL_STR_ID,
                    "plugin": path[0],
                    "menu_path": "/".join(path),
                    "action": path[-1],
                    "enabled": bool(action.isEnabled()),
                    "tip": action.toolTip() or "",
                }
            )
    return {"plugin_actions": plugins_info}


def _tool_trigger_plugin_action(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    action: str,
    panel: str | None = None,
    plugin: str | None = None,
) -> dict[str, Any]:
    """Trigger a plugin menu action by name or full menu path."""
    if panel is None:
        panels = [mainwindow.signalpanel, mainwindow.imagepanel]
    else:
        panels = [_resolve_panel(mainwindow, panel)]

    candidates: list[tuple[Any, list[str], QW.QAction]] = []
    for panel_widget in panels:
        for path, qaction in _collect_plugin_actions_for_panel(panel_widget):
            if (
                plugin is not None
                and _normalise(plugin).lower() not in _normalise(path[0]).lower()
            ):
                continue
            if _action_matches(path, action):
                candidates.append((panel_widget, path, qaction))

    if not candidates:
        raise ValueError(
            f"No plugin action matches {action!r}"
            + (f" in plugin {plugin!r}" if plugin else "")
            + ". Use 'list_plugin_actions' to discover available actions."
        )
    if len(candidates) > 1:
        paths = [f"{p.PANEL_STR_ID}: {'/'.join(path)}" for p, path, _ in candidates]
        raise ValueError(
            f"Ambiguous plugin action {action!r}, multiple matches: {paths}. "
            "Pass an exact 'menu_path' (and 'plugin'/'panel' if needed)."
        )

    panel_widget, path, qaction = candidates[0]
    # Activate the panel hosting the action so that select-conditions and
    # the plugin's `signalpanel`/`imagepanel` accessors target the right one.
    proxy.set_current_panel(panel_widget.PANEL_STR_ID)
    if not qaction.isEnabled():
        raise ValueError(
            f"Plugin action {'/'.join(path)!r} is currently disabled. "
            "Make sure a compatible object is selected (use 'list_objects' "
            "and 'select_objects' first)."
        )
    qaction.trigger()
    return {
        "panel": panel_widget.PANEL_STR_ID,
        "plugin": path[0],
        "menu_path": "/".join(path),
    }


# =============================================================================
# Creation tools
# =============================================================================


def _tool_create_synthetic_signal(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    title: str,
    kind: str = "sin",
    npoints: int = 1024,
    xmin: float = 0.0,
    xmax: float = 1.0,
    frequency: float = 10.0,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_level: float = 0.0,
) -> dict[str, Any]:
    x = np.linspace(xmin, xmax, int(npoints))
    if kind == "sin":
        y = amplitude * np.sin(2 * np.pi * frequency * x) + offset
    elif kind == "cos":
        y = amplitude * np.cos(2 * np.pi * frequency * x) + offset
    elif kind == "gauss":
        center = 0.5 * (xmin + xmax)
        sigma = max((xmax - xmin) / 10.0, 1e-12)
        y = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + offset
    elif kind == "noise":
        y = amplitude * np.random.standard_normal(x.size) + offset
    elif kind == "ramp":
        y = amplitude * (x - xmin) / max(xmax - xmin, 1e-12) + offset
    else:
        raise ValueError(
            f"Unknown signal kind {kind!r} (expected sin, cos, gauss, noise, ramp)."
        )
    if noise_level:
        y = y + noise_level * np.random.standard_normal(y.size)
    proxy.add_signal(title, x.astype(float), y.astype(float))
    return {"title": title, "size": int(x.size), "kind": kind}


def _tool_create_synthetic_image(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    title: str,
    kind: str = "gauss2d",
    width: int = 256,
    height: int = 256,
    amplitude: float = 1.0,
    sigma: float = 32.0,
    noise_level: float = 0.0,
) -> dict[str, Any]:
    h, w = int(height), int(width)
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "gauss2d":
        cx, cy = w / 2.0, h / 2.0
        data = amplitude * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    elif kind == "ramp":
        data = amplitude * (xx / max(w - 1, 1))
    elif kind == "noise":
        data = amplitude * np.random.standard_normal((h, w))
    elif kind == "checker":
        data = amplitude * (((xx // 16) + (yy // 16)) % 2).astype(float)
    else:
        raise ValueError(
            f"Unknown image kind {kind!r} (expected gauss2d, ramp, noise, checker)."
        )
    if noise_level:
        data = data + noise_level * np.random.standard_normal(data.shape)
    proxy.add_image(title, data.astype(float))
    return {"title": title, "shape": [h, w], "kind": kind}


def _tool_load_file(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    filename: str,
    panel: str | None = None,
) -> dict[str, Any]:
    panel_widget = _resolve_panel(mainwindow, panel)
    proxy.set_current_panel(panel_widget.PANEL_STR_ID)
    objs = panel_widget.load_from_files([filename])
    return {
        "filename": filename,
        "panel": panel_widget.PANEL_STR_ID,
        "loaded": [{"uuid": o.uuid, "title": o.title} for o in objs],
    }


# =============================================================================
# Processing tool
# =============================================================================


def _tool_apply_operation(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    name: str,
    parameters: dict[str, Any] | None = None,
    panel: str | None = None,
    target_uuid: str | None = None,
) -> dict[str, Any]:
    if panel is not None:
        proxy.set_current_panel(panel)
    if target_uuid is not None:
        proxy.select_objects([target_uuid])

    panel_widget = _resolve_panel(mainwindow, proxy.get_current_panel())
    processor = panel_widget.processor
    feature = processor.get_feature(name)
    param_obj = None
    if parameters and feature.paramclass is not None:
        param_obj = feature.paramclass()
        for key, value in parameters.items():
            try:
                setattr(param_obj, key, value)
            except Exception as exc:  # pylint: disable=broad-except
                raise ValueError(
                    f"Cannot set parameter {key!r} on "
                    f"{feature.paramclass.__name__}: {exc}"
                ) from exc
    proxy.calc(name, param_obj)
    return {"operation": name, "panel": proxy.get_current_panel()}


# =============================================================================
# Macro tool
# =============================================================================


def _proxy_public_api() -> set[str]:
    """Return the set of public method names of :class:`BaseProxy`."""
    # pylint: disable-next=import-outside-toplevel
    from datalab.control.baseproxy import BaseProxy  # noqa: WPS433

    return {
        name
        for name, member in inspect.getmembers(BaseProxy, inspect.isfunction)
        if not name.startswith("_")
    }


def _validate_macro_code(code: str) -> list[str]:
    """Validate macro source code statically.

    Returns:
        List of human-readable warnings/errors. Empty list means the code
        looks fine. Syntax errors are returned as a single-element list.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"SyntaxError: {exc.msg} (line {exc.lineno})"]

    proxy_names: set[str] = set()
    for node in ast.walk(tree):
        # Detect `proxy = RemoteProxy(...)` / `proxy = LocalProxy(...)`
        if isinstance(node, ast.Assign):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in ("RemoteProxy", "LocalProxy")
            ):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        proxy_names.add(tgt.id)
    if not proxy_names:
        proxy_names = {"proxy"}  # heuristic default

    public = _proxy_public_api()
    warnings: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in proxy_names
            and node.attr not in public
            and node.attr not in seen
        ):
            seen.add(node.attr)
            warnings.append(
                f"Unknown method 'proxy.{node.attr}' (line {node.lineno}). "
                f"Use 'get_api_help' to see the available methods."
            )
    return warnings


def _wait_macro_finished(macro, console, console_before_len: int, timeout: float):
    """Block until ``macro`` finishes (or ``timeout`` elapses); return output."""
    loop = QC.QEventLoop()
    macro.FINISHED.connect(loop.quit)
    timed_out = {"flag": False}

    def _on_timeout() -> None:
        timed_out["flag"] = True
        loop.quit()

    timer = QC.QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(_on_timeout)
    timer.start(int(max(timeout, 0.1) * 1000))
    loop.exec_()
    timer.stop()

    full = console.toPlainText()
    output = full[console_before_len:]
    return output, timed_out["flag"]


def _tool_create_and_run_macro(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    title: str,
    code: str,
    autorun: bool = True,
    timeout: float = 30.0,
) -> dict[str, Any]:
    # Static validation FIRST so we don't bother running invalid code.
    warnings = _validate_macro_code(code)
    syntax_errors = [w for w in warnings if w.startswith("SyntaxError")]
    if syntax_errors:
        return {
            "title": title,
            "ok": False,
            "validation_errors": syntax_errors,
            "hint": "Fix the syntax errors and call create_and_run_macro again.",
        }

    macropanel = mainwindow.macropanel
    macro = macropanel.add_macro_with_code(title, code)

    result: dict[str, Any] = {"title": macro.title, "validation_warnings": warnings}
    if not autorun:
        result["autorun"] = False
        return result

    console = macropanel.console
    before_len = len(console.toPlainText())
    macro.run()
    output, timed_out = _wait_macro_finished(macro, console, before_len, timeout)

    exit_code = macro.get_exit_code()
    # Trim very long outputs to keep token budget reasonable.
    trimmed = output if len(output) <= 4000 else "...[truncated]...\n" + output[-4000:]
    result.update(
        {
            "autorun": True,
            "exit_code": exit_code,
            "ok": (exit_code == 0) and not timed_out,
            "timed_out": timed_out,
            "console_output": trimmed,
        }
    )
    if timed_out:
        result["hint"] = (
            "Macro did not finish within the timeout. Increase 'timeout' or "
            "simplify the macro."
        )
    elif exit_code != 0:
        result["hint"] = (
            "Macro failed. Read 'console_output' (Python traceback included), "
            "fix the code and call create_and_run_macro again."
        )
    return result


def _tool_get_macro_console_output(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    last_n_chars: int = 4000,
) -> dict[str, Any]:
    text = mainwindow.macropanel.console.toPlainText()
    if last_n_chars and len(text) > last_n_chars:
        text = "...[truncated]...\n" + text[-int(last_n_chars) :]
    return {"console_output": text}


# =============================================================================
# API help tool
# =============================================================================


def _format_signature(member) -> str:
    try:
        return str(inspect.signature(member))
    except (TypeError, ValueError):
        return "(...)"


def _members_help(cls, kind: str = "method") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, member in sorted(inspect.getmembers(cls)):
        if name.startswith("_"):
            continue
        if kind == "method" and not callable(member):
            continue
        doc = inspect.getdoc(member) or ""
        first = doc.strip().splitlines()[0] if doc else ""
        entry: dict[str, Any] = {"name": name, "doc": first}
        if callable(member):
            entry["signature"] = _format_signature(member)
        out.append(entry)
    return out


def _tool_get_api_help(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    symbol: str,
) -> dict[str, Any]:
    sym = symbol.lower()
    if sym in ("proxy", "remoteproxy", "localproxy", "baseproxy"):
        # pylint: disable-next=import-outside-toplevel
        from datalab.control.baseproxy import BaseProxy  # noqa: WPS433

        return {"symbol": "RemoteProxy", "methods": _members_help(BaseProxy)}
    if sym == "signalobj":
        # pylint: disable-next=import-outside-toplevel
        from sigima.objects import SignalObj  # noqa: WPS433

        return {
            "symbol": "SignalObj",
            "attributes": [
                "x",
                "y",
                "dx",
                "dy",
                "title",
                "xunit",
                "yunit",
                "xlabel",
                "ylabel",
                "metadata",
                "roi",
                "uuid",
            ],
            "methods": _members_help(SignalObj),
        }
    if sym == "imageobj":
        # pylint: disable-next=import-outside-toplevel
        from sigima.objects import ImageObj  # noqa: WPS433

        return {
            "symbol": "ImageObj",
            "attributes": [
                "data",
                "title",
                "x0",
                "y0",
                "dx",
                "dy",
                "xunit",
                "yunit",
                "zunit",
                "metadata",
                "roi",
                "uuid",
            ],
            "methods": _members_help(ImageObj),
        }
    if sym in ("sigima.params", "params"):
        # pylint: disable-next=import-outside-toplevel
        import sigima.params as sp  # noqa: WPS433

        names = sorted(n for n in dir(sp) if n.endswith("Param"))
        return {"symbol": "sigima.params", "param_classes": names}
    raise ValueError(
        f"Unknown symbol {symbol!r}. Use one of: "
        "'proxy', 'SignalObj', 'ImageObj', 'sigima.params'."
    )


# =============================================================================
# View capture tool (multimodal)
# =============================================================================


def _tool_capture_view(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    panel: str | None = None,
) -> ToolResult:
    panel_widget = _resolve_panel(mainwindow, panel)
    plotwidget = panel_widget.plothandler.plotwidget
    pixmap: QG.QPixmap = plotwidget.grab()
    if pixmap.isNull():
        return ToolResult(ok=False, error="Failed to grab plot widget pixmap.")

    buf = QC.QBuffer()
    buf.open(QC.QBuffer.WriteOnly)
    pixmap.save(buf, "PNG")
    b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
    buf.close()

    panel_id = panel_widget.PANEL_STR_ID
    summary = {
        "panel": panel_id,
        "size": [pixmap.width(), pixmap.height()],
        "format": "image/png",
        "note": ("Screenshot follows in the next user message as an image block."),
    }
    followup = ChatMessage(
        role="user",
        content=[
            {
                "type": "text",
                "text": (
                    f"Here is the current view of the {panel_id} panel "
                    "(captured via 'capture_view'):"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ],
    )
    return ToolResult(ok=True, data=summary, followup_messages=[followup])


# =============================================================================
# Schema definitions
# =============================================================================


_PANEL_PARAM = {
    "type": "string",
    "enum": ["signal", "image"],
    "description": "Target panel name. Defaults to the currently active panel.",
}


def build_default_registry() -> ToolRegistry:
    """Build the default tool registry exposed to the LLM."""
    reg = ToolRegistry()

    reg.register(
        Tool(
            name="list_objects",
            description=(
                "List objects (signals or images) in a DataLab panel. "
                "Returns titles and UUIDs."
            ),
            parameters={
                "type": "object",
                "properties": {"panel": _PANEL_PARAM},
                "additionalProperties": False,
            },
            handler=_tool_list_objects,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="get_current_panel",
            description="Return the name of the currently active panel.",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=_tool_get_current_panel,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="get_object_info",
            description=(
                "Return detailed information about a signal or image object: "
                "shape, dtype, ranges, units, ROI count, metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "nb_id_title": {
                        "type": ["string", "integer", "null"],
                        "description": (
                            "Object UUID, title or 1-based index. "
                            "Defaults to the current object."
                        ),
                    },
                    "panel": _PANEL_PARAM,
                },
                "additionalProperties": False,
            },
            handler=_tool_get_object_info,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="list_available_operations",
            description=(
                "List all computation features registered in a panel processor. "
                "Each entry includes the operation name, title, pattern "
                "(1_to_1, 1_to_0, n_to_1, 2_to_1, 1_to_n) and its parameter "
                "fields, suitable for calling 'apply_operation'."
            ),
            parameters={
                "type": "object",
                "properties": {"panel": _PANEL_PARAM},
                "additionalProperties": False,
            },
            handler=_tool_list_available_operations,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="list_plugin_actions",
            description=(
                "List menu actions contributed by registered DataLab plugins. "
                "Each entry includes the hosting panel, the plugin name, the "
                "full menu path (e.g. 'ASNR – Analyse spectrale/Correction/"
                "Corriger le spectre…'), the leaf action title, the current "
                "enabled state and the tooltip. Use this BEFORE calling "
                "'trigger_plugin_action' so you can pick the right action."
            ),
            parameters={
                "type": "object",
                "properties": {"panel": _PANEL_PARAM},
                "additionalProperties": False,
            },
            handler=_tool_list_plugin_actions,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="trigger_plugin_action",
            description=(
                "Trigger a plugin menu action by its title or its full menu "
                "path (as returned by 'list_plugin_actions'). Use this when "
                "the user asks for a feature provided by a plugin (e.g. "
                "'corrige le spectre avec le plugin ASNR'). The action runs "
                "synchronously on the active panel; ensure a compatible "
                "object is selected first if needed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Action title or full menu path "
                            "(e.g. 'Corriger le spectre' or "
                            "'ASNR – Analyse spectrale/Correction/"
                            "Corriger le spectre')."
                        ),
                    },
                    "plugin": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional plugin name to disambiguate when "
                            "several plugins expose actions with similar "
                            "titles."
                        ),
                    },
                    "panel": _PANEL_PARAM,
                },
                "required": ["action"],
                "additionalProperties": False,
            },
            handler=_tool_trigger_plugin_action,
        )
    )
    reg.register(
        Tool(
            name="create_synthetic_signal",
            description=(
                "Create a synthetic signal and add it to the signal panel. "
                "Supported kinds: 'sin', 'cos', 'gauss', 'noise', 'ramp'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["sin", "cos", "gauss", "noise", "ramp"],
                        "default": "sin",
                    },
                    "npoints": {"type": "integer", "minimum": 2, "default": 1024},
                    "xmin": {"type": "number", "default": 0.0},
                    "xmax": {"type": "number", "default": 1.0},
                    "frequency": {"type": "number", "default": 10.0},
                    "amplitude": {"type": "number", "default": 1.0},
                    "offset": {"type": "number", "default": 0.0},
                    "noise_level": {"type": "number", "default": 0.0},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=_tool_create_synthetic_signal,
        )
    )
    reg.register(
        Tool(
            name="create_synthetic_image",
            description=(
                "Create a synthetic image and add it to the image panel. "
                "Supported kinds: 'gauss2d', 'ramp', 'noise', 'checker'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["gauss2d", "ramp", "noise", "checker"],
                        "default": "gauss2d",
                    },
                    "width": {"type": "integer", "minimum": 2, "default": 256},
                    "height": {"type": "integer", "minimum": 2, "default": 256},
                    "amplitude": {"type": "number", "default": 1.0},
                    "sigma": {"type": "number", "default": 32.0},
                    "noise_level": {"type": "number", "default": 0.0},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            handler=_tool_create_synthetic_image,
        )
    )
    reg.register(
        Tool(
            name="load_file",
            description=(
                "Load a signal or image file into DataLab. The file format is "
                "auto-detected based on the filename extension."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "panel": _PANEL_PARAM,
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
            handler=_tool_load_file,
        )
    )
    reg.register(
        Tool(
            name="apply_operation",
            description=(
                "Apply a registered computation feature to the current selection. "
                "Use 'list_available_operations' first to discover the operation "
                "name and its parameter fields. Pass parameter values as a dict "
                "matching the parameter class field names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Operation name (e.g. 'normalize').",
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Parameter values keyed by field name "
                            "(see 'list_available_operations')."
                        ),
                        "additionalProperties": True,
                    },
                    "panel": _PANEL_PARAM,
                    "target_uuid": {
                        "type": ["string", "null"],
                        "description": (
                            "If provided, select this object before running the "
                            "operation."
                        ),
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=_tool_apply_operation,
        )
    )
    reg.register(
        Tool(
            name="create_and_run_macro",
            description=(
                "Create a new macro tab in the Macro panel with the given Python "
                "code, then optionally run it SYNCHRONOUSLY and return the macro "
                "console output (stdout + stderr, including Python tracebacks) "
                "and exit code. The macro has access to "
                "'datalab.control.proxy.RemoteProxy' to drive DataLab. The code "
                "is statically validated (syntax + unknown 'proxy.*' methods) "
                "before execution. Use this for complex multi-step scripts; "
                "prefer 'apply_operation' for atomic actions. If 'ok' is False, "
                "read 'console_output' or 'validation_errors' and try again."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "code": {
                        "type": "string",
                        "description": "Full Python source code of the macro.",
                    },
                    "autorun": {"type": "boolean", "default": True},
                    "timeout": {
                        "type": "number",
                        "default": 30.0,
                        "description": (
                            "Maximum number of seconds to wait for the macro to finish."
                        ),
                    },
                },
                "required": ["title", "code"],
                "additionalProperties": False,
            },
            handler=_tool_create_and_run_macro,
        )
    )
    reg.register(
        Tool(
            name="get_macro_console_output",
            description=(
                "Return the current contents of the Macro panel console "
                "(includes prior macro runs)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "last_n_chars": {"type": "integer", "default": 4000},
                },
                "additionalProperties": False,
            },
            handler=_tool_get_macro_console_output,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="get_api_help",
            description=(
                "Return the public API of DataLab objects: methods of "
                "RemoteProxy, attributes/methods of SignalObj or ImageObj, or "
                "the list of available 'sigima.params' parameter classes. "
                "Call this BEFORE writing a macro that uses unfamiliar methods "
                "to avoid hallucinating attribute names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "enum": [
                            "proxy",
                            "SignalObj",
                            "ImageObj",
                            "sigima.params",
                        ],
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
            handler=_tool_get_api_help,
            readonly=True,
        )
    )
    reg.register(
        Tool(
            name="capture_view",
            description=(
                "Grab a PNG screenshot of the current signal or image plot "
                "widget and inject it as an image into the conversation, so "
                "that the assistant can visually inspect the data (signal "
                "shape, image appearance, presence of artefacts, result of a "
                "processing step, etc.). Requires a multimodal-capable model."
            ),
            parameters={
                "type": "object",
                "properties": {"panel": _PANEL_PARAM},
                "additionalProperties": False,
            },
            handler=_tool_capture_view,
            readonly=True,
        )
    )

    return reg
