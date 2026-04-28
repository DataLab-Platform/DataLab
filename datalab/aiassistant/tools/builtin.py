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

from typing import TYPE_CHECKING, Any

import numpy as np

from datalab.aiassistant.tools.registry import Tool, ToolRegistry

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


def _tool_create_and_run_macro(
    proxy: LocalProxy,
    mainwindow: DLMainWindow,
    title: str,
    code: str,
    autorun: bool = True,
) -> dict[str, Any]:
    macropanel = mainwindow.macropanel
    macro = macropanel.add_macro_with_code(title, code)
    if autorun:
        proxy.run_macro(macro.title)
    return {"title": macro.title, "autorun": bool(autorun)}


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
                "code, then optionally run it. The macro has access to "
                "'datalab.control.proxy.RemoteProxy' to drive DataLab. Use this "
                "for complex multi-step scripts; prefer 'apply_operation' for "
                "atomic actions."
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
                },
                "required": ["title", "code"],
                "additionalProperties": False,
            },
            handler=_tool_create_and_run_macro,
        )
    )

    return reg
