# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. History panel (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import functools
import html
import inspect
import json
import logging
import os
import os.path as osp
import warnings
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Generator
from uuid import uuid4

import numpy as np
import sigima.proc.image
import sigima.proc.signal
from guidata.configtools import get_icon
from guidata.dataset.conv import dataset_to_json, json_to_dataset
from guidata.dataset.datatypes import DataSet
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from qtpy.compat import getopenfilename, getsavefilename
from sigima.objects import ImageObj, SignalObj

from datalab.config import Conf, _
from datalab.gui import ObjItf
from datalab.gui.panel.base import AbstractPanel
from datalab.objectmodel import get_uuid
from datalab.utils.qthelpers import qt_try_loadsave_file, save_restore_stds

if TYPE_CHECKING:
    import guidata.dataset as gds

    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.processor.base import BaseProcessor, FeatureNotFoundError
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


# Keys used in the kwargs dict to mark DataSet payloads, so that the
# serialization layer can round-trip them as JSON strings instead of pickling
# arbitrary Python objects.
HISTORY_SCHEMA_VERSION = 1
# Per-action schema. Bumped to 4 to persist the ``_saved_kwargs`` snapshot
# used by the Edit mode Restore feature: persisting it across save/reload
# lets the user revert edits even after closing and re-opening the session,
# as long as Edit mode has not been definitively committed (toggled off).
# Schema version 3 introduced the ``plugin_origin`` field used to track the
# originating plugin of a compute action (see ``BaseProcessor.add_feature``).
# Sessions/actions with ``schema_version <= 2`` are still supported: the
# loader leaves ``plugin_origin`` as ``None`` and missing-plugin replay
# simply produces a generic ``FeatureNotFoundError`` instead of a
# plugin-aware warning. Schema version 2 introduced the ``output_uuids``
# field used by the bijective action ↔ output mapping (see
# ``HistoryPanel._output_to_action``). Sessions/actions with
# ``schema_version <= 3`` leave ``_saved_kwargs`` as ``None`` on load
# (no pending edits to restore).
HISTORY_ACTION_SCHEMA_VERSION = 4
_DATASET_MARKER = "__dataset_json__"
_DATASET_LIST_MARKER = "__dataset_list_json__"
_ROI_MARKER = "__roi_json__"

_logger = logging.getLogger(__name__)


def _numpy_to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_json_safe(i) for i in obj]
    return obj


def _encode_roi(roi: Any) -> str:
    """Encode a sigima ROI object to a JSON string via ``to_dict()``."""
    from sigima.objects.base import BaseROI  # local to avoid circular import

    if not isinstance(roi, BaseROI):
        raise TypeError(f"Expected BaseROI instance, got {type(roi)!r}")
    roi_dict = _numpy_to_json_safe(roi.to_dict())
    # Store the concrete class so we can reconstruct on decode.
    payload = {
        "module": type(roi).__module__,
        "class": type(roi).__qualname__,
        "data": roi_dict,
    }
    return json.dumps(payload)


def _decode_roi(encoded: str) -> Any:
    """Decode a JSON string back to a sigima ROI object.

    Only classes from trusted ``sigima.`` modules that are actual
    :class:`sigima.objects.base.BaseROI` subclasses are allowed.

    Raises:
        ValueError: If the module is not a trusted sigima module or the
            resolved class is not a BaseROI subclass.
    """
    import importlib

    from sigima.objects.base import BaseROI  # local to avoid circular import

    _TRUSTED_ROI_MODULE_PREFIX = "sigima."

    payload = json.loads(encoded)
    module_name = payload["module"]
    class_name = payload["class"]

    if not module_name.startswith(_TRUSTED_ROI_MODULE_PREFIX):
        raise ValueError(
            f"Untrusted ROI module {module_name!r}: "
            f"only modules under {_TRUSTED_ROI_MODULE_PREFIX!r} are allowed"
        )

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)

    if not (isinstance(cls, type) and issubclass(cls, BaseROI)):
        raise ValueError(
            f"{module_name}.{class_name} is not a BaseROI subclass"
        )

    return cls.from_dict(payload["data"])


def _encode_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Encode kwargs for HDF5 storage: replace ``DataSet``, ``list[DataSet]``,
    and sigima ROI values with marker dicts holding their JSON representation.

    All other values must already be HDF5-friendly primitives (str, int, float,
    bool, list/tuple of the same).

    Args:
        kwargs: Raw kwargs dict (may contain ``DataSet`` or ROI instances).

    Returns:
        A new dict with special values wrapped in marker dicts.
    """
    from sigima.objects.base import BaseROI  # local to avoid circular import

    encoded: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, DataSet):
            encoded[key] = {_DATASET_MARKER: dataset_to_json(value)}
        elif isinstance(value, BaseROI):
            encoded[key] = {_ROI_MARKER: _encode_roi(value)}
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, DataSet) for item in value)
        ):
            encoded[key] = {
                _DATASET_LIST_MARKER: [dataset_to_json(item) for item in value]
            }
        else:
            encoded[key] = value
    return encoded


def _decode_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`_encode_kwargs`."""
    decoded: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, dict) and _DATASET_MARKER in value:
            try:
                decoded[key] = json_to_dataset(value[_DATASET_MARKER])
            except Exception:  # pylint: disable=broad-except
                warnings.warn(
                    _("Failed to deserialize history DataSet kwarg %r.") % key
                )
                decoded[key] = None
        elif isinstance(value, dict) and _ROI_MARKER in value:
            try:
                decoded[key] = _decode_roi(value[_ROI_MARKER])
            except Exception as exc:
                raise ValueError(
                    f"Failed to deserialize history ROI kwarg {key!r}: {exc}"
                ) from exc
        elif isinstance(value, dict) and _DATASET_LIST_MARKER in value:
            try:
                decoded[key] = [
                    json_to_dataset(item) for item in value[_DATASET_LIST_MARKER]
                ]
            except Exception:  # pylint: disable=broad-except
                warnings.warn(
                    _("Failed to deserialize history DataSet-list kwarg %r.") % key
                )
                decoded[key] = []
        else:
            decoded[key] = value
    return decoded


def _copy_history_value(value: Any) -> Any:
    """Return an independent copy of a history-serializable value."""
    from sigima.objects.base import BaseROI  # local to avoid circular import

    if callable(value):
        raise TypeError("History duplication does not support callable kwargs")
    if isinstance(value, DataSet):
        return json_to_dataset(dataset_to_json(value))
    if isinstance(value, BaseROI):
        return _decode_roi(_encode_roi(value))
    if isinstance(value, dict):
        return {key: _copy_history_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_history_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_history_value(item) for item in value)
    return deepcopy(value)


def get_datetime_str() -> str:
    """Return current date and time as a string"""
    return QC.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")


def add_to_history(kwargs_names: list[str] | None = None, title: str | None = None):
    """Method decorator to add the method call to the history panel as a UI entry.

    Args:
        kwargs_names: List of keyword arguments to add to the history action.
         Defaults to None.
        title: Title of the history action. Defaults to None.
    """
    if kwargs_names is None:
        kwargs_names = []

    def add_to_history_decorator(func):
        """Decorator function"""

        @functools.wraps(func)
        def method_wrapper(*args, **kwargs):
            """Decorator wrapper function"""
            self: BaseDataPanel | BaseProcessor = args[0]
            history: HistoryPanel = self.mainwindow.historypanel
            histkwargs = {k: kwargs[k] for k in kwargs_names if k in kwargs}
            target = _resolve_self_target(self)
            if target is not None:
                history.add_ui_entry(
                    kwargs.get("title", title) or func.__name__,
                    target=target,
                    method_name=func.__name__,
                    save_state=kwargs.get("save_state", True),
                    **histkwargs,
                )
            return func(*args, **kwargs)

        return method_wrapper

    return add_to_history_decorator


def _resolve_self_target(self_obj: Any) -> str | None:
    """Resolve a 'self' instance to a string target understood by replay.

    Used by the legacy ``@add_to_history`` decorator. Returns None when no
    safe routing is possible (in which case the entry is skipped).
    """
    panel_str = getattr(self_obj, "PANEL_STR_ID", None)
    if panel_str == "signal":
        return "signalpanel"
    if panel_str == "image":
        return "imagepanel"
    return None


# ---------------------------------------------------------------------------
# HistoryAction
# ---------------------------------------------------------------------------


class HistoryAction(ObjItf):
    """Object representing an action in the history panel.

    An action is a serialisable description of either a *compute* call (resolved
    via the panel processor's feature registry) or a *UI* call (resolved as a
    method on a known target -- ``mainwindow``/``signalpanel``/``imagepanel``).

    No Python ``Callable`` is ever pickled: a compute action is identified by
    ``(panel_str, func_name, pattern)`` and a UI action by ``(target,
    method_name)``. ``DataSet`` payloads inside ``kwargs`` are serialised with
    :func:`guidata.dataset.conv.dataset_to_json`.
    """

    KIND_COMPUTE = "compute"
    KIND_UI = "ui"

    FUNC_EDIT_MODE = "edit"  # Name of the function parameter to enable edit mode
    # Methods that create new data objects. During non-persistent (output-suppressed)
    # replay, these UI actions are skipped so the panel object count stays stable.
    UI_CREATION_METHODS: frozenset[str] = frozenset({"new_object"})

    def __init__(
        self,
        title: str = "",
        kind: str = KIND_UI,
        # --- compute-only --------------------------------------------------
        panel_str: str | None = None,
        func_name: str | None = None,
        pattern: str | None = None,
        # --- ui-only -------------------------------------------------------
        target: str | None = None,
        method_name: str | None = None,
        # --- common --------------------------------------------------------
        kwargs: dict[str, Any] | None = None,
        state: WorkspaceState | None = None,
        plugin_origin: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.__title = title or ""
        self.kind = kind
        # Compute kind:
        self.panel_str = panel_str
        self.func_name = func_name
        self.pattern = pattern
        # UI kind:
        self.target = target
        self.method_name = method_name
        # Common:
        self.kwargs: dict[str, Any] = (
            {} if kwargs is None else {k: v for k, v in kwargs.items() if v is not None}
        )
        self.state = WorkspaceState() if state is None else state
        self.dtstr: str = get_datetime_str()
        self.uuid: str = str(uuid4())
        self.schema_version: int = HISTORY_ACTION_SCHEMA_VERSION
        # UUIDs of the data objects produced by this action (bijective mapping
        # maintained by :class:`HistoryPanel`). Populated post-compute via
        # :meth:`HistoryPanel.register_action_outputs`. Empty for ``1_to_0``
        # patterns, for UI actions, and for legacy (schema_version=1) sessions
        # loaded from disk (the heuristic fallback then takes over).
        self.output_uuids: list[str] = []
        # Plugin origin descriptor for compute actions (None for built-in
        # Sigima/DataLab features). Populated at registration time by
        # :meth:`BaseProcessor.add_feature` and propagated through
        # ``add_compute_entry_from_pp``. See
        # :func:`datalab.gui.processor.base._detect_plugin_origin` for shape.
        # Persisted as a JSON string in HDF5 (schema_version >= 3).
        self.plugin_origin: dict[str, Any] | None = plugin_origin
        # Transient flag (NOT serialized): set during a cascade recompute to
        # display a "stale" visual marker in the tree. Cleared once the
        # action has been recomputed.
        self.is_stale: bool = False
        # Snapshot of original kwargs before edit-mode modification.
        # Set lazily when the first edit-mode change touches this action.
        # Persisted to HDF5 (schema_version >= 4) so that the Restore
        # action still works after a save/reload cycle while Edit mode is
        # active. Cleared by ``discard_snapshot`` (definitive commit when
        # toggling Edit mode off) or ``restore_kwargs`` (Restore button).
        self._saved_kwargs: dict[str, Any] | None = None

    def snapshot_kwargs(self) -> None:
        """Save a copy of the current kwargs as the pre-edit baseline.

        No-op if a snapshot already exists (preserves the original baseline
        across multiple edit-mode replays).
        """
        if self._saved_kwargs is None:
            self._saved_kwargs = {
                key: _copy_history_value(value)
                for key, value in self.kwargs.items()
            }

    def restore_kwargs(self) -> None:
        """Restore kwargs from the saved snapshot and clear the snapshot."""
        if self._saved_kwargs is not None:
            self.kwargs = self._saved_kwargs
            self._saved_kwargs = None

    def discard_snapshot(self) -> None:
        """Discard the saved snapshot (accept current kwargs as definitive)."""
        self._saved_kwargs = None

    @property
    def has_pending_edits(self) -> bool:
        """Return True if this action has unsaved edit-mode changes."""
        return self._saved_kwargs is not None

    def regenerate_uuid(self):
        """Regenerate UUID after loading from a file (no-op: per-action UUID)."""

    def copy(self, title_suffix: str | None = None) -> HistoryAction:
        """Return an independent copy of this history action."""
        state = self.state.copy()
        title = self.title
        if title_suffix:
            title = f"{title} {title_suffix}"
        new_action = HistoryAction(
            title=title,
            kind=self.kind,
            panel_str=self.panel_str,
            func_name=self.func_name,
            pattern=self.pattern,
            target=self.target,
            method_name=self.method_name,
            kwargs={
                key: _copy_history_value(value)
                for key, value in self.kwargs.items()
            },
            state=state,
        )
        new_action.output_uuids = list(self.output_uuids)
        # Note: _saved_kwargs is intentionally NOT propagated to the copy.
        # Copying an action acts as an implicit commit (no pending edits).
        return new_action

    def copy_with_uuid_remap(
        self, uuid_remap: dict[str, dict[str, str]]
    ) -> HistoryAction:
        """Return a copy of this action with all captured UUIDs rewritten.

        Args:
            uuid_remap: Per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``
             used to translate captured UUIDs to the cloned objects created by
             the Duplicate operation.

        Returns:
            A new independent :class:`HistoryAction` with remapped UUIDs.
        """
        new_action = self.copy()
        # Rewrite state.selection
        for pstr, uuids in new_action.state.selection.items():
            pmap = uuid_remap.get(pstr, {})
            new_action.state.selection[pstr] = [pmap.get(u, u) for u in uuids]
        # Rewrite state.object_metadata keys
        for pstr, metadata in new_action.state.object_metadata.items():
            pmap = uuid_remap.get(pstr, {})
            new_action.state.object_metadata[pstr] = {
                pmap.get(uuid, uuid): val for uuid, val in metadata.items()
            }
        # Rewrite obj2_uuids in kwargs
        obj2 = new_action.kwargs.get("obj2_uuids")
        if obj2:
            if isinstance(obj2, str):
                obj2 = [obj2]
            pstr = new_action.panel_str or ""
            pmap = uuid_remap.get(pstr, {})
            rewritten = [pmap.get(u, u) for u in obj2]
            new_action.kwargs["obj2_uuids"] = (
                rewritten[0] if len(rewritten) == 1 else rewritten
            )
        # Rewrite output_uuids — they reference the target panel.
        if new_action.output_uuids:
            pstr = new_action.panel_str or ""
            pmap = uuid_remap.get(pstr, {})
            new_action.output_uuids = [
                pmap.get(u, u) for u in new_action.output_uuids
            ]
        return new_action

    @property
    def title(self) -> str:
        """Return object title"""
        return self.__title

    # ------------------------------------------------------------------
    # Description rendering (used by the tree view)
    # ------------------------------------------------------------------

    def __iter_param_kwargs(self) -> Generator[Any, None, None]:
        """Yield kwargs values whose name ends with ``param`` (typically DataSets)."""
        for kwname, value in self.kwargs.items():
            if kwname.endswith("param") and value is not None:
                yield value

    @property
    def description(self) -> str:
        """Return object description (string representing function parameters)"""
        desc = ""
        no_parameters = True
        for param in self.__iter_param_kwargs():
            if desc:
                desc += os.linesep
            desc += str(param)
            no_parameters = False
        if desc or no_parameters:
            if desc:
                return desc
        # Fall back to a textual hint of the resolved callable
        return self.__fallback_doc()

    def __fallback_doc(self) -> str:
        """Return a single-line docstring for the underlying call, if available."""
        try:
            func = self._resolve_callable()
        except Exception:  # pylint: disable=broad-except
            return ""
        doc = getattr(func, "__doc__", None) or ""
        return doc.splitlines()[0] if doc else ""

    @property
    def description_summary(self) -> str:
        """Return a short, single-line summary of the description (collapsed view).

        For DataSet parameters, uses the dataset title followed by a compact
        representation of its public fields ("name=value, ..."). Falls back to
        the first non-empty line of the full description when no DataSet is
        present.
        """
        summaries: list[str] = []
        for param in self.__iter_param_kwargs():
            if isinstance(param, DataSet):
                title = param.get_title() or ""
                # Collect "name=value" for each non-private item of the DataSet.
                pairs: list[str] = []
                for item in param.get_items():
                    name = item.get_name()
                    if name.startswith("_"):
                        continue
                    try:
                        value = item.get_value(param)
                    except Exception:  # pylint: disable=broad-except
                        continue
                    # Format floats compactly, leave other reprs as-is
                    if isinstance(value, float):
                        value_str = f"{value:g}"
                    else:
                        value_str = str(value)
                    pairs.append(f"{name}={value_str}")
                if pairs:
                    summaries.append(
                        f"{title}: {', '.join(pairs)}" if title else ", ".join(pairs)
                    )
                elif title:
                    summaries.append(title)
        if summaries:
            return " | ".join(summaries)
        for line in self.description.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @property
    def description_html(self) -> str:
        """Return rich-text (HTML) description used for the expanded view."""
        # Normal path
        parts: list[str] = []
        no_parameters = True
        for param in self.__iter_param_kwargs():
            no_parameters = False
            if isinstance(param, DataSet):
                parts.append(param.to_html())
            else:
                parts.append(html.escape(str(param)).replace("\n", "<br>"))
        if parts:
            return "<br><br>".join(parts)
        if no_parameters:
            text = self.description
            if not text:
                return ""
            return html.escape(text).replace("\n", "<br>")
        return ""

    def to_macro_code(
        self,
        step_index: int,
        input_var: str,
        imports: set[str],
        obj2_var: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Return Python source lines for this action as a standalone sigima call.

        Args:
            step_index: Step number for variable naming.
            input_var: Name of the input variable from the previous step.
            imports: Mutable set of import statements accumulated by the caller.
            obj2_var: Resolved variable name for the second operand (2-to-1
                pattern). When ``None``, the second operand is left as a
                placeholder.

        Returns:
            Tuple of (code_lines, output_var_name). ``output_var_name`` is
            ``None`` for UI-kind actions (no data output).
        """
        if self.kind != self.KIND_COMPUTE:
            return [f"# (UI) {self.title}  [skipped]"], None

        lines: list[str] = []
        output_var = f"result_{step_index}"

        # Determine the sigima module alias
        if self.panel_str == "signal":
            mod_alias = "sips"
            imports.add("import sigima.proc.signal as sips")
        elif self.panel_str == "image":
            mod_alias = "sipi"
            imports.add("import sigima.proc.image as sipi")
        else:
            lines.append(
                f"# {self.title}  [unknown panel: {self.panel_str}]"
            )
            return lines, None

        lines.append(f"# Step {step_index}: {self.title}")

        param = self.kwargs.get("param")
        param_var: str | None = None

        if param is not None and isinstance(param, DataSet):
            param_var = f"param_{step_index}"
            param_class = type(param).__qualname__
            param_module = type(param).__module__
            imports.add(f"from {param_module} import {param_class}")
            lines.append(f"{param_var} = {param_class}()")
            # Reconstruct each attribute
            for item in param._items:  # noqa: SLF001
                attr_name = item._name  # noqa: SLF001
                value = getattr(param, attr_name, None)
                if value is not None:
                    lines.append(
                        f"{param_var}.{attr_name} = {value!r}"
                    )

        # Build the function call
        func_call = f"{mod_alias}.{self.func_name}"
        if self.pattern in ("1_to_1", "1_to_0"):
            if param_var:
                lines.append(
                    f"{output_var} = {func_call}"
                    f"({input_var}, {param_var})"
                )
            else:
                lines.append(
                    f"{output_var} = {func_call}({input_var})"
                )
        elif self.pattern == "n_to_1":
            if param_var:
                lines.append(
                    f"{output_var} = {func_call}"
                    f"([{input_var}], {param_var})"
                )
            else:
                lines.append(
                    f"{output_var} = {func_call}([{input_var}])"
                )
        elif self.pattern == "2_to_1":
            second = obj2_var or "...  # TODO: provide second operand"
            if param_var:
                lines.append(
                    f"{output_var} = {func_call}"
                    f"({input_var}, {second}, {param_var})"
                )
            else:
                lines.append(
                    f"{output_var} = {func_call}"
                    f"({input_var}, {second})"
                )
        elif self.pattern == "1_to_n":
            lines.append(
                f"{output_var} = {func_call}({input_var})"
            )
        else:
            lines.append(f"# Unknown pattern {self.pattern!r}")
            return lines, None

        return lines, output_var

    # ------------------------------------------------------------------
    # Workspace-state delegation
    # ------------------------------------------------------------------

    def is_current_state_compatible(
        self, mainwindow: DLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state."""
        return self.state.is_current_state_compatible(mainwindow, restore_selection)

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the associated workspace state."""
        self.state.restore(mainwindow)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def _resolve_target(self, mainwindow: DLMainWindow) -> Any:
        """Resolve the target object (UI kind) from the mainwindow."""
        attr = self.target or "mainwindow"
        if attr == "mainwindow":
            return mainwindow
        return getattr(mainwindow, attr)

    def _resolve_panel(self, mainwindow: DLMainWindow):
        """Resolve the data panel for a compute action."""
        if self.panel_str == "signal":
            return mainwindow.signalpanel
        if self.panel_str == "image":
            return mainwindow.imagepanel
        raise ValueError(
            f"Unknown panel_str {self.panel_str!r} for compute history action"
        )

    def _resolve_callable(self) -> Callable | None:
        """Best-effort lookup of the underlying callable, for description only."""
        if self.kind == self.KIND_COMPUTE and self.func_name:
            for module in (sigima.proc.signal, sigima.proc.image):
                func = getattr(module, self.func_name, None)
                if callable(func):
                    return func
        return None

    def _resolve_obj_by_uuid(self, mainwindow: DLMainWindow, uuid: str) -> Any | None:
        """Look up an object by UUID across both data panels."""
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            try:
                return panel.objmodel[uuid]
            except KeyError:
                continue
        return None

    def replay(
        self,
        mainwindow: DLMainWindow,
        restore_selection: bool,
        edit: bool,
        uuid_remap: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Replay the action.

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the workspace selection before replaying
             a UI-kind action. Ignored for compute-kind actions: their semantics
             depends on which objects are selected (e.g. ``n_to_1`` aggregators
             such as ``average`` require their captured multi-object selection),
             so the captured selection is always restored before running the
             computation.
            edit: if True, always open the dialog boxes to edit parameters; if False,
             use the parameters captured when the action was recorded
            uuid_remap: optional per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``
             used during full-session replay to translate captured UUIDs to the
             freshly-created ones. Defaults to an empty (identity) mapping.
        """
        if uuid_remap is None:
            uuid_remap = {}
        # Suppress history capture during replay to avoid recording
        # synthetic entries when the processor re-executes features.
        # The context manager is reentrant, so nesting with
        # HistoryPanel.replay_restore_actions() is safe.
        hpanel = getattr(mainwindow, "historypanel", None)
        if hpanel is not None:
            ctx = hpanel.replaying()
        else:
            from contextlib import nullcontext

            ctx = nullcontext()
        with ctx:
            self._replay_inner(mainwindow, restore_selection, edit, uuid_remap)

    def _replay_inner(
        self,
        mainwindow: DLMainWindow,
        restore_selection: bool,
        edit: bool,
        uuid_remap: dict[str, dict[str, str]],
    ) -> None:
        """Inner replay logic, always called under the replaying guard."""
        if self.kind == self.KIND_COMPUTE:
            # Compute actions are selection-driven: restore the captured
            # selection (translated through ``uuid_remap`` for session
            # replays) whenever it is still resolvable so chained replays
            # (especially ``n_to_1`` / ``2_to_1`` / ``1_to_n`` patterns)
            # operate on the original input objects rather than on whatever
            # the previous action left selected. When the captured UUIDs no
            # longer exist (e.g. heuristic remap missed an object), fall
            # back to the current selection -- replay may still fail
            # downstream, but with the native processor error rather than
            # an opaque ``WorkspaceState`` incompatibility.
            translated = self._translate_state(uuid_remap)
            if translated.is_current_state_compatible(mainwindow, False):
                translated.restore(mainwindow)
            self._replay_compute(mainwindow, edit, uuid_remap)
        else:
            if restore_selection:
                self.state.restore(mainwindow)
            self._replay_ui(mainwindow, edit)

    def _translate_state(self, uuid_remap: dict[str, dict[str, str]]) -> WorkspaceState:
        """Return a copy of ``self.state`` whose captured UUIDs have been
        translated through ``uuid_remap`` (identity when no mapping)."""
        if not uuid_remap:
            return self.state
        translated = WorkspaceState()
        for panel_str, uuids in self.state.selection.items():
            panel_map = uuid_remap.get(panel_str, {})
            translated.selection[panel_str] = [panel_map.get(u, u) for u in uuids]
        translated.states = dict(self.state.states)
        translated.titles = dict(self.state.titles)
        for panel_str, metadata in self.state.object_metadata.items():
            panel_map = uuid_remap.get(panel_str, {})
            translated.object_metadata[panel_str] = {
                panel_map.get(uuid, uuid): dict(signature)
                for uuid, signature in metadata.items()
            }
        return translated

    def _replay_compute(
        self,
        mainwindow: DLMainWindow,
        edit: bool,
        uuid_remap: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Replay a compute-kind action via ``processor.run_feature``."""
        if self.pattern == "multiple_1_to_1":
            raise NotImplementedError(
                _("Replaying compound 'multiple_1_to_1' actions is not supported yet.")
            )
        panel = self._resolve_panel(mainwindow)
        processor = panel.processor
        feature = processor.get_feature(self.func_name)
        run_kwargs: dict[str, Any] = {self.FUNC_EDIT_MODE: edit}

        param = self.kwargs.get("param")
        if self.pattern in {"1_to_1", "1_to_0", "n_to_1"}:
            if param is not None:
                run_kwargs["param"] = param
            if self.pattern == "n_to_1" and "pairwise" in self.kwargs:
                run_kwargs["pairwise"] = self.kwargs["pairwise"]
        elif self.pattern == "2_to_1":
            uuids = self.kwargs.get("obj2_uuids") or []
            if isinstance(uuids, str):
                uuids = [uuids]
            # Translate captured UUIDs through ``uuid_remap`` (session replay).
            # ``uuid_remap`` keys are ``panel.PANEL_STR_ID`` (matches
            # ``WorkspaceState.selection`` keys and
            # ``HistoryAction.panel_str``).
            panel_map = (uuid_remap or {}).get(panel.PANEL_STR_ID, {})
            uuids = [panel_map.get(u, u) for u in uuids]
            objs2 = [
                obj
                for obj in (self._resolve_obj_by_uuid(mainwindow, u) for u in uuids)
                if obj is not None
            ]
            if not objs2:
                raise ValueError(
                    _("Cannot replay 2-to-1 action: source object(s) missing.")
                )
            run_kwargs["obj2"] = objs2[0] if len(objs2) == 1 else objs2
            if param is not None:
                run_kwargs["param"] = param
            if "pairwise" in self.kwargs:
                run_kwargs["pairwise"] = self.kwargs["pairwise"]
        elif self.pattern == "1_to_n":
            params = self.kwargs.get("params") or []
            run_kwargs["params"] = params
        else:
            raise ValueError(f"Unknown compute pattern: {self.pattern!r}")
        processor.run_feature(feature, **run_kwargs)

    def _replay_ui(self, mainwindow: DLMainWindow, edit: bool) -> None:
        """Replay a UI-kind action by calling ``target.method_name(**kwargs)``."""
        hpanel = mainwindow.historypanel
        if (
            hpanel is not None
            and hpanel.is_output_suppressed()
            and self.method_name in self.UI_CREATION_METHODS
        ):
            return  # Skip creation UI during non-persistent replay
        target = self._resolve_target(mainwindow)
        # Safety guard for destructive UI actions: if the action would delete
        # objects but the captured selection no longer resolves to existing
        # UUIDs in the target panel, skip the call rather than delete whatever
        # is currently selected (which would silently destroy unrelated data).
        DESTRUCTIVE_METHODS = {"remove_object", "remove_group", "delete_all_objects"}
        if self.method_name in DESTRUCTIVE_METHODS:
            if target is None:
                _logger.warning(
                    "Skipping destructive replay '%s': target '%s' not found",
                    self.method_name, self.target,
                )
                return
            panel_str = getattr(target, "PANEL_STR_ID", None)
            if panel_str and self.state and self.state.selection.get(panel_str):
                existing_uuids = {
                    get_uuid(o)
                    for o in getattr(target, "objmodel", [])
                    if o is not None
                }
                captured = set(self.state.selection.get(panel_str, []))
                if not (captured & existing_uuids):
                    _logger.warning(
                        "Skipping destructive replay '%s': none of the captured "
                        "UUIDs %s exist in panel '%s' anymore",
                        self.method_name, list(captured), panel_str,
                    )
                    return
        method = getattr(target, self.method_name)
        call_kwargs = dict(self.kwargs)
        # Inject edit mode if the method supports it
        try:
            sig = inspect.signature(method)
            if self.FUNC_EDIT_MODE in sig.parameters:
                call_kwargs[self.FUNC_EDIT_MODE] = edit
        except (TypeError, ValueError):
            pass
        method(**call_kwargs)

    # ------------------------------------------------------------------
    # Serialisation -- no Callable is ever pickled
    # ------------------------------------------------------------------

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this action."""
        with writer.group("schema_version"):
            writer.write(self.schema_version)
        with writer.group("kind"):
            writer.write(self.kind)
        with writer.group("title"):
            writer.write(self.__title)
        if self.panel_str is not None:
            with writer.group("panel_str"):
                writer.write(self.panel_str)
        if self.func_name is not None:
            with writer.group("func_name"):
                writer.write(self.func_name)
        if self.pattern is not None:
            with writer.group("pattern"):
                writer.write(self.pattern)
        if self.target is not None:
            with writer.group("target"):
                writer.write(self.target)
        if self.method_name is not None:
            with writer.group("method_name"):
                writer.write(self.method_name)
        encoded = _encode_kwargs(self.kwargs)
        if encoded:
            with writer.group("kwargs"):
                writer.write_dict(encoded)
        # ``saved_kwargs`` (schema_version >= 4): persisted Edit mode
        # snapshot so the Restore button keeps working after save/reload.
        # Skipped (group omitted) when no pending edits exist, keeping the
        # on-disk layout byte-identical to schema v3 in the common case.
        if self._saved_kwargs is not None:
            encoded_saved = _encode_kwargs(self._saved_kwargs)
            # Write the group unconditionally (even when empty) so that the
            # round-trip preserves the distinction between None (no pending
            # edits) and {} (degenerate empty snapshot, keeps has_pending_edits).
            with writer.group("saved_kwargs"):
                writer.write_dict(encoded_saved)
        # Only emit ``output_uuids`` when non-empty: the schema_version field
        # already distinguishes v2 (with bijective mapping) from legacy v1.
        # Empty lists are skipped to avoid h5py edge cases with empty arrays.
        if self.output_uuids:
            with writer.group("output_uuids"):
                writer.write(list(self.output_uuids))
        # ``plugin_origin`` (schema_version >= 3): stored as a JSON string so
        # the HDF5 schema stays trivially round-trippable. Skipped when None
        # to keep built-in-only sessions byte-identical to schema v2.
        if self.plugin_origin is not None:
            with writer.group("plugin_origin"):
                writer.write(json.dumps(self.plugin_origin))
        with writer.group("state"):
            self.state.serialize(writer)
        with writer.group("dtstr"):
            writer.write(self.dtstr)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this action."""
        self.schema_version = reader.read(
            "schema_version", default=HISTORY_SCHEMA_VERSION
        )
        with reader.group("kind"):
            self.kind = reader.read_any()
        with reader.group("title"):
            self.__title = reader.read_any()
        # Optional descriptors are written conditionally; check existence in
        # the underlying HDF5 group before reading to avoid leaking ``__seq``
        # frames on the option stack via guidata's read_any fallback path.
        current = reader.h5
        for option in reader.option:
            current = current.require_group(option)
        for attr in ("panel_str", "func_name", "pattern", "target", "method_name"):
            if attr in current.attrs or attr in current:
                with reader.group(attr):
                    setattr(self, attr, reader.read_any())
            else:
                setattr(self, attr, None)
        if "kwargs" in current.attrs or "kwargs" in current:
            with reader.group("kwargs"):
                raw = reader.read_dict()
            self.kwargs = _decode_kwargs(raw)
        else:
            self.kwargs = {}
        # ``saved_kwargs`` was introduced in schema_version=4. Legacy
        # actions (v1, v2, v3) leave it as ``None`` — no Edit mode
        # snapshot to restore.
        if "saved_kwargs" in current.attrs or "saved_kwargs" in current:
            with reader.group("saved_kwargs"):
                raw_saved = reader.read_dict()
            self._saved_kwargs = _decode_kwargs(raw_saved)
        else:
            self._saved_kwargs = None
        # ``output_uuids`` was introduced in schema_version=2. Legacy actions
        # leave it empty; consumers fall back to the heuristic matcher.
        if "output_uuids" in current.attrs or "output_uuids" in current:
            with reader.group("output_uuids"):
                raw_outputs = reader.read_any()
            if raw_outputs is None:
                self.output_uuids = []
            else:
                self.output_uuids = [str(u) for u in raw_outputs]
        else:
            self.output_uuids = []
        # ``plugin_origin`` was introduced in schema_version=3. Legacy actions
        # (v1, v2) leave it as ``None``: a subsequent replay of a missing
        # plugin function will then surface a generic ``FeatureNotFoundError``
        # instead of the richer plugin-aware warning.
        if "plugin_origin" in current.attrs or "plugin_origin" in current:
            with reader.group("plugin_origin"):
                raw_origin = reader.read_any()
            if raw_origin in (None, ""):
                self.plugin_origin = None
            else:
                try:
                    self.plugin_origin = json.loads(raw_origin)
                except (TypeError, ValueError):
                    _logger.warning(
                        "Failed to decode plugin_origin for action %s; "
                        "falling back to None.",
                        self.uuid,
                    )
                    self.plugin_origin = None
        else:
            self.plugin_origin = None
        with reader.group("state"):
            self.state.deserialize(reader)
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()


class WorkspaceState:
    """Object representing the workspace state at a given time.

    The workspace state stores the per-panel selection of objects by **UUID**
    (robust against reordering, renaming or interleaved insertions). For
    informative display, it also retains the data shape and title of each
    selected object at the time of capture.
    """

    def __init__(self) -> None:
        """Create a new workspace state"""
        # The selection is stored as a dictionary where the key is the panel name
        # and the value is the list of UUIDs of selected objects.
        self.selection: dict[str, list[str]] = {}
        # The states are stored as a dictionary where the key is the panel name
        # and the value is the list of states (str) of the objects in the panel. The
        # state is a string containing the object data shape (kept for informative
        # display only -- not used for selection matching anymore).
        self.states: dict[str, list[str]] = {}
        # The titles are stored as a dictionary where the key is the panel name and the
        # value is the list of titles of the objects in the panel. The title is only
        # informative and is not used to determine if two objects have the same state.
        self.titles: dict[str, list[str]] = {}
        # Structured data signatures of selected objects, keyed by panel name and UUID.
        # This is the current schema used for compatibility checks. Missing metadata
        # means a pre-Gate-2 history and falls back to UUID-existence validation.
        self.object_metadata: dict[str, dict[str, dict[str, Any]]] = {}

    def copy(self) -> WorkspaceState:
        """Return an independent copy of this workspace state."""
        state = WorkspaceState()
        state.selection = deepcopy(self.selection)
        state.states = deepcopy(self.states)
        state.titles = deepcopy(self.titles)
        state.object_metadata = deepcopy(self.object_metadata)
        return state

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this workspace state

        Args:
            writer: Writer
        """
        with writer.group("selection"):
            writer.write_dict(self.selection)
        with writer.group("states"):
            writer.write_dict(self.states)
        with writer.group("titles"):
            writer.write_dict(self.titles)
        with writer.group("object_metadata"):
            writer.write_dict(self.object_metadata)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this workspace state

        Args:
            reader: Reader
        """
        with reader.group("selection"):
            self.selection = reader.read_dict()
        with reader.group("states"):
            self.states = reader.read_dict()
        with reader.group("titles"):
            self.titles = reader.read_dict()
        current = reader.h5
        for option in reader.option:
            current = current[option]
        if "object_metadata" in current.attrs or "object_metadata" in current:
            with reader.group("object_metadata"):
                self.object_metadata = reader.read_dict()
        else:
            self.object_metadata = {}
        # Normalize legacy translated keys to stable panel identifiers.
        self.selection = self._normalize_panel_keys(self.selection)
        self.states = self._normalize_panel_keys(self.states)
        self.titles = self._normalize_panel_keys(self.titles)
        self.object_metadata = self._normalize_panel_keys(self.object_metadata)

    def get_current_selection(self, mainwindow: DLMainWindow) -> dict[str, list[str]]:
        """Get the current selection in the workspace, keyed by panel name and
        valued by the list of selected object UUIDs.

        Args:
            mainwindow: DataLab's main window

        Returns:
            Current selection in the workspace, by panel name → list of UUIDs.
        """
        selection: dict[str, list[str]] = {}
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            selection[panel.PANEL_STR_ID] = [
                get_uuid(obj)
                for obj in panel.objview.get_sel_objects(include_groups=True)
            ]
        return selection

    @staticmethod
    def get_object_metadata(obj: Any) -> dict[str, Any]:
        """Return a stable data signature for an object."""
        data = getattr(obj, "data", None)
        shape = getattr(data, "shape", None)
        if shape is None:
            return {}
        shape = [int(size) for size in shape]
        ndim = getattr(data, "ndim", len(shape))
        return {"shape": shape, "ndim": int(ndim)}

    @staticmethod
    def _normalize_object_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Normalize object metadata loaded from HDF5 for comparison."""
        shape = metadata.get("shape")
        if shape is None:
            return {}
        shape = [int(size) for size in shape]
        ndim = metadata.get("ndim", len(shape))
        return {"shape": shape, "ndim": int(ndim)}

    # Mapping from legacy translated panel keys to stable identifiers.
    # Covers the English translations; other locales are handled by the
    # catch-all ``"signal"``/``"image"`` substring heuristic below.
    _LEGACY_PANEL_KEY_MAP: dict[str, str] = {
        "Signal Panel": "signal",
        "Image Panel": "image",
    }

    @classmethod
    def _normalize_panel_key(cls, key: str) -> str:
        """Map a potentially translated panel key to its stable identifier."""
        if key in ("signal", "image"):
            return key
        mapped = cls._LEGACY_PANEL_KEY_MAP.get(key)
        if mapped is not None:
            return mapped
        # Heuristic for non-English translations: look for the stable ID
        # substring in the key (e.g. "Panneau signal" → "signal").
        lowered = key.lower()
        for stable_id in ("signal", "image"):
            if stable_id in lowered:
                return stable_id
        return key

    @classmethod
    def _normalize_panel_keys(cls, d: dict) -> dict:
        """Return *d* with all top-level keys normalized to stable panel IDs."""
        return {cls._normalize_panel_key(k): v for k, v in d.items()}

    def save(self, mainwindow: DLMainWindow) -> None:
        """Save the current workspace state

        Args:
            mainwindow: DataLab's main window
        """
        self.selection = self.get_current_selection(mainwindow)
        self.object_metadata = {}
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            sel_uuids = self.selection[panel.PANEL_STR_ID]
            self.states[panel.PANEL_STR_ID] = [
                str(obj.data.shape)
                for obj in panel.objmodel
                if get_uuid(obj) in sel_uuids
            ]
            self.titles[panel.PANEL_STR_ID] = [
                obj.title for obj in panel.objmodel if get_uuid(obj) in sel_uuids
            ]
            # Store metadata for ALL panel objects (not just selected) so that
            # the dict key order captures the full panel ordering.  During
            # session replay the key order lets us sort old UUIDs by their
            # original panel position, which prevents non-commutative 2_to_1
            # operand swaps in the positional-fallback code path.
            # ``is_current_state_compatible`` only checks *selected* UUIDs, so
            # the extra entries are harmless for compatibility validation.
            self.object_metadata[panel.PANEL_STR_ID] = {
                get_uuid(obj): self.get_object_metadata(obj)
                for obj in panel.objmodel
            }

    def is_current_state_compatible(  # pylint: disable=unused-argument
        self, mainwindow: DLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state.

        Compatibility means that **every** UUID recorded in the saved selection
        still exists in the corresponding panel. When structured object metadata
        is available (current schema), each selected object's data shape and
        dimensions must also match the saved signature. Histories without this
        metadata fall back to legacy UUID-existence validation.

        Args:
            mainwindow: DataLab's main window
            restore_selection: Unused (kept for API symmetry). With UUID-based
             identity, the compatibility check no longer depends on the current
             selection -- it only depends on object existence.

        Returns:
            True if every saved UUID still exists in its panel and saved
            metadata, when available, still matches.
        """
        if not self.selection:
            return True
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            saved_uuids = self.selection.get(panel.PANEL_STR_ID, [])
            existing_uuids = set(panel.objmodel.get_object_ids())
            saved_metadata = self.object_metadata.get(panel.PANEL_STR_ID, {})
            for uuid in saved_uuids:
                if uuid not in existing_uuids:
                    return False
                if uuid in saved_metadata:
                    current = self.get_object_metadata(panel.objmodel[uuid])
                    current = self._normalize_object_metadata(current)
                    saved = self._normalize_object_metadata(saved_metadata[uuid])
                    if saved and current != saved:
                        return False
        return True

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the workspace state by selecting the recorded UUIDs.

        Args:
            mainwindow: DataLab's main window

        Raises:
            ValueError: If at least one of the saved UUIDs no longer exists in
             its panel.
        """
        if not self.selection:
            return
        if not self.is_current_state_compatible(mainwindow, False):
            raise ValueError(
                "Current workspace state is not compatible with saved state"
            )
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            uuids = self.selection.get(panel.PANEL_STR_ID, [])
            if uuids:
                panel.objview.select_objects(uuids)


class HistorySession:
    """Object representing a history session, i.e. a list of actions.

    A history session is a list of actions that can be replayed in the same order
    as they were added to the history session. The history session can be saved to
    a file and loaded from a file.

    Args:
        title: Title of the history session
        number: Number of the history session
    """

    def __init__(self, title: str = "", number: int = 0) -> None:
        """Create a new history session"""
        prefix = _("Session")
        self.title = title if title else f"{prefix} {number:03d}"
        self.number = number
        self.dtstr: str = get_datetime_str()
        self.actions: list[HistoryAction] = []
        self.schema_version: int = HISTORY_SCHEMA_VERSION

    def add_action(self, action: HistoryAction) -> None:
        """Add an action to the history session

        Args:
            action: Action to add
        """
        self.actions.append(action)

    def copy(
        self, title: str | None = None, action_title_suffix: str | None = None
    ) -> HistorySession:
        """Return an independent copy of this history session."""
        session = HistorySession(title=title or self.title, number=self.number)
        session.actions = [
            action.copy(title_suffix=action_title_suffix) for action in self.actions
        ]
        return session

    def copy_with_uuid_remap(
        self, title: str, uuid_remap: dict[str, dict[str, str]]
    ) -> HistorySession:
        """Return a copy of this session with all UUIDs rewritten via ``uuid_remap``.

        Used by the Duplicate operation to build an independent session whose
        captured object references point to the cloned data objects.

        Args:
            title: Title for the new session.
            uuid_remap: Per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``.

        Returns:
            A new :class:`HistorySession` with all captured UUIDs remapped.
        """
        session = HistorySession(title=title, number=self.number)
        session.actions = [
            action.copy_with_uuid_remap(uuid_remap) for action in self.actions
        ]
        return session

    def is_current_state_compatible(
        self, mainwindow: DLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the selection before checking the state

        Returns:
            bool: True if the current workspace state is compatible with the saved state
        """
        if self.actions:
            return self.actions[0].is_current_state_compatible(
                mainwindow, restore_selection
            )
        return True

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the state of the workspace associated to the first action of session

        Args:
            mainwindow: DataLab's main window
        """
        if self.actions:
            self.actions[0].restore(mainwindow)

    def replay(
        self, mainwindow: DLMainWindow, restore_selection: bool, edit: bool
    ) -> None:
        """Replay the history session

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the workspace selection before replaying
            edit: if True, always open the dialog boxes to edit parameters, if False,
             use the parameters passed when creating the action
        """
        # Per-panel ``{old_uuid: new_uuid}`` mapping, populated as UI actions
        # create new objects. Used by compute actions to translate their
        # captured selection (and ``obj2_uuids``) into the freshly-created
        # UUIDs of the current replay, so chained ``n_to_1`` / ``2_to_1`` /
        # ``1_to_n`` actions operate on the correct inputs. Keys are
        # ``panel.PANEL_STR_ID`` (matches ``WorkspaceState.selection`` keys).
        panels = (mainwindow.signalpanel, mainwindow.imagepanel)
        uuid_remap: dict[str, dict[str, str]] = {p.PANEL_STR_ID: {} for p in panels}
        # FIFO of newly-created UUIDs not yet claimed by a remap entry --
        # required because most creation UI actions (e.g. ``new_signal``)
        # are recorded with ``save_state=False`` (empty captured selection),
        # so we cannot pair captured-vs-new UUIDs by position at UI time.
        # Subsequent compute actions claim from this queue on demand.
        unclaimed: dict[str, list[str]] = {p.PANEL_STR_ID: [] for p in panels}
        def _claim_unmapped(
            pstr: str,
            old_uuids: list[str],
            action: HistoryAction,
        ) -> None:
            """Claim unclaimed new UUIDs for *old_uuids* not yet in uuid_remap.

            Uses title matching (scanning the full unclaimed queue) followed by
            panel-order index alignment to deterministically pair old UUIDs
            to the correct new UUIDs, regardless of creation order.
            """
            # Collect unmapped UUIDs (deduplicated, preserving first-seen order).
            all_unmapped: list[str] = []
            seen: set[str] = set()
            for u in old_uuids:
                if u not in seen and u not in uuid_remap.get(pstr, {}):
                    all_unmapped.append(u)
                    seen.add(u)
            if not all_unmapped:
                return
            # Re-sort by recorded panel position when available.
            panel_order = list(
                action.state.object_metadata.get(pstr, {}).keys()
            )
            if panel_order and all(u in panel_order for u in all_unmapped):
                all_unmapped.sort(key=panel_order.index)
            queue = unclaimed.get(pstr) or []
            if not queue:
                return
            # Build old UUID → title from captured state and object_metadata.
            sel_uuids = action.state.selection.get(pstr, [])
            sel_titles = action.state.titles.get(pstr, [])
            old_titles: dict[str, str] = {}
            for _u, _t in zip(sel_uuids, sel_titles):
                if _u in seen:
                    old_titles[_u] = _t
            obj_meta = action.state.object_metadata.get(pstr, {})
            for _u in all_unmapped:
                if _u not in old_titles and _u in obj_meta:
                    meta = obj_meta[_u]
                    if isinstance(meta, dict) and "title" in meta:
                        old_titles[_u] = meta["title"]
            # Build new UUID → title from the live panel (full queue).
            new_titles: dict[str, str] = {}
            panel_obj = None
            for p in panels:
                if p.PANEL_STR_ID == pstr:
                    panel_obj = p
                    break
            if panel_obj is not None:
                for nu in queue:
                    try:
                        new_titles[nu] = panel_obj.objmodel[nu].title
                    except KeyError:
                        pass
            # Phase 1: title matching against the FULL queue.
            assigned_old: set[str] = set()
            assigned_new: set[str] = set()
            for ou in all_unmapped:
                if ou not in old_titles:
                    continue
                title = old_titles[ou]
                candidates = [
                    nu
                    for nu in queue
                    if nu not in assigned_new
                    and new_titles.get(nu) == title
                ]
                if len(candidates) == 1:
                    uuid_remap.setdefault(pstr, {})[ou] = candidates[0]
                    assigned_old.add(ou)
                    assigned_new.add(candidates[0])
            # Phase 2: positional fallback using panel-order alignment.
            # Two modes depending on whether the remaining queue covers all
            # free recorded panel slots:
            #
            # A) Absolute index alignment (len(rem_queue) == len(free_indices)):
            #    Each free panel_order index maps 1-to-1 to a queue slot.
            #    This ensures e.g. the second-created object maps to the
            #    second queue entry even when only a subset of old UUIDs
            #    needs claiming.
            #
            # B) Relative order fallback (queue is a strict subset):
            #    The queue only contains later compute-created objects while
            #    earlier full-panel entries are absent.  Absolute alignment
            #    would leave non-first old UUIDs unmapped.  Instead, zip
            #    rem_old (already sorted by panel order) with rem_queue
            #    sequentially.
            rem_old = [u for u in all_unmapped if u not in assigned_old]
            if rem_old and panel_order:
                rem_queue = [u for u in queue if u not in assigned_new]
                # Find which panel_order indices are "free" (unclaimed).
                free_indices: list[int] = []
                for idx, po_uuid in enumerate(panel_order):
                    if po_uuid not in uuid_remap.get(pstr, {}):
                        if po_uuid not in assigned_old:
                            free_indices.append(idx)
                if len(rem_queue) == len(free_indices):
                    # Mode A: absolute index alignment.
                    idx_to_new: dict[int, str] = {}
                    for qi, fi in enumerate(free_indices):
                        if qi < len(rem_queue):
                            idx_to_new[fi] = rem_queue[qi]
                    for ou in rem_old:
                        if ou in panel_order:
                            idx = panel_order.index(ou)
                            if idx in idx_to_new:
                                nu = idx_to_new[idx]
                                uuid_remap.setdefault(pstr, {})[ou] = nu
                                assigned_new.add(nu)
                else:
                    # Mode B: relative order fallback.
                    for ou, nu in zip(rem_old, rem_queue):
                        uuid_remap.setdefault(pstr, {})[ou] = nu
                        assigned_new.add(nu)
            elif rem_old:
                # No panel_order available: sequential fallback.
                rem_queue = [u for u in queue if u not in assigned_new]
                for ou, nu in zip(rem_old, rem_queue):
                    uuid_remap.setdefault(pstr, {})[ou] = nu
                    assigned_new.add(nu)
            # Remove all assigned new UUIDs from the unclaimed queue.
            if assigned_new:
                unclaimed[pstr] = [u for u in queue if u not in assigned_new]

        for action in self.actions[:]:
            before = {p.PANEL_STR_ID: set(p.objmodel.get_object_ids()) for p in panels}
            if action.kind == HistoryAction.KIND_COMPUTE:
                # Lazy-resolve any captured UUIDs missing from the remap by
                # claiming from ``unclaimed`` (deterministic: title + panel-order).
                pstr = action.panel_str or ""
                captured = action.state.selection.get(pstr, [])
                if action.pattern == "2_to_1":
                    # For 2_to_1: collect ALL unmapped old UUIDs from both
                    # captured selection and obj2_uuids in one batch so
                    # operand order is preserved by the helper.
                    obj2 = action.kwargs.get("obj2_uuids") or []
                    if isinstance(obj2, str):
                        obj2 = [obj2]
                    _claim_unmapped(pstr, list(obj2) + list(captured), action)
                else:
                    # For all other compute patterns (1_to_1, n_to_1, etc.):
                    # use the same deterministic helper.
                    _claim_unmapped(pstr, list(captured), action)
            action.replay(
                mainwindow,
                restore_selection=restore_selection,
                edit=edit,
                uuid_remap=uuid_remap,
            )
            # Post-action bookkeeping: track new/removed UUIDs for *every*
            # action kind so that later actions consuming compute-created
            # outputs can resolve them through ``uuid_remap`` / ``unclaimed``.
            for panel in panels:
                pstr = panel.PANEL_STR_ID
                current_ids = set(panel.objmodel.get_object_ids())
                new_uuids = [
                    u
                    for u in panel.objmodel.get_object_ids()
                    if u not in before[pstr]
                ]
                # Drop vanished UUIDs from the unclaimed queue and the
                # reverse remap entries (e.g. ``Remove selected objects``):
                # this keeps the FIFO claim in sync with the live panel
                # contents during chained creation/removal replays.
                removed_uuids = before[pstr] - current_ids
                if removed_uuids:
                    unclaimed[pstr] = [
                        u for u in unclaimed.get(pstr, []) if u not in removed_uuids
                    ]
                    panel_map = uuid_remap.get(pstr, {})
                    for old_key in [
                        k for k, v in panel_map.items() if v in removed_uuids
                    ]:
                        panel_map.pop(old_key, None)
                if not new_uuids:
                    continue
                if action.kind == HistoryAction.KIND_UI:
                    captured = action.state.selection.get(pstr, [])
                    if captured:
                        # Captured post-action selection available: pair
                        # captured UUIDs with new UUIDs by position.
                        for old_uuid, new_uuid in zip(captured, new_uuids):
                            uuid_remap.setdefault(pstr, {})[old_uuid] = new_uuid
                        # Any extra newly-created UUIDs go to the queue.
                        unclaimed.setdefault(pstr, []).extend(
                            new_uuids[len(captured) :]
                        )
                    else:
                        # No captured selection (typical of ``new_signal``):
                        # queue all new UUIDs for lazy claiming.
                        unclaimed.setdefault(pstr, []).extend(new_uuids)
                else:
                    # Compute actions: queue all newly-created UUIDs so
                    # later actions can lazily claim them.  Do NOT map
                    # captured input UUIDs to output UUIDs — compute
                    # inputs and outputs are semantically different.
                    unclaimed.setdefault(pstr, []).extend(new_uuids)

        # Visually close the replay: select the output of the last compute
        # action so the user sees the final result highlighted in the panel.
        # Without this, the very last action's output is never selected
        # (intermediate actions are implicitly "closed" by the next
        # iteration's input restore).
        if self.actions:
            last = self.actions[-1]
            if last.kind == HistoryAction.KIND_COMPUTE:
                hpanel = getattr(mainwindow, "historypanel", None)
                if hpanel is not None:
                    output_uuid = hpanel._action_output_uuid(last)
                    if output_uuid:
                        panel_str = last.panel_str or ""
                        panel_map = uuid_remap.get(panel_str, {})
                        mapped_uuid = panel_map.get(output_uuid, output_uuid)
                        for panel in panels:
                            if panel.PANEL_STR_ID == panel_str:
                                try:
                                    panel.objview.select_objects([mapped_uuid])
                                except KeyError:
                                    pass
                                break

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this history session

        Args:
            writer: Writer
        """
        with writer.group("schema_version"):
            writer.write(self.schema_version)
        with writer.group("title"):
            writer.write(self.title)
        with writer.group("number"):
            writer.write(self.number)
        with writer.group("dtstr"):
            writer.write(self.dtstr)
        writer.write_object_list(self.actions, "actions")

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this history session

        Args:
            reader: Reader
        """
        self.schema_version = reader.read(
            "schema_version", default=HISTORY_SCHEMA_VERSION
        )
        with reader.group("title"):
            self.title = reader.read_any()
        with reader.group("number"):
            self.number = reader.read_any()
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()
        self.actions = reader.read_object_list("actions", HistoryAction)

    def remove_action(self, action: HistoryAction) -> None:
        """Remove an action from the history session

        This implies removing all subsequent actions. If action is not found, this
        fails silently.

        Args:
            action: Action to remove
        """
        if action in self.actions:
            index = self.actions.index(action)
            self.actions = self.actions[:index]


class CollapsibleDescriptionWidget(QW.QWidget):
    """Compact, expandable cell widget for the history Description column.

    Shows a single-line summary by default; a chevron toggle reveals the full
    HTML description (mirroring the *Properties* tab rendering).
    """

    toggled = QC.Signal(bool)

    def __init__(
        self,
        summary: str,
        html_text: str,
        expanded: bool = False,
        parent: QW.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._summary = summary
        self._html = html_text
        self._expanded = expanded

        self._toggle = QW.QToolButton(self)
        self._toggle.setAutoRaise(True)
        self._toggle.setCheckable(True)
        self._toggle.setFocusPolicy(QC.Qt.NoFocus)
        self._toggle.setArrowType(QC.Qt.RightArrow)
        self._toggle.setToolTip(_("Show details"))

        self._label = QW.QLabel(self)
        self._label.setTextFormat(QC.Qt.RichText)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(QC.Qt.TextSelectableByMouse)
        self._label.setAlignment(QC.Qt.AlignTop | QC.Qt.AlignLeft)

        layout = QW.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toggle, 0, QC.Qt.AlignTop)
        layout.addWidget(self._label, 1)

        # Hide the toggle when there is nothing more to show than the summary.
        if not self._html or self._html_matches_summary():
            self._toggle.setVisible(False)

        self._toggle.toggled.connect(self._on_toggled)
        self._refresh()

    def _html_matches_summary(self) -> bool:
        """Return True when the HTML rendering would not add information."""
        return self._html.strip() == html.escape(self._summary).strip()

    def _on_toggled(self, checked: bool) -> None:
        self._expanded = checked
        self._refresh()
        self.toggled.emit(checked)

    def _refresh(self) -> None:
        if self._expanded:
            self._toggle.setArrowType(QC.Qt.DownArrow)
            self._toggle.setToolTip(_("Hide details"))
            self._label.setText(self._html or html.escape(self._summary))
        else:
            self._toggle.setArrowType(QC.Qt.RightArrow)
            self._toggle.setToolTip(_("Show details"))
            self._label.setText(html.escape(self._summary))
        self.updateGeometry()

    def is_expanded(self) -> bool:
        """Return current expanded state."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Programmatically set the expanded state."""
        if expanded == self._expanded:
            return
        self._toggle.setChecked(expanded)


class HistoryTree(QW.QTreeWidget):
    """Tree widget for the history panel"""

    DESCRIPTION_COLUMN = 2
    COMPATIBILITY_ROLE = QC.Qt.UserRole + 1

    def __init__(self, parent: QW.QWidget) -> None:
        """Create a new history tree widget"""
        super().__init__(parent)
        self.setHeaderLabels([_("Title"), _("Date and time"), _("Description")])
        self.setContextMenuPolicy(QC.Qt.CustomContextMenu)
        self.setSelectionMode(QW.QAbstractItemView.ContiguousSelection)
        self.setUniformRowHeights(False)
        header = self.header()
        header.setSectionResizeMode(self.DESCRIPTION_COLUMN, QW.QHeaderView.Stretch)
        # Per-action expanded state, preserved across repopulate (delete/replay).
        self.__expanded_state: dict[str, bool] = {}

    def __on_description_toggled(self, uuid: str, expanded: bool) -> None:
        """Remember the expanded state of a description cell."""
        self.__expanded_state[uuid] = expanded
        # Force the tree to recompute row heights now that the label content
        # has changed.
        self.scheduleDelayedItemsLayout()

    def __install_description_widget(
        self, item: QW.QTreeWidgetItem, action: HistoryAction
    ) -> None:
        """Attach the collapsible description widget to ``item`` (column 2).

        The item must already be inserted in the tree before calling this.
        """
        expanded = self.__expanded_state.get(action.uuid, False)
        widget = CollapsibleDescriptionWidget(
            action.description_summary,
            action.description_html,
            expanded=expanded,
            parent=self,
        )
        widget.toggled.connect(
            lambda checked, uuid=action.uuid: self.__on_description_toggled(
                uuid, checked
            )
        )
        # Clear any text the item may carry for that column to avoid double
        # rendering behind the widget.
        item.setText(self.DESCRIPTION_COLUMN, "")
        self.setItemWidget(item, self.DESCRIPTION_COLUMN, widget)

    @classmethod
    def action_to_tree_item(cls, action: HistoryAction) -> QW.QTreeWidgetItem:
        """Convert an action to a tree item

        Args:
            action: Action to convert

        Returns:
            QW.QTreeWidgetItem: Tree item
        """
        # Description column is left empty: a CollapsibleDescriptionWidget is
        # installed by ``HistoryTree`` once the item is inserted in the tree.
        item = QW.QTreeWidgetItem([action.title, action.dtstr, ""])
        item.setData(0, QC.Qt.UserRole, action.uuid)
        item.setData(0, cls.COMPATIBILITY_ROLE, True)
        return item

    def update_compatibility_states(
        self, history_sessions: list[HistorySession], mainwindow: DLMainWindow
    ) -> None:
        """Update action item visual state from workspace compatibility."""
        default_brush = QG.QBrush()
        disabled_brush = QG.QBrush(
            self.palette().color(QG.QPalette.Disabled, QG.QPalette.Text)
        )
        compatible_tip = _("Action is compatible with the current workspace state.")
        incompatible_tip = _(
            "Action is not compatible with the current workspace state."
        )
        for i in range(self.topLevelItemCount()):
            session_item = self.topLevelItem(i)
            for j in range(session_item.childCount()):
                child = session_item.child(j)
                uuid = child.data(0, QC.Qt.UserRole)
                action = self.get_action_from_uuid(uuid, history_sessions)
                compatible = action.is_current_state_compatible(
                    mainwindow, restore_selection=True
                )
                child.setData(0, self.COMPATIBILITY_ROLE, compatible)
                brush = default_brush if compatible else disabled_brush
                icon = get_icon("apply.svg") if compatible else get_icon("delete.svg")
                child.setIcon(0, icon)
                for col in range(self.columnCount()):
                    child.setForeground(col, brush)
                    child.setToolTip(
                        col, compatible_tip if compatible else incompatible_tip
                    )

    def __forget_orphan_expanded_states(
        self, history_sessions: list[HistorySession]
    ) -> None:
        """Drop expanded-state entries for actions that no longer exist."""
        live_uuids = {
            action.uuid for session in history_sessions for action in session.actions
        }
        self.__expanded_state = {
            uuid: state
            for uuid, state in self.__expanded_state.items()
            if uuid in live_uuids
        }

    def populate_tree(self, history_sessions: list[HistorySession]) -> None:
        """Populate the history tree widget

        Args:
            history_sessions: List of history sessions
        """
        self.__forget_orphan_expanded_states(history_sessions)
        self.clear()
        for session in history_sessions:
            ritem = QW.QTreeWidgetItem([session.title, session.dtstr])
            ritem.setData(0, self.COMPATIBILITY_ROLE, True)
            self.addTopLevelItem(ritem)
            for action in session.actions:
                child = self.action_to_tree_item(action)
                ritem.addChild(child)
                self.__install_description_widget(child, action)
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def rearrange_tree(self) -> None:
        """Rearrange the history tree widget"""
        self.expandAll()
        for col in (0, 1):
            self.resizeColumnToContents(col)

    def add_action_to_tree(self, action: HistoryAction) -> None:
        """Add an action to the history tree widget

        Args:
            action: Action to add
        """
        item = self.action_to_tree_item(action)
        ritem = self.topLevelItem(self.topLevelItemCount() - 1)
        ritem.addChild(item)
        self.__install_description_widget(item, action)

    def refresh_action_item(self, action: HistoryAction) -> None:
        """Refresh the tree item corresponding to ``action``.

        Re-installs the description widget so it reflects the current
        ``action.kwargs`` (e.g. after the user edited a ``param`` via the
        Processing tab of the Signal/Image panel). Also applies a light
        orange background when ``action.is_stale`` is True, to signal that
        the action is currently being recomputed in a cascade.
        """
        target_uuid = action.uuid
        stale_brush = QG.QBrush(QG.QColor(255, 220, 150))  # light orange
        normal_brush = QG.QBrush()
        iterator = QW.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QC.Qt.UserRole) == target_uuid:
                # Remove and re-install the collapsible description widget so
                # it reflects the mutated ``action.kwargs``.
                self.removeItemWidget(item, self.DESCRIPTION_COLUMN)
                self.__install_description_widget(item, action)
                brush = stale_brush if action.is_stale else normal_brush
                for col in range(self.columnCount()):
                    item.setBackground(col, brush)
                self.scheduleDelayedItemsLayout()
                return
            iterator += 1

    def get_action_from_uuid(
        self, uuid: str, history_sessions: list[HistorySession]
    ) -> HistoryAction:
        """Get the action from its UUID

        Args:
            uuid: Action UUID
            history_sessions: List of history sessions

        Returns:
            HistoryAction: Action
        """
        for session in history_sessions:
            for action in session.actions:
                if action.uuid == uuid:
                    return action
        raise ValueError("Action not found")

    def get_selected_actions_or_sessions(
        self, history_sessions: list[HistorySession]
    ) -> list[HistoryAction | HistorySession]:
        """Get the selected actions or sessions

        Args:
            history_sessions: List of history sessions

        Returns:
            list[HistoryAction | HistorySession]: List of selected actions or sessions
        """
        selected: list[HistoryAction | HistorySession] = []
        for item in self.selectedItems():
            if item.parent() is None:
                index = self.indexOfTopLevelItem(item)
                selected.append(history_sessions[index])
            else:
                uuid = item.data(0, QC.Qt.UserRole)
                selected.append(self.get_action_from_uuid(uuid, history_sessions))
        return selected

    def get_selected_actions(
        self, history_sessions: list[HistorySession]
    ) -> list[HistoryAction]:
        """Get the selected actions

        Args:
            history_sessions: List of history sessions

        Returns:
            list[HistoryAction]: List of selected actions
        """
        selected: list[HistoryAction] = []
        for item in self.selectedItems():
            if item.parent() is not None:
                uuid = item.data(0, QC.Qt.UserRole)
                selected.append(self.get_action_from_uuid(uuid, history_sessions))
        return selected


class WorkspaceStateWidget(QW.QWidget):
    """Side-by-side tables showing the workspace state captured by a history action.

    Left table: signals (title + data shape).
    Right table: images (title + data shape/dimensions).
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(parent)
        self._signal_table = QW.QTableWidget(0, 2, self)
        self._signal_table.setHorizontalHeaderLabels([_("Signal"), _("Shape")])
        self._signal_table.horizontalHeader().setStretchLastSection(True)
        self._signal_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self._signal_table.setSelectionMode(QW.QAbstractItemView.NoSelection)
        self._signal_table.verticalHeader().hide()

        self._image_table = QW.QTableWidget(0, 2, self)
        self._image_table.setHorizontalHeaderLabels([_("Image"), _("Dimensions")])
        self._image_table.horizontalHeader().setStretchLastSection(True)
        self._image_table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self._image_table.setSelectionMode(QW.QAbstractItemView.NoSelection)
        self._image_table.verticalHeader().hide()

        splitter = QW.QSplitter(QC.Qt.Horizontal, self)
        splitter.addWidget(self._signal_table)
        splitter.addWidget(self._image_table)
        layout = QW.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def update_from_state(self, state: WorkspaceState | None) -> None:
        """Populate tables from a WorkspaceState."""
        self._signal_table.setRowCount(0)
        self._image_table.setRowCount(0)
        if state is None:
            return
        self._populate_table(self._signal_table, state, "signal")
        self._populate_table(self._image_table, state, "image")

    @staticmethod
    def _populate_table(
        table: QW.QTableWidget, state: WorkspaceState, panel_key: str
    ) -> None:
        """Fill a table from the state for a given panel key."""
        titles = state.titles.get(panel_key, [])
        shapes = state.states.get(panel_key, [])
        metadata = state.object_metadata.get(panel_key, {})
        uuids = state.selection.get(panel_key, [])
        # Use metadata keyed by UUID when available
        rows: list[tuple[str, str]] = []
        for i, uuid in enumerate(uuids):
            title = titles[i] if i < len(titles) else uuid[:8]
            meta = metadata.get(uuid, {})
            shape = meta.get("shape")
            if shape is not None:
                shape_str = " × ".join(str(s) for s in shape)
            elif i < len(shapes):
                shape_str = shapes[i]
            else:
                shape_str = "—"
            rows.append((title, shape_str))
        table.setRowCount(len(rows))
        for row_idx, (title, shape_str) in enumerate(rows):
            table.setItem(row_idx, 0, QW.QTableWidgetItem(title))
            table.setItem(row_idx, 1, QW.QTableWidgetItem(shape_str))


class HistoryPanel(AbstractPanel, DockableWidgetMixin):
    """History panel"""

    LOCATION = QC.Qt.RightDockWidgetArea
    PANEL_STR = _("History panel")

    H5_PREFIX = "DataLab_His"

    SIG_OBJECT_MODIFIED = QC.Signal()

    FILE_FILTERS = f"{_('History files')} (*.dlhist)"

    def __init__(self, parent: DLMainWindow) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.PANEL_STR)
        self.setWindowIcon(get_icon("history.svg"))
        self.setOrientation(QC.Qt.Vertical)

        self.__record_mode = False
        self.__edit_mode = False
        # When `__replaying` is True, calls to `add_entry` are silently
        # ignored. This prevents replay/recompute paths from polluting the
        # history with synthetic entries triggered by their own internal
        # `compute_*` calls. Use `replaying()` as a context manager.
        self.__replaying = False
        self.__output_suppressed = False
        # Guard flag used by `__sync_panel_selection` to prevent re-entry
        # while the data panels are being updated programmatically.
        self.__syncing = False
        self._cascade_in_progress = False
        self.__delete_action: QW.QAction | None = None
        self.__duplicate_action: QW.QAction | None = None
        self.__step_prev_action: QW.QAction | None = None
        self.__step_next_action: QW.QAction | None = None
        self.__restore_selection_action: QW.QAction | None = None
        self.__edit_action: QW.QAction | None = None
        self.__menu_actions: list[QW.QAction] = self.__create_menu_actions()

        self.mainwindow = parent
        self.tree = HistoryTree(self)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.replay_restore_actions)
        self.tree.itemSelectionChanged.connect(self.__sync_panel_selection)
        self.tree.itemSelectionChanged.connect(self.__update_actions_state)
        self.tree.itemSelectionChanged.connect(self.__update_state_widget)

        self.__state_widget = WorkspaceStateWidget(self)

        toolbar = QW.QToolBar(self)
        add_actions(toolbar, self.__menu_actions)
        widget = QW.QWidget(self)
        layout = QW.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        layout.addWidget(self.__state_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        self.addWidget(widget)

        self.__history_sessions: list[HistorySession] = []
        self.__session_increment = 0
        # Bijective action ↔ output mapping (C1). Both dicts are kept in sync
        # by :meth:`register_action_outputs` and pruned by
        # :meth:`_prune_output_mapping` when objects are removed from a panel.
        # Reconstructed at HDF5 load time from each action's ``output_uuids``.
        self._action_output_uuids: dict[str, list[str]] = {}
        self._output_to_action: dict[str, str] = {}
        # Warnings collected during ``recompute_cascade`` (and the helpers
        # it dispatches to). Aggregated into a single user-facing dialog
        # at the end of the cascade so deleted output objects / missing
        # sources / unsupported patterns do not spam the user.
        self._cascade_warnings: list[str] = []
        # Actions detected as broken (missing plugin) during a cascade run.
        # Transient — intentionally not persisted: recalculated from scratch
        # on each cascade run, similar to ``is_stale``.
        # The visual stale marker is *not* cleared for these so the user
        # sees them flagged in the tree until the plugin is reinstalled and
        # the cascade is re-run successfully.
        self._broken_actions: set[str] = set()
        # Re-entrancy guard for :meth:`_reconnect_chain_after_removal` (see
        # method docstring): ``recompute_cascade`` pumps the event loop and
        # could deliver a queued ``SIG_OBJECT_REMOVED`` mid-reconnection.
        self.__reconnecting = False
        for panel in (self.mainwindow.signalpanel, self.mainwindow.imagepanel):
            panel.SIG_OBJECT_ADDED.connect(self.refresh_compatibility_items)
            panel.SIG_OBJECT_ADDED.connect(self.__refresh_obj_ids_snapshot)
            panel.SIG_OBJECT_REMOVED.connect(self.refresh_compatibility_items)
            panel.SIG_OBJECT_REMOVED.connect(
                functools.partial(self._reconnect_chain_after_removal, panel)
            )
            panel.SIG_OBJECT_REMOVED.connect(self._prune_output_mapping)
            panel.SIG_OBJECT_MODIFIED.connect(self.refresh_compatibility_items)
        self.__refresh_obj_ids_snapshot()
        self.__update_actions_state()
        self.refresh_compatibility_items()

    def __refresh_obj_ids_snapshot(self) -> None:
        """Cache the current object ids of both data panels.

        ``SIG_OBJECT_REMOVED`` carries no payload, so the set of just-deleted
        objects is recovered by diffing this snapshot against the live model
        inside :meth:`_reconnect_chain_after_removal`.
        """
        self.__obj_ids_snapshot = {
            self.mainwindow.signalpanel.PANEL_STR_ID: set(
                self.mainwindow.signalpanel.objmodel.get_object_ids()
            ),
            self.mainwindow.imagepanel.PANEL_STR_ID: set(
                self.mainwindow.imagepanel.objmodel.get_object_ids()
            ),
        }

    def __update_actions_state(self) -> None:
        """Update the enabled state of menu actions depending on history content."""
        has_history = len(self) > 0
        for action in (
            self.__delete_action,
            self.__duplicate_action,
        ):
            if action is not None:
                action.setEnabled(has_history)
        if self.__step_prev_action is not None:
            self.__step_prev_action.setEnabled(self.__can_step_prev())
        if self.__step_next_action is not None:
            self.__step_next_action.setEnabled(self.__can_step_next())
        if self.__restore_selection_action is not None:
            self.__restore_selection_action.setEnabled(
                self.__edit_mode or self._has_any_pending_edits()
            )

    def _has_any_pending_edits(self) -> bool:
        """Return True if any action across all sessions has a pending
        Edit mode snapshot (i.e. uncommitted edits that Restore can revert).
        """
        return any(
            action.has_pending_edits
            for session in self.__history_sessions
            for action in session.actions
        )

    def __update_state_widget(self) -> None:
        """Update the workspace state widget from the currently selected action."""
        action = self.__current_action()
        if action is not None:
            self.__state_widget.update_from_state(action.state)
        else:
            self.__state_widget.update_from_state(None)

    def __create_menu_actions(self) -> list[QW.QAction]:
        """Create menu actions for the history panel

        Returns:
            list[QW.QAction]: List of menu actions
        """
        edit_action = create_action(
            self,
            _("Edit mode"),
            toggled=self.toggle_edit_mode,
            icon=get_icon("edit_mode.svg"),
        )
        edit_action.setChecked(self.__edit_mode)
        self.__edit_action = edit_action
        record_action = create_action(
            self,
            _("Record mode"),
            toggled=self.toggle_record_mode,
            icon=get_icon("record.svg"),
        )
        record_action.setChecked(self.__record_mode)
        new_session_action = create_action(
            self,
            _("New session"),
            self.create_new_session,
            icon=get_icon("libre-gui-add.svg"),
            tip=_("Start a new history session"),
        )
        open_action = create_action(
            self,
            _("Open history file..."),
            triggered=lambda checked=False: self.open_dlhist_file(),
            icon=get_icon("fileopen_h5.svg"),
            tip=_("Open history from a standalone .dlhist file"),
        )
        save_action = create_action(
            self,
            _("Save history file..."),
            triggered=lambda checked=False: self.save_to_dlhist_file(),
            icon=get_icon("filesave_h5.svg"),
            tip=_("Save history to a standalone .dlhist file"),
        )
        self.__delete_action = create_action(
            self,
            _("Delete"),
            self.delete_selected,
            icon=get_icon("delete.svg"),
        )
        self.__duplicate_action = create_action(
            self,
            _("Duplicate"),
            self.duplicate_selected_entries,
            icon=get_icon("duplicate.svg"),
            tip=_("Duplicate selected history action/session"),
        )
        self.__step_prev_action = create_action(
            self,
            _("Previous step"),
            triggered=self._step_prev,
            icon=get_icon("libre-gui-arrow-left.svg"),
            tip=_("Select the previous action in the current session"),
            shortcut=QG.QKeySequence("Ctrl+Left"),
        )
        self.__step_next_action = create_action(
            self,
            _("Next step"),
            triggered=self._step_next,
            icon=get_icon("libre-gui-arrow-right.svg"),
            tip=_("Select the next action in the current session"),
            shortcut=QG.QKeySequence("Ctrl+Right"),
        )
        generate_macro_action = create_action(
            self,
            _("Generate macro"),
            self.generate_macro,
            icon=get_icon("console.svg"),
            tip=_("Generate a Python macro script from history"),
        )
        remove_incompatible_action = create_action(
            self,
            _("Remove incompatible"),
            self.remove_incompatible_actions,
            icon=get_icon("edit/delete_all.svg"),
            tip=_("Remove actions incompatible with the current workspace"),
        )
        self.__restore_selection_action = create_action(
            self,
            _("Restore parameters"),
            lambda: self.replay_restore_actions(
                restore_selection=True, replay=False
            ),
            icon=get_icon("restore_selection.svg"),
            tip=_("Restore original parameters (discard edit-mode changes)"),
        )
        return [
            record_action,
            new_session_action,
            None,
            open_action,
            save_action,
            None,
            self.__step_prev_action,
            self.__step_next_action,
            None,
            create_action(
                self,
                _("Replay"),
                lambda: self.replay_restore_actions(restore_selection=False),
                icon=get_icon("replay.svg"),
            ),
            self.__restore_selection_action,
            edit_action,
            None,
            self.__duplicate_action,
            generate_macro_action,
            None,
            remove_incompatible_action,
            self.__delete_action,
        ]

    def toggle_edit_mode(self, checked: bool) -> None:
        """Toggle edit mode.

        Toggling Edit mode off is a **definitive commit**: all parameter
        changes performed during the session become permanent and Restore
        is no longer available for them. When pending edits exist, the
        user is asked to confirm; refusing leaves Edit mode enabled.

        Args:
            checked: True if the edit mode is checked, False otherwise.
        """
        if not checked and self._has_any_pending_edits():
            reply = QW.QMessageBox.question(
                self.mainwindow,
                _("Commit edit mode changes?"),
                _(
                    "You are about to exit Edit mode.\n\n"
                    "All parameter changes made during this session will be "
                    "permanently kept.\n"
                    "This action cannot be undone — Restore will no longer "
                    "be available.\n\n"
                    "Do you want to continue?"
                ),
                QW.QMessageBox.Yes | QW.QMessageBox.No,
                QW.QMessageBox.No,
            )
            if reply != QW.QMessageBox.Yes:
                # Re-check the action without triggering toggle_edit_mode
                # again (blockSignals prevents recursion).
                if self.__edit_action is not None:
                    self.__edit_action.blockSignals(True)
                    self.__edit_action.setChecked(True)
                    self.__edit_action.blockSignals(False)
                return
        self.__edit_mode = checked
        if not checked:
            # Exiting edit mode: accept all pending edits (discard snapshots)
            for session in self.__history_sessions:
                for action in session.actions:
                    action.discard_snapshot()
        self.__update_actions_state()

    def toggle_record_mode(self, checked: bool) -> None:
        """Toggle record mode

        Args:
            checked: True if the record mode is checked, False otherwise
        """
        self.__record_mode = checked

    def is_edit_mode(self) -> bool:
        """Return True when the History panel is in edit (parameter testing) mode."""
        return self.__edit_mode

    @contextmanager
    def replaying(self) -> Generator[None, None, None]:
        """Context manager suppressing history capture during its scope.

        Used by replay / recompute paths to avoid double-capture: the
        generic ``compute_*`` methods of the processor would otherwise
        register synthetic entries every time recompute or replay
        triggers them.
        """
        previous = self.__replaying
        self.__replaying = True
        try:
            yield
        finally:
            self.__replaying = previous

    def is_replaying(self) -> bool:
        """Return True when an external replay/recompute is in progress."""
        return self.__replaying

    @contextmanager
    def output_suppressed(self) -> Generator[None, None, None]:
        """Context manager suppressing compute outputs during its scope.

        When active, :meth:`BaseProcessor._add_object_to_appropriate_panel`
        and :meth:`BaseProcessor._create_group_for_result` become no-ops so
        that History Panel replay can execute computations without altering
        Signal/Image panel object counts.  Reentrant-safe.
        """
        previous = self.__output_suppressed
        self.__output_suppressed = True
        try:
            yield
        finally:
            self.__output_suppressed = previous

    def is_output_suppressed(self) -> bool:
        """Return True when compute outputs must not be added to panels."""
        return self.__output_suppressed

    def show_context_menu(self, pos: QC.QPoint) -> None:
        """Show the context menu

        Args:
            pos: Position of the context menu
        """
        self.refresh_compatibility_items()
        menu = QW.QMenu()
        add_actions(menu, self.__menu_actions)
        menu.exec_(self.tree.mapToGlobal(pos))

    def get_action_from_uuid(self, uuid: str) -> HistoryAction:
        """Get the action from its UUID

        Args:
            uuid: Action UUID

        Returns:
            HistoryAction: Action
        """
        for session in self.__history_sessions:
            for action in session.actions:
                if action.uuid == uuid:
                    return action
        raise ValueError("Action not found")

    def replay_restore_actions(
        self, replay: bool = True, restore_selection: bool = True
    ) -> None:
        """Replay and/or restore selection for the selected actions.

        When nothing is selected in the tree, replays the entire last session.
        Replay is non-persistent: compute actions run but their outputs are
        NOT added to Signal/Image panels.  UI-creation actions are always
        skipped during replay because source objects already exist.
        """
        self.refresh_compatibility_items()
        selected = self.tree.get_selected_actions_or_sessions(self.__history_sessions)
        if not selected:
            if not self.__history_sessions:
                return
            # Nothing selected → replay the last session
            selected = [self.__history_sessions[-1]]
        for session_or_action in selected:
            # B4: if a stale action is Played, recompute its cascade in-place
            # instead of running the standard non-persistent replay.
            if (
                isinstance(session_or_action, HistoryAction)
                and session_or_action.is_stale
            ):
                self.recompute_cascade(session_or_action)
                continue
            if not session_or_action.is_current_state_compatible(
                self.mainwindow, restore_selection=restore_selection
            ):
                QW.QMessageBox.critical(
                    self.mainwindow,
                    _("Error"),
                    _("The current workspace state is not compatible with the action."),
                )
                return
            if replay:
                if self.__edit_mode and isinstance(
                    session_or_action, HistoryAction
                ):
                    self._edit_mode_replay(session_or_action)
                elif self.__edit_mode and isinstance(
                    session_or_action, HistorySession
                ):
                    self._view_only_session_replay(
                        session_or_action, restore_selection
                    )
                else:
                    with self.replaying(), self.output_suppressed():
                        session_or_action.replay(
                            self.mainwindow,
                            restore_selection=restore_selection,
                            edit=self.__edit_mode,
                        )
            elif restore_selection:
                # Restore button (replay=False, restore_selection=True):
                # if Edit mode is active OR there are persisted pending
                # edits (e.g. after a save/reload), revert the parameter
                # snapshots in place. Otherwise behave as a workspace
                # selection restore.
                if self.__edit_mode or self._has_any_pending_edits():
                    self._restore_action_params(session_or_action)
                else:
                    session_or_action.restore(self.mainwindow)

    def _prompt_edit_action_params(self, action: HistoryAction) -> bool | None:
        """Open the parameter dialog for *action* according to its pattern.

        Returns:
            ``True``  – user accepted; ``action.kwargs`` mutated, snapshot taken.
            ``False`` – user cancelled the dialog.
            ``None``  – nothing to edit (no param/params, or unsupported pattern).
        """
        import copy  # pylint: disable=import-outside-toplevel

        pattern = action.pattern
        if pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
            param = action.kwargs.get("param")
            if param is None:
                return None
            edited = copy.deepcopy(param)
            if not edited.edit(parent=self.mainwindow):
                return False
            action.snapshot_kwargs()
            action.kwargs["param"] = edited
            return True
        if pattern == "1_to_n":
            params = action.kwargs.get("params") or []
            if not params:
                return None
            edited_params = [copy.deepcopy(p) for p in params]
            # Local import: ``gds`` is not used elsewhere in this module.
            import guidata.dataset as gds  # pylint: disable=import-outside-toplevel

            group = gds.DataSetGroup(edited_params, title=_("Parameters"))
            if not group.edit(parent=self.mainwindow):
                return False
            action.snapshot_kwargs()
            action.kwargs["params"] = edited_params
            return True
        # multiple_1_to_1 or any unknown pattern: nothing to edit.
        return None

    def _edit_mode_replay(self, action: HistoryAction) -> None:
        """Replay a single action in edit mode: open param dialog, update
        kwargs on accept, recompute in-place and cascade downstream.

        Supports every compute pattern (1_to_1, 1_to_n, n_to_1, 2_to_1,
        1_to_0). ``multiple_1_to_1`` is currently not supported: the dialog
        is skipped for that action but the rest of the chain is still
        processed.

        UI actions and pattern-less entries fall back to normal replay.

        The parameter dialog is opened for the root action AND for every
        downstream action in the cascade, in topological order.  If the
        user cancels ANY dialog, all snapshots already created in this
        pass are rolled back and nothing is recomputed.
        """
        if action.kind != HistoryAction.KIND_COMPUTE or action.pattern is None:
            with self.replaying(), self.output_suppressed():
                action.replay(self.mainwindow, restore_selection=True, edit=True)
            return

        chain: list[HistoryAction] = [action] + self.get_downstream_actions(
            action
        )
        edited_actions: list[HistoryAction] = []
        for a in chain:
            result = self._prompt_edit_action_params(a)
            if result is False:
                # User cancelled – rollback every snapshot taken so far.
                for done in edited_actions:
                    done.restore_kwargs()
                    self.tree.refresh_action_item(done)
                return
            if result is True:
                edited_actions.append(a)

        for a in edited_actions:
            self.tree.refresh_action_item(a)

        # Recompute root in-place, then cascade with the pre-computed
        # descendants list.  Re-using the chain avoids a second call to
        # ``get_downstream_actions`` whose result could diverge after the
        # root output's metadata has been rewritten by the root recompute.
        downstream = chain[1:]
        self._recompute_action_in_place(action)
        self.recompute_cascade(action, descendants=downstream)

        # Belt-and-suspenders refresh: ensure the tree description widgets
        # and the Signal/Image panels reflect the final state for every
        # action in the chain (root included).
        for a in chain:
            self.tree.refresh_action_item(a)
        QW.QApplication.processEvents()

    def _show_readonly_param_dialog(
        self, dataset: gds.DataSet | gds.DataSetGroup
    ) -> None:
        """Show a parameter dialog identical to the edit dialog but read-only.

        Builds the same guidata dialog as edit mode (``DataSetEditDialog`` for a
        single dataset, ``DataSetGroupEditDialog`` for a group) so the appearance
        and title match exactly, then disables every input field (and its label)
        so the parameters are displayed but cannot be modified. The OK button is
        kept so the dialog can be dismissed.

        Args:
            dataset: The dataset (or dataset group) whose parameters are shown.
        """
        # Local import: not used elsewhere in this module.
        import guidata.dataset as gds  # pylint: disable=import-outside-toplevel
        from guidata.dataset.qtwidgets import (  # pylint: disable=import-outside-toplevel
            DataSetEditDialog,
            DataSetGroupEditDialog,
        )

        # A DataSetGroup (1_to_n) needs the tabbed group dialog; a single
        # DataSet uses the standard edit dialog. Both expose ``edit_layout``.
        if isinstance(dataset, gds.DataSetGroup):
            dialog = DataSetGroupEditDialog(dataset, parent=self.mainwindow)
        else:
            dialog = DataSetEditDialog(dataset, parent=self.mainwindow)
        for edl in dialog.edit_layout:
            for widget in edl.widgets:
                if widget.group is not None:
                    widget.group.setEnabled(False)
                if widget.label is not None:
                    widget.label.setEnabled(False)
        dialog.exec()

    def _view_only_session_replay(
        self,
        session: HistorySession,
        restore_selection: bool,
    ) -> None:
        """Replay a session in edit mode with read-only parameter dialogs.

        When edit mode is active and a full session is replayed, editable
        dialogs would be misleading because modifications cannot be propagated
        through the cascade.  Instead, show each compute action's parameters in
        a dialog that looks identical to the edit dialog but whose fields are
        disabled, and keep the History tree and the data panel synchronized with
        the action being shown.  The session is then replayed with ``edit=False``.
        """
        import copy  # pylint: disable=import-outside-toplevel

        # Local import: ``gds`` is not used elsewhere in this module.
        import guidata.dataset as gds  # pylint: disable=import-outside-toplevel

        for action in session.actions:
            if action.kind != HistoryAction.KIND_COMPUTE:
                continue
            pattern = action.pattern
            # Sync the History tree and the data panel to this action so the
            # user follows the replay step by step (same behaviour as edit mode).
            self.__select_action_in_tree(action)
            QW.QApplication.processEvents()
            if pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
                param = action.kwargs.get("param")
                if param is not None:
                    self._show_readonly_param_dialog(copy.deepcopy(param))
            elif pattern == "1_to_n":
                params = action.kwargs.get("params") or []
                if params:
                    group = gds.DataSetGroup(
                        [copy.deepcopy(p) for p in params],
                        title=_("Parameters"),
                    )
                    self._show_readonly_param_dialog(group)

        with self.replaying(), self.output_suppressed():
            session.replay(
                self.mainwindow,
                restore_selection=restore_selection,
                edit=False,
            )

    def _restore_action_params(
        self, item: HistoryAction | HistorySession
    ) -> None:
        """Restore original kwargs from snapshot and recompute in-place.

        Used by the Restore action in edit mode to discard parameter
        changes and revert to the pre-edit state.
        """
        actions: list[HistoryAction]
        if isinstance(item, HistorySession):
            actions = [
                a for a in item.actions if a.kind == HistoryAction.KIND_COMPUTE
            ]
        else:
            actions = [item]
        for action in actions:
            if not action.has_pending_edits:
                continue
            action.restore_kwargs()
            self.tree.refresh_action_item(action)
            self._recompute_action_in_place(action)
            self.recompute_cascade(action)
        # Snapshots may have been consumed: refresh button states so the
        # Restore action disables itself once no pending edits remain
        # (relevant outside Edit mode after a save/reload restore).
        self.__update_actions_state()

    def _find_parent_session(self, action: HistoryAction) -> HistorySession | None:
        """Return the session that contains ``action``, or None if not found.

        Args:
            action: Action to search for.

        Returns:
            Parent :class:`HistorySession`, or ``None``.
        """
        for session in self.__history_sessions:
            if action in session.actions:
                return session
        return None

    # ------------------------------------------------------------------
    # Sync History tree selection → Signal/Image panel (B1)
    # ------------------------------------------------------------------

    def __resolve_panel_for_action(
        self, action: HistoryAction
    ) -> BaseDataPanel | None:
        """Return the data panel targeted by ``action``, or ``None``."""
        if action.kind != HistoryAction.KIND_COMPUTE:
            return None
        if action.panel_str == "signal":
            return self.mainwindow.signalpanel
        if action.panel_str == "image":
            return self.mainwindow.imagepanel
        return None

    def __find_output_object_uuid(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> str | None:
        """Find the UUID of the output object produced by ``action`` in ``panel``.

        Primary path: consult the bijective ``_action_output_uuids`` mapping
        populated when the action was recorded (or rebuilt from HDF5 at load).
        Returns the first registered output that still exists in ``panel``.

        Fallback path (legacy v1 sessions, or actions whose outputs were
        re-created without registration): search the panel for an object whose
        ``processing_parameters`` metadata has ``source_uuid`` matching one of
        the action's recorded selection UUIDs and whose ``func_name`` equals
        the action's ``func_name``.
        """
        # Primary: bijective mapping (C1).
        registered = self._action_output_uuids.get(action.uuid)
        if registered:
            existing_ids = set(panel.objmodel.get_object_ids())
            for out_uuid in registered:
                if out_uuid in existing_ids:
                    return out_uuid
        # Fallback heuristic for legacy sessions.
        if action.func_name is None:
            return None
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            extract_processing_parameters,
        )

        recorded_uuids = set(action.state.selection.get(panel.PANEL_STR_ID, []))
        if not recorded_uuids:
            return None
        for obj in panel.objmodel:
            pp = extract_processing_parameters(obj)
            if pp is None or pp.func_name != action.func_name:
                continue
            if pp.source_uuid is not None and pp.source_uuid in recorded_uuids:
                return get_uuid(obj)
            if pp.source_uuids is not None and recorded_uuids.intersection(
                pp.source_uuids
            ):
                return get_uuid(obj)
        return None

    def find_action_for_output(
        self, output_uuid: str, func_name: str
    ) -> HistoryAction | None:
        """Find the :class:`HistoryAction` that produced ``output_uuid``.

        Primary path: consult the bijective ``_output_to_action`` mapping. This
        is exact and resolves the ambiguity of repeated applications of the
        same ``func_name`` on the same source.

        Fallback path: searches all sessions (most-recent first) using the
        heuristic ``(func_name, source_uuid)`` matching for legacy v1 sessions
        without a registered output mapping.

        Args:
            output_uuid: UUID of the output object (signal or image).
            func_name: Processing function name expected on the action.

        Returns:
            The matching action, or ``None`` if no match is found.
        """
        if not self.__history_sessions:
            return None
        # Primary: bijective mapping (C1).
        action_uuid = self._output_to_action.get(output_uuid)
        if action_uuid is not None:
            for session in self.__history_sessions:
                for action in session.actions:
                    if action.uuid == action_uuid:
                        # Sanity check: func_name must still match.
                        if action.func_name == func_name:
                            return action
                        return None
        # Fallback heuristic for legacy sessions.
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            extract_processing_parameters,
        )

        # Find which panel contains output_uuid
        panel: BaseDataPanel | None = None
        output_obj = None
        for p in (self.mainwindow.signalpanel, self.mainwindow.imagepanel):
            try:
                output_obj = p.objmodel[output_uuid]
                panel = p
                break
            except KeyError:
                continue
        if panel is None or output_obj is None:
            return None
        pp = extract_processing_parameters(output_obj)
        if pp is None or pp.func_name != func_name:
            return None
        target_source_uuid = pp.source_uuid
        if target_source_uuid is None:
            return None
        # Search every session (most-recent first) instead of only [-1].
        # Uniqueness is guaranteed by target_source_uuid (remapped per-session
        # during duplication).
        for current_session in reversed(self.__history_sessions):
            for action in reversed(current_session.actions):
                if action.kind != HistoryAction.KIND_COMPUTE:
                    continue
                if action.func_name != func_name:
                    continue
                if action.panel_str != panel.PANEL_STR_ID:
                    continue
                captured = action.state.selection.get(panel.PANEL_STR_ID, [])
                if captured and captured[0] == target_source_uuid:
                    return action
        return None

    def refresh_action(self, action: HistoryAction) -> None:
        """Refresh the tree display for ``action`` after its kwargs were mutated.

        Used by :meth:`ObjectProp.apply_processing_parameters` to update the
        Description column when the user edits a ``param`` from the Processing
        tab of the Signal/Image panel.
        """
        self.tree.refresh_action_item(action)

    # ------------------------------------------------------------------
    # B4: Cascade recompute of downstream actions after a param edit
    # ------------------------------------------------------------------

    def _get_session_of(self, action: HistoryAction) -> HistorySession | None:
        """Return the session that contains ``action``, or None."""
        for session in self.__history_sessions:
            if action in session.actions:
                return session
        return None

    def _action_output_uuid(self, action: HistoryAction) -> str | None:
        """Return the UUID of the object produced by ``action``, or ``None``.

        Scans the target panel's object model for an object whose
        :class:`ProcessingParameters` metadata matches the action's
        ``func_name`` and one of its captured source UUIDs.
        """
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return None
        return self.__find_output_object_uuid(panel, action)

    def _action_consumes_any(
        self, action: HistoryAction, uuids: set[str]
    ) -> bool:
        """Return True if ``action``'s input UUIDs intersect ``uuids``."""
        if action.kind != HistoryAction.KIND_COMPUTE:
            return False
        pstr = action.panel_str or ""
        captured: set[str] = set(action.state.selection.get(pstr, []))
        obj2 = action.kwargs.get("obj2_uuids")
        if obj2:
            if isinstance(obj2, str):
                captured.add(obj2)
            else:
                captured.update(obj2)
        return bool(captured & uuids)

    def _collect_downstream_uuids(self, action: HistoryAction) -> set[str]:
        """Return the transitive closure of output UUIDs descending from
        ``action`` within the current session (excluding ``action`` itself).
        """
        if not self.__history_sessions:
            return set()
        current = self._get_session_of(action)
        if current is None:
            return set()
        root_out = self._action_output_uuid(action)
        if root_out is None:
            return set()
        closure: set[str] = {root_out}
        # Walk only actions positioned strictly after ``action`` in the
        # current session, in chronological order.
        idx = current.actions.index(action)
        for downstream in current.actions[idx + 1 :]:
            if downstream.kind != HistoryAction.KIND_COMPUTE:
                continue
            if not self._action_consumes_any(downstream, closure):
                continue
            out_uuid = self._action_output_uuid(downstream)
            if out_uuid is not None:
                closure.add(out_uuid)
        closure.discard(root_out)
        return closure

    def get_downstream_actions(
        self, action: HistoryAction
    ) -> list[HistoryAction]:
        """Return the actions of the current session that depend (transitively)
        on ``action``'s output, in topological order (direct children first).
        """
        if not self.__history_sessions:
            return []
        current = self._get_session_of(action)
        if current is None:
            return []
        root_out = self._action_output_uuid(action)
        if root_out is None:
            return []
        closure: set[str] = {root_out}
        downstream: list[HistoryAction] = []
        idx = current.actions.index(action)
        for candidate in current.actions[idx + 1 :]:
            if candidate.kind != HistoryAction.KIND_COMPUTE:
                continue
            if not self._action_consumes_any(candidate, closure):
                continue
            downstream.append(candidate)
            out_uuid = self._action_output_uuid(candidate)
            if out_uuid is not None:
                closure.add(out_uuid)
        return downstream

    # ------------------------------------------------------------------
    # In-place recompute dispatcher (C2): one helper per pattern.
    #
    # All helpers retrieve the existing output object(s) via the bijective
    # ``_action_output_uuids`` mapping (C1), recompute via the processor's
    # ``recompute_*`` methods (which do not register history nor add to the
    # panel), then update the existing object in place (data + title +
    # metadata) and refresh the view. Missing output objects (deleted by
    # the user) are reported via :attr:`_cascade_warnings`.
    # ------------------------------------------------------------------

    def _resolve_target_outputs(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> tuple[list[str], list[str]]:
        """Return ``(existing, missing)`` UUIDs registered for ``action``.

        Args:
            panel: Data panel owning the action's outputs.
            action: History action whose outputs must be resolved.

        Returns:
            A pair of UUID lists: those still present in ``panel`` (in
            registration order) and those that were deleted.
        """
        registered = list(self._action_output_uuids.get(action.uuid, []))
        existing_ids = set(panel.objmodel.get_object_ids())
        existing: list[str] = [u for u in registered if u in existing_ids]
        missing: list[str] = [u for u in registered if u not in existing_ids]
        return existing, missing

    def _update_obj_in_place(
        self,
        target_obj: SignalObj | ImageObj,
        new_obj: SignalObj | ImageObj,
    ) -> None:
        """Copy data + title + metadata from ``new_obj`` onto ``target_obj``.

        Preserves the target's identity (UUID, panel position, references)
        while reflecting all user-visible changes produced by a recompute.

        Args:
            target_obj: Existing object to mutate in place.
            new_obj: Fresh object produced by a ``recompute_*`` call.
        """
        target_obj.title = new_obj.title
        if isinstance(target_obj, SignalObj):
            target_obj.xydata = new_obj.xydata
        else:
            target_obj.data = new_obj.data
            target_obj.invalidate_maskdata_cache()
        # Replace compute-related metadata with the fresh set.  In Edit mode
        # the object is updated in place (not recreated), so stale analysis
        # result keys (Geometry_*, Table_*) or obsolete metadata options from
        # the previous compute would persist with a simple ``update()``.
        # We clear and repopulate, but preserve the target's ``__uuid`` which
        # is its identity key managed by the object model.
        try:
            saved_uuid = target_obj.metadata.get("__uuid")
            saved_number = target_obj.metadata.get("__number")
            target_obj.metadata.clear()
            target_obj.metadata.update(new_obj.metadata)
            if saved_uuid is not None:
                target_obj.metadata["__uuid"] = saved_uuid
            if saved_number is not None:
                target_obj.metadata["__number"] = saved_number
        except AttributeError:
            pass

    def _refresh_target(
        self, panel: BaseDataPanel, output_uuid: str
    ) -> None:
        """Refresh tree item + plot for ``output_uuid`` in ``panel``.

        Also updates the Properties panel when the refreshed object is
        currently selected, marks the object as freshly processed so the
        Processing tab is shown, and emits ``SIG_OBJECT_MODIFIED`` so
        that compatibility icons are refreshed.
        """
        panel.objview.update_item(output_uuid)
        panel.refresh_plot(output_uuid, update_items=True, force=True)

        # Update the Properties panel if the recomputed object is selected
        try:
            obj = panel.objmodel[output_uuid]
        except KeyError:
            obj = None
        if obj is not None:
            if obj is panel.objview.get_current_object():
                panel.objprop.update_properties_from(obj, force_tab="processing")
            else:
                # Mark as freshly processed so the Processing tab opens on select
                panel.objprop.mark_as_freshly_processed(obj)

        # Notify listeners (e.g. compatibility icons in history tree)
        panel.SIG_OBJECT_MODIFIED.emit()

    def _record_missing_outputs(
        self, action: HistoryAction, missing: list[str]
    ) -> None:
        """Log + queue a user-facing warning for deleted output objects."""
        if not missing:
            return
        name = action.func_name or action.title or action.uuid
        _logger.warning(
            "Cascade recompute: %d output(s) missing for action %s (%s).",
            len(missing),
            action.uuid,
            name,
        )
        self._cascade_warnings.append(
            _(
                "Action %s has been edited but its target output object(s) "
                "no longer exist — skipping."
            )
            % name
        )

    def _recompute_action_in_place(self, action: HistoryAction) -> None:
        """Re-run ``action`` on the existing output object(s) (same UUIDs).

        Dispatches to a per-pattern helper. Missing target outputs are
        recorded in :attr:`_cascade_warnings` and silently skipped so the
        rest of the cascade can keep running.

        When the underlying processor feature is missing (e.g. the originating
        plugin was uninstalled), :class:`FeatureNotFoundError` is caught here
        and the action is flagged as broken (``is_stale`` left at ``True``
        beyond the cascade so the visual marker persists). A localised warning
        including the plugin origin and required parameter class is appended
        to :attr:`_cascade_warnings`. The cascade continues with the remaining
        actions.

        Args:
            action: History action to recompute in place.
        """
        if action.kind != HistoryAction.KIND_COMPUTE:
            return
        method = {
            "1_to_1": self._recompute_1_to_1_in_place,
            "1_to_n": self._recompute_1_to_n_in_place,
            "n_to_1": self._recompute_n_to_1_in_place,
            "2_to_1": self._recompute_2_to_1_in_place,
            "1_to_0": self._recompute_1_to_0_in_place,
        }.get(action.pattern or "")
        if method is None:
            _logger.warning(
                "Cascade recompute: unsupported pattern %r for action %s.",
                action.pattern,
                action.uuid,
            )
            self._cascade_warnings.append(
                _("Action %s uses pattern %r which is not recomputable yet.")
                % (action.func_name or action.uuid, action.pattern)
            )
            return
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            FeatureNotFoundError,
        )

        try:
            method(action)
        except FeatureNotFoundError as exc:
            self._handle_missing_feature(action, exc)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.exception(
                "Cascade recompute failed for action %s (%s): %s",
                action.uuid,
                action.func_name,
                exc,
            )
            self._cascade_warnings.append(
                _("Recompute failed for action %s: %s")
                % (action.func_name or action.uuid, exc)
            )

    def _handle_missing_feature(
        self, action: HistoryAction, exc: "FeatureNotFoundError"
    ) -> None:
        """Flag ``action`` as broken (missing plugin) and queue a user warning.

        Args:
            action: Action whose feature could not be resolved.
            exc: The raised :class:`FeatureNotFoundError`.
             ``action.plugin_origin`` is the authoritative source.
             ``exc.plugin_origin`` is kept only as a safety net for future
             paths where ``get_feature`` might be called without forwarding
             the action's ``plugin_origin``.
        """
        action.is_stale = True
        self._broken_actions.add(action.uuid)
        plugin_origin = action.plugin_origin or exc.plugin_origin or {}
        directory = (plugin_origin.get("directory") if plugin_origin else None) or "?"
        param = action.kwargs.get("param")
        paramclass = (
            exc.paramclass_name
            or (type(param).__name__ if param is not None else "—")
        )
        func_name = action.func_name or exc.func_name or action.uuid
        # Format validated with the operator: "{directory}/plugins:{func_name}"
        location = f"{directory}/plugins:{func_name}"
        _logger.warning(
            "Cascade recompute: plugin missing for action %s (%s) — %s.",
            action.uuid,
            func_name,
            location,
        )
        self._cascade_warnings.append(
            _(
                "Action %(name)s skipped: plugin '%(loc)s' is missing.\n"
                "Required parameter class: %(param)s\n"
                "Reinstall the plugin to re-enable this action."
            )
            % {"name": func_name, "loc": location, "param": paramclass}
        )

    def _recompute_1_to_1_in_place(self, action: HistoryAction) -> None:
        """Recompute a single 1-to-1 action in place."""
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            ProcessingParameters,
            extract_processing_parameters,
            insert_processing_parameters,
        )

        existing, missing = self._resolve_target_outputs(panel, action)
        # Fall back to legacy heuristic when bijective mapping is unavailable.
        if not existing and not missing:
            legacy = self.__find_output_object_uuid(panel, action)
            if legacy is not None:
                existing = [legacy]
        self._record_missing_outputs(action, missing)
        if not existing:
            return
        output_uuid = existing[0]
        try:
            output_obj = panel.objmodel[output_uuid]
        except KeyError:
            return
        pp = extract_processing_parameters(output_obj)
        if pp is None or pp.source_uuid is None:
            return
        try:
            source_obj = panel.objmodel[pp.source_uuid]
        except KeyError:
            self._cascade_warnings.append(
                _("Action %s: source object was deleted — skipping.")
                % (action.func_name or action.uuid)
            )
            return
        param = action.kwargs.get("param")
        new_obj = panel.processor.recompute_1_to_1(
            action.func_name, source_obj, param,
            plugin_origin=action.plugin_origin,
        )
        if new_obj is None:
            return
        self._update_obj_in_place(output_obj, new_obj)
        insert_processing_parameters(
            output_obj,
            ProcessingParameters(
                func_name=pp.func_name,
                pattern=pp.pattern,
                param=param if param is not None else pp.param,
                source_uuid=pp.source_uuid,
            ),
        )
        panel.processor.auto_recompute_analysis(output_obj)
        self._refresh_target(panel, output_uuid)

    def _recompute_1_to_n_in_place(self, action: HistoryAction) -> None:
        """Recompute a 1-to-n action in place: replace each of the N outputs."""
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            ProcessingParameters,
            extract_processing_parameters,
            insert_processing_parameters,
        )

        existing, missing = self._resolve_target_outputs(panel, action)
        self._record_missing_outputs(action, missing)
        if not existing:
            return
        # All outputs of a 1_to_n share the same source.
        try:
            first_obj = panel.objmodel[existing[0]]
        except KeyError:
            return
        pp = extract_processing_parameters(first_obj)
        if pp is None or pp.source_uuid is None:
            return
        try:
            source_obj = panel.objmodel[pp.source_uuid]
        except KeyError:
            self._cascade_warnings.append(
                _("Action %s: source object was deleted — skipping.")
                % (action.func_name or action.uuid)
            )
            return
        params = action.kwargs.get("params") or []
        if not params:
            return
        new_objs = panel.processor.recompute_1_to_n(
            action.func_name, source_obj, params,
            plugin_origin=action.plugin_origin,
        )
        if not new_objs:
            return
        # Map each output to its (re)computed counterpart by index. If the
        # cardinality changed (e.g. function now produces fewer outputs),
        # we update what we can and report the rest as missing.
        n = min(len(existing), len(new_objs))
        for idx in range(n):
            out_uuid = existing[idx]
            try:
                out_obj = panel.objmodel[out_uuid]
            except KeyError:
                continue
            new_obj = new_objs[idx]
            self._update_obj_in_place(out_obj, new_obj)
            new_param = params[idx] if idx < len(params) else None
            insert_processing_parameters(
                out_obj,
                ProcessingParameters(
                    func_name=action.func_name,
                    pattern="1-to-n",
                    param=new_param,
                    source_uuid=pp.source_uuid,
                ),
            )
            panel.processor.auto_recompute_analysis(out_obj)
            self._refresh_target(panel, out_uuid)
        if len(new_objs) != len(existing):
            _logger.warning(
                "1-to-n cardinality changed for action %s: %d outputs, %d existing.",
                action.uuid,
                len(new_objs),
                len(existing),
            )

    def _recompute_n_to_1_in_place(self, action: HistoryAction) -> None:
        """Recompute an n-to-1 action in place."""
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            ProcessingParameters,
            extract_processing_parameters,
            insert_processing_parameters,
        )

        existing, missing = self._resolve_target_outputs(panel, action)
        self._record_missing_outputs(action, missing)
        if not existing:
            return
        output_uuid = existing[0]
        try:
            output_obj = panel.objmodel[output_uuid]
        except KeyError:
            return
        pp = extract_processing_parameters(output_obj)
        source_uuids: list[str] = []
        if pp is not None and pp.source_uuids:
            source_uuids = list(pp.source_uuids)
        else:
            source_uuids = list(
                action.state.selection.get(panel.PANEL_STR_ID, [])
            )
        src_objs: list[SignalObj | ImageObj] = []
        for uuid in source_uuids:
            try:
                src_objs.append(panel.objmodel[uuid])
            except KeyError:
                continue
        if not src_objs:
            self._cascade_warnings.append(
                _("Action %s: all source objects were deleted — skipping.")
                % (action.func_name or action.uuid)
            )
            return
        param = action.kwargs.get("param")
        new_obj = panel.processor.recompute_n_to_1(
            action.func_name, src_objs, param,
            plugin_origin=action.plugin_origin,
        )
        if new_obj is None:
            return
        self._update_obj_in_place(output_obj, new_obj)
        insert_processing_parameters(
            output_obj,
            ProcessingParameters(
                func_name=action.func_name,
                pattern="n-to-1",
                param=param,
                source_uuids=[get_uuid(o) for o in src_objs],
            ),
        )
        panel.processor.auto_recompute_analysis(output_obj)
        self._refresh_target(panel, output_uuid)

    def _recompute_2_to_1_in_place(self, action: HistoryAction) -> None:
        """Recompute a 2-to-1 action in place (single or pairwise)."""
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            ProcessingParameters,
            extract_processing_parameters,
            insert_processing_parameters,
        )

        existing, missing = self._resolve_target_outputs(panel, action)
        self._record_missing_outputs(action, missing)
        if not existing:
            return
        param = action.kwargs.get("param")
        obj2_uuids = action.kwargs.get("obj2_uuids") or []
        if isinstance(obj2_uuids, str):
            obj2_uuids = [obj2_uuids]
        pairwise = bool(action.kwargs.get("pairwise"))
        # In pairwise mode, expect one output per (obj1, obj2) pair.
        # In single-operand mode, every output uses the same obj2.
        recorded_inputs = list(action.state.selection.get(panel.PANEL_STR_ID, []))
        for idx, out_uuid in enumerate(existing):
            try:
                output_obj = panel.objmodel[out_uuid]
            except KeyError:
                continue
            pp = extract_processing_parameters(output_obj)
            src_uuids = (
                list(pp.source_uuids)
                if pp is not None and pp.source_uuids
                else (recorded_inputs[idx : idx + 1] + obj2_uuids[idx : idx + 1]
                      if pairwise
                      else recorded_inputs[idx : idx + 1] + obj2_uuids[:1])
            )
            if len(src_uuids) < 2:
                self._cascade_warnings.append(
                    _("Action %s: missing source(s) for output #%d — skipping.")
                    % (action.func_name or action.uuid, idx + 1)
                )
                continue
            try:
                obj1 = panel.objmodel[src_uuids[0]]
                obj2 = panel.objmodel[src_uuids[1]]
            except KeyError:
                self._cascade_warnings.append(
                    _("Action %s: source object(s) were deleted — skipping.")
                    % (action.func_name or action.uuid)
                )
                continue
            new_obj = panel.processor.recompute_2_to_1(
                action.func_name, obj1, obj2, param,
                plugin_origin=action.plugin_origin,
            )
            if new_obj is None:
                continue
            self._update_obj_in_place(output_obj, new_obj)
            insert_processing_parameters(
                output_obj,
                ProcessingParameters(
                    func_name=action.func_name,
                    pattern="2-to-1",
                    param=param,
                    source_uuids=[get_uuid(obj1), get_uuid(obj2)],
                ),
            )
            panel.processor.auto_recompute_analysis(output_obj)
            self._refresh_target(panel, out_uuid)

    def _recompute_1_to_0_in_place(self, action: HistoryAction) -> None:
        """Recompute a 1-to-0 analysis on each source object in place."""
        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return
        # 1-to-0 produces no data object; recompute on each captured source
        # so analysis metadata stays consistent with the (possibly updated)
        # data of upstream actions.
        sources = list(action.state.selection.get(panel.PANEL_STR_ID, []))
        if not sources:
            return
        param = action.kwargs.get("param")
        missing: list[str] = []
        for uuid in sources:
            try:
                src_obj = panel.objmodel[uuid]
            except KeyError:
                missing.append(uuid)
                continue
            panel.processor.recompute_1_to_0(
                action.func_name, src_obj, param,
                plugin_origin=action.plugin_origin,
            )
            self._refresh_target(panel, uuid)
        if missing:
            self._cascade_warnings.append(
                _("Action %s: %d analysed object(s) were deleted — skipping.")
                % (action.func_name or action.uuid, len(missing))
            )

    def recompute_cascade(
        self,
        root_action: HistoryAction,
        descendants: list[HistoryAction] | None = None,
    ) -> None:
        """Recompute ``root_action``'s descendants in the current session
        in-place (on existing UUIDs).

        Each action involved is flagged ``is_stale`` (light-orange background
        in the tree) for the duration of the recompute, then cleared. The
        root action itself is normally NOT recomputed here (the caller has
        already updated its output object). For a stale Play, ``root_action``
        is included so its own flag is cleared.

        Every supported pattern (``1_to_1``, ``1_to_n``, ``n_to_1``,
        ``2_to_1``, ``1_to_0``) is dispatched through
        :meth:`_recompute_action_in_place`. Missing or unsupported items
        are reported via a single end-of-cascade warning dialog.

        Actions whose underlying processor feature is missing (e.g. plugin
        uninstalled) keep ``is_stale = True`` after the cascade completes,
        so the visual marker persists until the plugin is reinstalled.

        Args:
            root_action: Root action whose descendants must be recomputed.
            descendants: Pre-computed downstream actions. When ``None``
                (default), :meth:`get_downstream_actions` is called
                internally. Passing a pre-computed list avoids a redundant
                graph traversal when the caller has already resolved the
                chain (e.g. :meth:`_edit_mode_replay`).
        """
        if descendants is None:
            descendants = self.get_downstream_actions(root_action)
        if root_action.is_stale:
            descendants = [root_action] + descendants
        # Re-entrancy guard.
        if getattr(self, "_cascade_in_progress", False):
            self._flush_cascade_warnings()
            return
        if not descendants:
            # Still surface any warnings accumulated by a prior standalone
            # ``_recompute_action_in_place`` call (e.g. root recompute in
            # ``_edit_mode_replay``) before bailing out.
            self._flush_cascade_warnings()
            return
        # Reset the broken set: only the current cascade's outcomes count.
        self._broken_actions.clear()
        self._cascade_in_progress = True
        try:
            for action in descendants:
                action.is_stale = True
                self.tree.refresh_action_item(action)
            QW.QApplication.processEvents()
            for action in descendants:
                try:
                    self._recompute_action_in_place(action)
                finally:
                    # Keep the stale marker for actions flagged as broken
                    # (missing plugin) so the user can spot them in the tree.
                    if action.uuid not in self._broken_actions:
                        action.is_stale = False
                    self.tree.refresh_action_item(action)
                    QW.QApplication.processEvents()
        finally:
            for action in descendants:
                if action.is_stale and action.uuid not in self._broken_actions:
                    action.is_stale = False
                    self.tree.refresh_action_item(action)
            self._cascade_in_progress = False
        self._flush_cascade_warnings()

    def _flush_cascade_warnings(self) -> None:
        """Show + clear accumulated cascade warnings (no-op when empty)."""
        if self._cascade_warnings:
            QW.QMessageBox.warning(
                self.mainwindow,
                _("Cascade recompute"),
                _("Some downstream actions could not be recomputed:")
                + "\n\n• "
                + "\n• ".join(self._cascade_warnings),
            )
        self._cascade_warnings = []


    def __existing_input_uuids(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> list[str]:
        """Return recorded input UUIDs that still exist in ``panel``."""
        recorded = action.state.selection.get(panel.PANEL_STR_ID, [])
        existing: list[str] = []
        for uuid in recorded:
            try:
                panel.objmodel[uuid]
            except KeyError:
                continue
            existing.append(uuid)
        return existing

    def __sync_panel_selection(self) -> None:
        """Sync data panel selection from the currently selected tree item."""
        if self.__replaying or self.__syncing:
            return
        item = self.tree.currentItem()
        if item is None or not item.isSelected():
            return
        if item.parent() is None:
            # Session-level selection: peek the first compute action
            index = self.tree.indexOfTopLevelItem(item)
            if index < 0 or index >= len(self.__history_sessions):
                return
            session = self.__history_sessions[index]
            action = next(
                (a for a in session.actions if a.kind == HistoryAction.KIND_COMPUTE),
                None,
            )
            if action is None:
                return
        else:
            uuid = item.data(0, QC.Qt.UserRole)
            try:
                action = self.tree.get_action_from_uuid(
                    uuid, self.__history_sessions
                )
            except ValueError:
                return

        panel = self.__resolve_panel_for_action(action)
        if panel is None:
            return

        target_uuids: list[str] = []
        output_uuid = self.__find_output_object_uuid(panel, action)
        if output_uuid is not None:
            target_uuids = [output_uuid]
        else:
            target_uuids = self.__existing_input_uuids(panel, action)

        if not target_uuids:
            return

        self.__syncing = True
        try:
            with QC.QSignalBlocker(panel.objview):
                panel.objview.select_objects(target_uuids)
            self.mainwindow.set_current_panel(panel)
        finally:
            self.__syncing = False

    # ------------------------------------------------------------------
    # Step-by-step navigation (B2)
    # ------------------------------------------------------------------

    def __current_action(self) -> HistoryAction | None:
        """Return the action currently selected in the tree, or ``None``."""
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return None
        uuid = item.data(0, QC.Qt.UserRole)
        try:
            return self.tree.get_action_from_uuid(uuid, self.__history_sessions)
        except ValueError:
            return None

    def __current_session(self) -> HistorySession | None:
        """Return the session relevant for step navigation."""
        item = self.tree.currentItem()
        if item is not None:
            if item.parent() is None:
                index = self.tree.indexOfTopLevelItem(item)
                if 0 <= index < len(self.__history_sessions):
                    return self.__history_sessions[index]
            else:
                action = self.__current_action()
                if action is not None:
                    return self._find_parent_session(action)
        if self.__history_sessions:
            return self.__history_sessions[-1]
        return None

    def __can_step_prev(self) -> bool:
        """Return True if a previous action exists in the current session."""
        session = self.__current_session()
        if session is None or not session.actions:
            return False
        action = self.__current_action()
        if action is None or action not in session.actions:
            return False
        return session.actions.index(action) > 0

    def __can_step_next(self) -> bool:
        """Return True if a next action exists in the current session."""
        session = self.__current_session()
        if session is None or not session.actions:
            return False
        action = self.__current_action()
        if action is None or action not in session.actions:
            # No action selected — Next would land on the first one.
            return True
        return session.actions.index(action) < len(session.actions) - 1

    def __select_action_in_tree(self, action: HistoryAction) -> None:
        """Select ``action`` in the tree (triggers `__sync_panel_selection`)."""
        for i in range(self.tree.topLevelItemCount()):
            sess_item = self.tree.topLevelItem(i)
            for j in range(sess_item.childCount()):
                child = sess_item.child(j)
                if child.data(0, QC.Qt.UserRole) == action.uuid:
                    self.tree.clearSelection()
                    self.tree.setCurrentItem(child)
                    child.setSelected(True)
                    return

    def _step_prev(self) -> None:
        """Select the previous action in the current session."""
        if not self.__can_step_prev():
            return
        session = self.__current_session()
        action = self.__current_action()
        idx = session.actions.index(action)
        self.__select_action_in_tree(session.actions[idx - 1])
        self.__update_actions_state()

    def _step_next(self) -> None:
        """Select the next action in the current session."""
        if not self.__can_step_next():
            return
        session = self.__current_session()
        action = self.__current_action()
        if action is None or action not in session.actions:
            target = session.actions[0]
        else:
            target = session.actions[session.actions.index(action) + 1]
        self.__select_action_in_tree(target)
        self.__update_actions_state()

    def duplicate_selected_entries(self) -> None:
        """Duplicate selected sessions (with their data) into new independent sessions.

        For each selected session (or the parent session of a selected action),
        all referenced data objects are deep-copied into a new group and the
        session is duplicated with all UUID references rewritten to the clones.
        The result is an independent, editable and replayable session.
        """
        selected = self.tree.get_selected_actions_or_sessions(self.__history_sessions)
        if not selected:
            return
        # Normalise: resolve individual actions to their parent session, deduplicate.
        sessions_to_dup: list[HistorySession] = []
        seen: set[int] = set()
        for item in selected:
            if isinstance(item, HistorySession):
                session = item
            else:
                session = self._find_parent_session(item)
                if session is None:
                    continue
            if id(session) not in seen:
                seen.add(id(session))
                sessions_to_dup.append(session)

        copy_suffix = _("Copy")
        new_sessions: list[HistorySession] = []
        panel_map = {
            "signal": self.mainwindow.signalpanel,
            "image": self.mainwindow.imagepanel,
        }

        for session in sessions_to_dup:
            # 1. Collect all UUIDs referenced by this session
            uuids_by_panel: dict[str, set[str]] = {}
            for action in session.actions:
                for pstr, uuids in action.state.selection.items():
                    uuids_by_panel.setdefault(pstr, set()).update(uuids)
                for pstr, metadata in action.state.object_metadata.items():
                    uuids_by_panel.setdefault(pstr, set()).update(metadata.keys())
                obj2 = action.kwargs.get("obj2_uuids")
                if obj2:
                    pstr = action.panel_str or ""
                    if isinstance(obj2, str):
                        obj2 = [obj2]
                    uuids_by_panel.setdefault(pstr, set()).update(obj2)
                # Output UUIDs produced by this action (e.g. result of a
                # compute step). Without this, the last action's outputs
                # would be missing because no subsequent state captures them.
                if action.output_uuids:
                    pstr = action.panel_str or ""
                    uuids_by_panel.setdefault(pstr, set()).update(
                        action.output_uuids
                    )

            # 2. Clone objects and build uuid_remap
            uuid_remap: dict[str, dict[str, str]] = {}
            clones_by_pstr: dict[str, list] = {}
            group_title = f"{copy_suffix} - {session.title}"
            for pstr, uuids in uuids_by_panel.items():
                panel = panel_map.get(pstr)
                if panel is None:
                    continue
                uuid_remap[pstr] = {}
                existing_ids = set(panel.objmodel.get_object_ids())
                clones = []
                # Iterate in panel order (not set order) to preserve
                # the topological object ordering in the duplicated group.
                ordered_ids = [
                    u for u in panel.objmodel.get_object_ids() if u in uuids
                ]
                for old_uuid in ordered_ids:
                    if old_uuid not in existing_ids:
                        continue
                    obj = panel.objmodel[old_uuid]
                    clone = deepcopy(obj)
                    new_uuid = str(uuid4())
                    # SignalObj/ImageObj store UUID via metadata option
                    try:
                        clone.set_metadata_option("uuid", new_uuid)
                    except AttributeError:
                        clone.uuid = new_uuid
                    uuid_remap[pstr][old_uuid] = new_uuid
                    clones.append(clone)
                clones_by_pstr[pstr] = clones
                if clones:
                    group_id = get_uuid(panel.add_group(group_title))
                    for clone in clones:
                        panel.add_object(clone, group_id=group_id)

            # Second pass: remap source UUIDs in cloned objects'
            # processing_parameters so reprocessing in the Processing tab
            # uses the cloned source, not the original.
            from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
                PROCESSING_PARAMETERS_OPTION,
                ProcessingParameters,
            )

            for pstr_inner, clones_inner in clones_by_pstr.items():
                pmap = uuid_remap.get(pstr_inner, {})
                if not pmap:
                    continue
                for clone in clones_inner:
                    try:
                        pp_dict = clone.get_metadata_option(
                            PROCESSING_PARAMETERS_OPTION
                        )
                    except (AttributeError, ValueError):
                        continue
                    if not pp_dict:
                        continue
                    try:
                        pp = ProcessingParameters.from_dict(pp_dict)
                    except Exception:  # pylint: disable=broad-except
                        continue
                    changed = False
                    if pp.source_uuid is not None and pp.source_uuid in pmap:
                        pp.source_uuid = pmap[pp.source_uuid]
                        changed = True
                    if pp.source_uuids is not None:
                        new_src = [pmap.get(u, u) for u in pp.source_uuids]
                        if new_src != pp.source_uuids:
                            pp.source_uuids = new_src
                            changed = True
                    if changed:
                        try:
                            clone.set_metadata_option(
                                PROCESSING_PARAMETERS_OPTION, pp.to_dict()
                            )
                        except (AttributeError, ValueError):
                            pass

            # 3. Build the new session with remapped UUIDs
            self.__session_increment += 1
            title = f"{session.title} {copy_suffix}"
            new_session = session.copy_with_uuid_remap(
                title=title, uuid_remap=uuid_remap
            )
            new_session.number = self.__session_increment
            new_sessions.append(new_session)

            # Register output mappings for cloned actions so that
            # _resolve_target_outputs / get_downstream_actions work on
            # the duplicated session (same logic as read_h5_data).
            for action in new_session.actions:
                if action.output_uuids:
                    self._action_output_uuids[action.uuid] = list(
                        action.output_uuids
                    )
                    for out_uuid in action.output_uuids:
                        self._output_to_action[out_uuid] = action.uuid

        # Insert each duplicated session immediately after its original.
        offset = 0
        for original_session, new_session in zip(sessions_to_dup, new_sessions):
            idx = self.__history_sessions.index(original_session)
            self.__history_sessions.insert(idx + 1 + offset, new_session)
            offset += 1
        self.tree.populate_tree(self.__history_sessions)
        self.__select_sessions(new_sessions)
        self.refresh_compatibility_items()
        self.__update_actions_state()

    def generate_macro(self) -> None:
        """Generate a standalone Python script from selected history entries.

        The generated script uses sigima functions directly with proper variable
        chaining.  Object references (UUIDs) are resolved to variable names so
        that 2-to-1 operations reference the correct intermediate result.
        The script is copied to the clipboard and the user is notified.
        """
        selected = self.tree.get_selected_actions_or_sessions(
            self.__history_sessions
        )
        actions: list[HistoryAction] = []
        if not selected:
            for session in self.__history_sessions:
                actions.extend(session.actions)
        else:
            for item in selected:
                if isinstance(item, HistorySession):
                    actions.extend(item.actions)
                else:
                    actions.append(item)
        if not actions:
            return

        # Filter to compute-only actions for the pipeline
        compute_actions = [
            a for a in actions if a.kind == HistoryAction.KIND_COMPUTE
        ]
        if not compute_actions:
            QW.QMessageBox.information(
                self.mainwindow,
                _("Generate macro"),
                _("No compute actions to export."),
            )
            return

        # Determine input type from first action
        first_panel = compute_actions[0].panel_str
        if first_panel == "signal":
            obj_type = "SignalObj"
            obj_import = "from sigima.objects import SignalObj"
        else:
            obj_type = "ImageObj"
            obj_import = "from sigima.objects import ImageObj"

        imports: set[str] = set()
        imports.add(obj_import)
        body_lines: list[str] = []

        # UUID → variable mapping for resolving object references.
        # Populated with input UUIDs ("src", "src_2", ...) and enriched
        # with each step's output UUID after code generation.
        uuid_to_var: dict[str, str] = {}

        # Extra input parameters discovered during generation (second
        # operands that are not produced by any previous step).
        extra_inputs: list[str] = []

        # Seed the mapping with the first action's input selection.
        first_sel = compute_actions[0].state.selection.get(
            compute_actions[0].panel_str, []
        )
        for i, uuid in enumerate(first_sel):
            var = "src" if i == 0 else f"src_{i + 1}"
            uuid_to_var[uuid] = var

        step = 0
        current_var = "src"

        for action in compute_actions:
            step += 1

            # Resolve input variable from the action's selection UUIDs.
            sel_uuids = action.state.selection.get(
                action.panel_str or "", []
            )
            if sel_uuids and sel_uuids[0] in uuid_to_var:
                input_var = uuid_to_var[sel_uuids[0]]
            else:
                input_var = current_var

            # Resolve second operand for 2-to-1 patterns.
            obj2_var: str | None = None
            if action.pattern == "2_to_1":
                obj2_uuids = action.kwargs.get("obj2_uuids", [])
                if isinstance(obj2_uuids, str):
                    obj2_uuids = [obj2_uuids]
                if obj2_uuids:
                    obj2_uuid = obj2_uuids[0]
                    if obj2_uuid in uuid_to_var:
                        obj2_var = uuid_to_var[obj2_uuid]
                    else:
                        # External input — add as function parameter.
                        obj2_var = f"obj2_{step}"
                        uuid_to_var[obj2_uuid] = obj2_var
                        extra_inputs.append(obj2_var)

            code_lines, output_var = action.to_macro_code(
                step, input_var, imports, obj2_var=obj2_var
            )
            body_lines.extend(code_lines)
            body_lines.append("")

            if output_var is not None:
                current_var = output_var
                # Map the output UUID so subsequent steps can reference it.
                output_uuid = self._action_output_uuid(action)
                if output_uuid:
                    uuid_to_var[output_uuid] = output_var
                # Also register any new UUIDs from the action's selection
                # that we haven't seen yet (secondary selections).
                for uuid in sel_uuids[1:]:
                    if uuid not in uuid_to_var:
                        uuid_to_var[uuid] = input_var

        # Build the function signature with extra inputs.
        params_str = f"src: {obj_type}"
        for extra in extra_inputs:
            params_str += f", {extra}: {obj_type}"

        # Assemble the full script
        sorted_imports = sorted(imports)
        script_lines: list[str] = [
            '"""',
            "DataLab — standalone processing pipeline",
            f"Generated from history ({len(compute_actions)} steps)",
            '"""',
            "",
        ]
        script_lines.extend(sorted_imports)
        script_lines.append("")
        script_lines.append("")
        script_lines.append(
            f"def process({params_str}) -> {obj_type}:"
        )
        script_lines.append(
            '    """Apply the recorded processing pipeline."""'
        )
        for line in body_lines:
            script_lines.append(f"    {line}" if line else "")
        script_lines.append(f"    return {current_var}")
        script_lines.append("")
        script_lines.append("")
        script_lines.append('if __name__ == "__main__":')
        script_lines.append(
            "    # Standalone execution: run from DataLab's Macro panel."
        )
        script_lines.append("    # Operates on the current object of the target panel.")
        script_lines.append(
            "    from datalab.control.proxy import RemoteProxy"
        )
        script_lines.append("")
        script_lines.append("    proxy = RemoteProxy()")
        panel_str = compute_actions[0].panel_str or (
            "signal" if obj_type == "SignalObj" else "image"
        )
        script_lines.append(f'    proxy.set_current_panel("{panel_str}")')
        script_lines.append("    src = proxy.get_object()")
        script_lines.append("    if src is None:")
        script_lines.append(
            f'        raise RuntimeError("No current object in panel: {panel_str}")'
        )
        if extra_inputs:
            n_extra = len(extra_inputs)
            script_lines.append(
                "    _uuids = [u for u in proxy.get_sel_object_uuids()"
                " if u != src.uuid]"
            )
            script_lines.append(f"    if len(_uuids) < {n_extra}:")
            script_lines.append(
                "        raise RuntimeError("
            )
            script_lines.append(
                f'            "Pipeline needs {n_extra} extra selected'
                ' object(s) besides the current one"'
            )
            script_lines.append("        )")
            for idx, extra in enumerate(extra_inputs):
                script_lines.append(
                    f"    {extra} = proxy.get_object("
                    f'_uuids[{idx}], "{panel_str}")'
                )
        extra_args = "".join(f", {e}" for e in extra_inputs)
        script_lines.append(f"    result = process(src{extra_args})")
        script_lines.append("    proxy.add_object(result)")
        script_lines.append('    print(f"Pipeline applied: {result.title}")')
        script_lines.append("")

        script = "\n".join(script_lines)
        QW.QApplication.clipboard().setText(script)
        QW.QMessageBox.information(
            self.mainwindow,
            _("Generate macro"),
            _("Macro script copied to clipboard (%d actions).")
            % len(compute_actions),
        )

    def __select_sessions(self, sessions: list[HistorySession]) -> None:
        """Select top-level tree items matching ``sessions``."""
        self.tree.clearSelection()
        for session in sessions:
            index = self.__history_sessions.index(session)
            item = self.tree.topLevelItem(index)
            item.setSelected(True)
            self.tree.setCurrentItem(item)

    def delete_selected(self) -> None:
        """Delete the selected actions or sessions (with confirmation).

        When a top-level session is selected, the entire session is deleted.
        When individual actions are selected, they and all subsequent actions
        in their parent session are removed. After deletion, the first
        available item in the tree is selected automatically.
        """
        selected = self.tree.get_selected_actions_or_sessions(self.__history_sessions)
        if not selected:
            return
        has_individual_actions = any(
            isinstance(item, HistoryAction) for item in selected
        )
        if has_individual_actions:
            msg = _(
                "Do you really want to delete the selected items?\n\n"
                "Note: deleting an action also removes all subsequent "
                "actions in the same session."
            )
        else:
            msg = _("Do you really want to delete the selected items?")
        reply = QW.QMessageBox.question(
            self.mainwindow,
            _("Delete"),
            msg,
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if reply != QW.QMessageBox.Yes:
            return
        sessions_to_remove: set[int] = set()
        for item in selected:
            if isinstance(item, HistorySession):
                sessions_to_remove.add(id(item))
            else:
                # Individual action: remove from its parent session
                for session in self.__history_sessions:
                    if item in session.actions:
                        session.remove_action(item)
                        if not session.actions:
                            sessions_to_remove.add(id(session))
                        break
        self.__history_sessions = [
            s for s in self.__history_sessions if id(s) not in sessions_to_remove
        ]
        self.tree.populate_tree(self.__history_sessions)
        self.refresh_compatibility_items()
        self.__update_actions_state()
        # Auto-select the first available item after deletion
        if self.tree.topLevelItemCount() > 0:
            first = self.tree.topLevelItem(0)
            self.tree.setCurrentItem(first)
            first.setSelected(True)

    def remove_incompatible_actions(self) -> None:
        """Remove all actions whose workspace state is incompatible.

        Shows a confirmation dialog listing how many actions will be removed,
        then purges them from their sessions. Empty sessions are also removed.
        """
        incompatible: list[tuple[HistorySession, HistoryAction]] = []
        for session in self.__history_sessions:
            for action in session.actions:
                if not action.is_current_state_compatible(
                    self.mainwindow, restore_selection=True
                ):
                    incompatible.append((session, action))
        if not incompatible:
            QW.QMessageBox.information(
                self.mainwindow,
                _("Remove incompatible"),
                _("All actions are compatible with the current workspace."),
            )
            return
        reply = QW.QMessageBox.question(
            self.mainwindow,
            _("Remove incompatible"),
            _("%d incompatible action(s) will be removed. Continue?")
            % len(incompatible),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if reply != QW.QMessageBox.Yes:
            return
        for session, action in incompatible:
            if action in session.actions:
                session.actions.remove(action)
        # Remove empty sessions
        self.__history_sessions = [
            s for s in self.__history_sessions if s.actions
        ]
        self.tree.populate_tree(self.__history_sessions)
        self.refresh_compatibility_items()
        self.__update_actions_state()

    def save_to_dlhist_file(self, filename: str | None = None) -> bool:
        """Save the History Panel content to a standalone ``.dlhist`` file.

        Args:
            filename: History filename. If None, a file dialog is opened.

        Returns:
            True if the history was saved, False if the operation was canceled.
        """
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getsavefilename(
                    self, _("Save history file"), basedir, self.FILE_FILTERS
                )
        if not filename:
            return False
        if osp.splitext(filename)[1] == "":
            filename += ".dlhist"
        with qt_try_loadsave_file(self.parentWidget(), filename, "save"):
            Conf.main.base_dir.set(filename)
            from datalab.h5.native import NativeH5Writer  # pylint: disable=C0415

            with NativeH5Writer(filename) as writer:
                # Make the .dlhist file self-contained: store the signal and
                # image panel objects (all of them) alongside the history, so
                # that reopening restores both the data objects and the history
                # that references them. Each section is read back by its own
                # H5_PREFIX key, so the write order is not significant.
                self.mainwindow.signalpanel.serialize_to_hdf5(writer)
                self.mainwindow.imagepanel.serialize_to_hdf5(writer)
                self.serialize_to_hdf5(writer)
        return True

    def open_dlhist_file(self, filename: str | None = None) -> bool:
        """Open a standalone ``.dlhist`` file into the History Panel.

        Args:
            filename: History filename. If None, a file dialog is opened.

        Returns:
            True if the history was loaded, False if the operation was canceled.
        """
        if filename is None:
            basedir = Conf.main.base_dir.get()
            with save_restore_stds():
                filename, _filt = getopenfilename(
                    self, _("Open history file"), basedir, self.FILE_FILTERS
                )
        if not filename:
            return False
        with qt_try_loadsave_file(self.parentWidget(), filename, "load"):
            Conf.main.base_dir.set(filename)
            from datalab.h5.native import NativeH5Reader  # pylint: disable=C0415

            with NativeH5Reader(filename) as reader:
                # A self-contained .dlhist file stores the signal and image
                # panel objects in addition to the history sessions. The way
                # they are restored depends on whether the workspace is already
                # in use (data objects OR history): a pristine workspace is
                # loaded directly while preserving UUIDs, otherwise the file
                # is imported as new groups/sessions.
                workspace_in_use = (
                    self.mainwindow.signalpanel.objmodel.get_object_ids()
                    or self.mainwindow.imagepanel.objmodel.get_object_ids()
                    or bool(self.__history_sessions)
                )
                if workspace_in_use:
                    # Workspace not empty: import the objects into new groups
                    # with fresh UUIDs and append the history as new sessions
                    # whose references are remapped to the imported objects.
                    self.__import_dlhist_into_new_session(reader)
                else:
                    # Workspace empty: load directly, preserving original UUIDs
                    # (reset_all=True) so that history references stay valid.
                    self.mainwindow.signalpanel.deserialize_from_hdf5(
                        reader, reset_all=True
                    )
                    self.mainwindow.imagepanel.deserialize_from_hdf5(
                        reader, reset_all=True
                    )
                    self.deserialize_from_hdf5(reader)
        return True

    def __import_dlhist_into_new_session(self, reader: NativeH5Reader) -> None:
        """Import a ``.dlhist`` file into new groups and new history sessions.

        Used when the workspace already contains objects: the file's signal and
        image objects are imported into fresh groups with regenerated UUIDs, and
        the history sessions are appended as new independent sessions whose action
        references are remapped to the freshly imported objects.

        Args:
            reader: HDF5 reader positioned on a ``.dlhist`` file.
        """
        panel_map = {
            "signal": self.mainwindow.signalpanel,
            "image": self.mainwindow.imagepanel,
        }
        uuid_remap: dict[str, dict[str, str]] = {}
        imported_by_pstr: dict[str, list] = {}
        # 1. Import objects from each panel (each panel is read by its own
        #    H5_PREFIX key). Read each object preserving its original UUID to
        #    capture the old->new mapping, then assign a fresh UUID so that the
        #    imported objects keep an independent identity.
        for pstr, panel in panel_map.items():
            uuid_remap[pstr] = {}
            imported: list = []
            imported_by_pstr[pstr] = imported
            if panel.H5_PREFIX not in reader.h5:
                continue
            with reader.group(panel.H5_PREFIX):
                for name in reader.h5.get(panel.H5_PREFIX, []):
                    with reader.group(name):
                        group = panel.add_group("")
                        with reader.group("title"):
                            group.title = reader.read_str()
                        for obj_name in reader.h5.get(
                            f"{panel.H5_PREFIX}/{name}", []
                        ):
                            obj = panel.deserialize_object_from_hdf5(
                                reader, obj_name, reset_all=True
                            )
                            old_uuid = get_uuid(obj)
                            new_uuid = str(uuid4())
                            # SignalObj/ImageObj store UUID via metadata option
                            try:
                                obj.set_metadata_option("uuid", new_uuid)
                            except AttributeError:
                                obj.uuid = new_uuid
                            uuid_remap[pstr][old_uuid] = new_uuid
                            panel.add_object(
                                obj, get_uuid(group), set_current=False
                            )
                            imported.append(obj)
                        panel.selection_changed()
        # 2. Remap source UUIDs in imported objects' processing_parameters so
        #    that reprocessing in the Processing tab uses the imported sources,
        #    not the originals (same logic as duplicate_selected_entries).
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            PROCESSING_PARAMETERS_OPTION,
            ProcessingParameters,
        )

        for pstr, objs in imported_by_pstr.items():
            pmap = uuid_remap.get(pstr, {})
            if not pmap:
                continue
            for obj in objs:
                try:
                    pp_dict = obj.get_metadata_option(PROCESSING_PARAMETERS_OPTION)
                except (AttributeError, ValueError):
                    continue
                if not pp_dict:
                    continue
                try:
                    pp = ProcessingParameters.from_dict(pp_dict)
                except Exception:  # pylint: disable=broad-except
                    continue
                changed = False
                if pp.source_uuid is not None and pp.source_uuid in pmap:
                    pp.source_uuid = pmap[pp.source_uuid]
                    changed = True
                if pp.source_uuids is not None:
                    new_src = [pmap.get(u, u) for u in pp.source_uuids]
                    if new_src != pp.source_uuids:
                        pp.source_uuids = new_src
                        changed = True
                if changed:
                    try:
                        obj.set_metadata_option(
                            PROCESSING_PARAMETERS_OPTION, pp.to_dict()
                        )
                    except (AttributeError, ValueError):
                        pass
        # 3. Import history sessions as new independent sessions whose captured
        #    UUIDs are remapped to the imported objects.
        if self.H5_PREFIX not in reader.h5:
            return
        sessions = reader.read_object_list(self.H5_PREFIX, HistorySession) or []
        imported_suffix = _("Imported")
        new_sessions: list[HistorySession] = []
        for session in sessions:
            self.__session_increment += 1
            title = f"{session.title} {imported_suffix}"
            new_session = session.copy_with_uuid_remap(
                title=title, uuid_remap=uuid_remap
            )
            new_session.number = self.__session_increment
            new_sessions.append(new_session)
            # Register output mappings for imported actions so that
            # _resolve_target_outputs / get_downstream_actions work.
            for action in new_session.actions:
                if action.output_uuids:
                    self._action_output_uuids[action.uuid] = list(
                        action.output_uuids
                    )
                    for out_uuid in action.output_uuids:
                        self._output_to_action[out_uuid] = action.uuid
        self.__history_sessions.extend(new_sessions)
        self.tree.populate_tree(self.__history_sessions)
        self.refresh_compatibility_items()
        self.__update_actions_state()

    def refresh_compatibility_items(self, *args: Any) -> None:
        """Refresh action item compatibility markers in the tree."""
        del args
        self.tree.update_compatibility_states(self.__history_sessions, self.mainwindow)

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize whole panel to a HDF5 file

        Args:
            writer: HDF5 writer
        """
        writer.write_object_list(self.__history_sessions, self.H5_PREFIX)

    def deserialize_from_hdf5(
        self, reader: NativeH5Reader, reset_all: bool = False
    ) -> None:
        """Deserialize whole panel from a HDF5 file

        Args:
            reader: HDF5 reader
            reset_all: Unused (kept for compatibility with panel API)
        """
        if self.H5_PREFIX not in reader.h5:
            self.__history_sessions = []
            self.__session_increment = 0
            self.tree.populate_tree(self.__history_sessions)
            self.__update_actions_state()
            return
        self.__history_sessions: list[HistorySession] = (
            reader.read_object_list(self.H5_PREFIX, HistorySession) or []
        )
        if self.__history_sessions:
            self.__session_increment = self.__history_sessions[-1].number
        # Rebuild the bijective mapping from the loaded actions. Legacy
        # (v1) actions have empty ``output_uuids`` and contribute nothing
        # to the index — the heuristic fallback handles them.
        self._action_output_uuids = {}
        self._output_to_action = {}
        for session in self.__history_sessions:
            for action in session.actions:
                if action.output_uuids:
                    self._action_output_uuids[action.uuid] = list(
                        action.output_uuids
                    )
                    for out_uuid in action.output_uuids:
                        self._output_to_action[out_uuid] = action.uuid
        self.tree.populate_tree(self.__history_sessions)
        self.refresh_compatibility_items()
        self.__update_actions_state()

    def __len__(self) -> int:
        """Return number of objects"""
        return sum(len(session.actions) for session in self.__history_sessions)

    def __getitem__(self, nb: int) -> HistoryAction:
        """Return object from its number (1 to N)"""
        for session in self.__history_sessions:
            if nb <= len(session.actions):
                return session.actions[nb - 1]
            nb -= len(session.actions)
        raise IndexError("Index out of range")

    def __iter__(self) -> Generator[HistoryAction, None, None]:
        """Iterate over objects"""
        for session in self.__history_sessions:
            yield from session.actions

    def create_new_session(self) -> None:
        """Create a new history list"""
        self.__session_increment += 1
        session = HistorySession(number=self.__session_increment)
        self.__history_sessions.append(session)
        self.tree.populate_tree(self.__history_sessions)
        self.refresh_compatibility_items()

    def start_new_session_after_workspace_reset(self) -> None:
        """Start a new history session after a workspace reset, when useful."""
        if self.__history_sessions and self.__history_sessions[-1].actions:
            self.create_new_session()

    def add_compute_entry(
        self,
        action_title: str,
        panel_str: str,
        func_name: str,
        pattern: str,
        save_state: bool = True,
        output_uuids: list[str] | None = None,
        plugin_origin: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HistoryAction | None:
        """Record a *compute* action in the current history session.

        Args:
            action_title: Title shown in the history tree.
            panel_str: ``"signal"`` or ``"image"``.
            func_name: Sigima feature name (resolvable via
             :meth:`BaseProcessor.get_feature`).
            pattern: One of ``"1_to_1"``, ``"1_to_0"``, ``"n_to_1"``, ``"2_to_1"``,
             ``"1_to_n"``, ``"multiple_1_to_1"`` (the latter is recorded for
             traceability but not replayable).
            save_state: If True, capture the workspace state for replay.
            output_uuids: Optional list of UUIDs of the data objects produced by
             this action. When known at call time, prefer passing it here so the
             bijective mapping is initialised in one step. Most callers do not
             know the outputs yet and instead wrap the compute call with
             :meth:`capture_outputs` (or call :meth:`register_action_outputs`
             explicitly afterwards) using the returned action.
            plugin_origin: Optional plugin origin descriptor (see
             :func:`datalab.gui.processor.base._detect_plugin_origin`). ``None``
             for built-in Sigima/DataLab features.
            **kwargs: Extra primitive kwargs (``param``, ``obj2_uuids``,
             ``obj2_name``, ``pairwise``, ``params`` (list of DataSet),
             ``func_names`` (list of str), ...). ``DataSet`` instances are
             serialised as JSON.

        Returns:
            The created :class:`HistoryAction`, or ``None`` if recording is
            disabled (record mode off or replay in progress).
        """
        if not self.__record_mode or self.__replaying:
            return None
        state = WorkspaceState()
        if save_state:
            state.save(self.mainwindow)
        # Deep-copy kwargs so each action owns independent parameter
        # instances. Without this, consecutive applications of the same
        # function (e.g. two gaussian_filter calls with different sigma)
        # would share a single DataSet object and editing one action's
        # parameters would silently mutate the other.
        action = HistoryAction(
            title=action_title,
            kind=HistoryAction.KIND_COMPUTE,
            panel_str=panel_str,
            func_name=func_name,
            pattern=pattern,
            kwargs=deepcopy(kwargs),
            state=state,
            plugin_origin=plugin_origin,
        )
        self.add_object(action)
        if output_uuids is not None:
            self.register_action_outputs(action, output_uuids)
        return action

    def add_compute_entry_from_pp(
        self,
        action_title: str,
        pp: Any,  # ProcessingParameters (avoid circular import)
        panel_str: str,
        save_state: bool = True,
        output_uuids: list[str] | None = None,
        plugin_origin: dict[str, Any] | None = None,
        **extras: Any,
    ) -> HistoryAction | None:
        """Record a *compute* action derived from a ``ProcessingParameters``.

        Bridges the dash-form pattern used in object metadata
        (``"1-to-1"`` …) with the underscore form expected by
        :class:`HistoryAction` (``"1_to_1"`` …) so that both sides share
        a single identity (``func_name`` / ``pattern`` / ``param``).

        Args:
            action_title: Title shown in the history tree.
            pp: :class:`~datalab.gui.processor.base.ProcessingParameters`
             instance describing the operation.
            panel_str: ``"signal"`` or ``"image"``.
            save_state: If True, capture the workspace state for replay.
            output_uuids: Optional list of UUIDs of the data objects produced
             by this action (see :meth:`add_compute_entry`).
            plugin_origin: Optional plugin origin descriptor (see
             :meth:`add_compute_entry`).
            **extras: Additional history-only kwargs (``obj2_uuids``,
             ``obj2_name``, ``pairwise``, ``params``, ``func_names``…).

        Returns:
            The created :class:`HistoryAction`, or ``None`` if recording is
            disabled.
        """
        hist_pattern = pp.pattern.replace("-", "_")
        kwargs: dict[str, Any] = {}
        if pp.param is not None and "param" not in extras and "params" not in extras:
            kwargs["param"] = pp.param
        kwargs.update(extras)
        return self.add_compute_entry(
            action_title,
            panel_str=panel_str,
            func_name=pp.func_name,
            pattern=hist_pattern,
            save_state=save_state,
            output_uuids=output_uuids,
            plugin_origin=plugin_origin,
            **kwargs,
        )

    def register_action_outputs(
        self, action: HistoryAction, output_uuids: list[str]
    ) -> None:
        """Register the data objects produced by ``action``.

        Maintains the bijective ``action → outputs`` and ``output → action``
        mappings. May be called multiple times for a given action (later calls
        replace earlier ones, e.g. after a cascade recompute).

        Args:
            action: The history action that produced the outputs.
            output_uuids: UUIDs of the produced data objects (empty for
             ``1_to_0`` analysis patterns and for UI actions that did not
             create new objects).
        """
        # Drop previous outputs for this action from the reverse index.
        previous = self._action_output_uuids.get(action.uuid, [])
        for prev_uuid in previous:
            if self._output_to_action.get(prev_uuid) == action.uuid:
                self._output_to_action.pop(prev_uuid, None)
        new_outputs = list(output_uuids)
        # Ownership transfer: if an output_uuid already belongs to a
        # *different* action, remove it from that action's output list so the
        # forward mapping stays consistent.  The HistoryAction object's
        # ``output_uuids`` attribute is NOT updated here because traversing all
        # sessions to locate the object would be expensive; the panel-level
        # dicts are the source of truth.
        for out_uuid in new_outputs:
            old_action_uuid = self._output_to_action.get(out_uuid)
            if old_action_uuid is not None and old_action_uuid != action.uuid:
                old_list = self._action_output_uuids.get(old_action_uuid)
                if old_list is not None:
                    try:
                        old_list.remove(out_uuid)
                    except ValueError:
                        pass
                    if not old_list:
                        del self._action_output_uuids[old_action_uuid]
                _logger.debug(
                    "Output %s transferred from action %s to %s",
                    out_uuid,
                    old_action_uuid,
                    action.uuid,
                )
        action.output_uuids = list(new_outputs)
        self._action_output_uuids[action.uuid] = new_outputs
        for out_uuid in new_outputs:
            self._output_to_action[out_uuid] = action.uuid

    @contextmanager
    def capture_outputs(
        self, action: HistoryAction | None
    ) -> Generator[None, None, None]:
        """Context manager: snapshot panel object IDs and record diffs as outputs.

        Use around any compute call when the produced UUIDs are not known
        upfront. On exit, every newly-added object (signal or image) is
        registered as an output of ``action`` via
        :meth:`register_action_outputs`. No-op when ``action`` is ``None``
        (recording disabled).

        Args:
            action: The history action being processed, or ``None``.
        """
        if action is None:
            yield
            return
        panels = (self.mainwindow.signalpanel, self.mainwindow.imagepanel)
        before = {
            p.PANEL_STR_ID: set(p.objmodel.get_object_ids()) for p in panels
        }
        try:
            yield
        finally:
            new_uuids: list[str] = []
            for p in panels:
                before_p = before[p.PANEL_STR_ID]
                for uid in p.objmodel.get_object_ids():
                    if uid not in before_p:
                        new_uuids.append(uid)
            self.register_action_outputs(action, new_uuids)

    def _prune_output_mapping(self) -> None:
        """Drop entries of :attr:`_output_to_action` whose object no longer exists.

        Connected to each data panel's ``SIG_OBJECT_REMOVED`` so that the
        reverse index stays consistent with the live workspace. The forward
        ``_action_output_uuids`` mapping is intentionally left intact: it
        records the *historical* outputs of each action (useful for replay
        and cascade introspection even after an output was deleted).
        """
        if not self._output_to_action:
            return
        alive: set[str] = set()
        for panel in (self.mainwindow.signalpanel, self.mainwindow.imagepanel):
            alive.update(panel.objmodel.get_object_ids())
        stale = [u for u in self._output_to_action if u not in alive]
        for u in stale:
            self._output_to_action.pop(u, None)

    def _reconnect_chain_after_removal(self, panel: BaseDataPanel) -> None:
        """Reconnect the processing chain after object(s) were deleted from a
        data panel, like removing a link from a linked list.

        Each deleted object that was an intermediate processing result has its
        downstream consumers reconnected to its own source (the first source for
        multi-source operations) and recomputed in cascade, so the chain keeps
        producing consistent results. Cases that cannot be reconnected (the
        deleted object has no valid source) are reported in a single warning but
        the deletion is always kept.

        Connected to each data panel's ``SIG_OBJECT_REMOVED`` (bound to the
        panel via ``functools.partial``). Runs before :meth:`_prune_output_mapping`
        so the bijective output map is still available.
        """
        pstr = panel.PANEL_STR_ID
        previous = self.__obj_ids_snapshot.get(pstr, set())
        current = set(panel.objmodel.get_object_ids())
        removed = previous - current
        if not removed or self.__reconnecting:
            return
        self.__reconnecting = True
        try:
            warnings: list[str] = []
            roots_to_recompute: list[HistoryAction] = []
            for x_uuid in removed:
                self.__reconnect_single_removed(
                    panel, x_uuid, warnings, roots_to_recompute
                )
            for action in roots_to_recompute:
                self._recompute_action_in_place(action)
                self.recompute_cascade(action)
            if warnings:
                QW.QMessageBox.warning(
                    self.mainwindow,
                    _("Delete"),
                    _(
                        "Some operations could not be reconnected after "
                        "deletion:"
                    )
                    + "\n\n• "
                    + "\n• ".join(warnings),
                )
            self.tree.populate_tree(self.__history_sessions)
            self.refresh_compatibility_items()
            self.__update_actions_state()
        finally:
            self.__reconnecting = False
            self.__refresh_obj_ids_snapshot()

    def __reconnect_single_removed(
        self,
        panel: BaseDataPanel,
        x_uuid: str,
        warnings: list[str],
        roots_to_recompute: list[HistoryAction],
    ) -> None:
        """Reconnect consumers of a single deleted object ``x_uuid``.

        Appends a localized message to ``warnings`` when reconnection is not
        possible, and appends each consumer's producing action to
        ``roots_to_recompute`` so the caller can recompute the cascade.
        """
        from datalab.gui.processor.base import (  # pylint: disable=import-outside-toplevel
            ProcessingParameters,
            extract_processing_parameters,
            insert_processing_parameters,
        )

        pstr = panel.PANEL_STR_ID
        # 1. Producing action of the deleted object (to learn its source).
        action_a = None
        action_a_uuid = self._output_to_action.get(x_uuid)
        if action_a_uuid is not None:
            for session in self.__history_sessions:
                for a in session.actions:
                    if a.uuid == action_a_uuid:
                        action_a = a
                        break
                if action_a is not None:
                    break
        # 2. Consumers: objects whose processing references x_uuid as a source.
        consumers: list[tuple[Any, Any]] = []
        for obj in panel.objmodel:
            pp = extract_processing_parameters(obj)
            if pp is None:
                continue
            if pp.source_uuid == x_uuid or (
                pp.source_uuids and x_uuid in pp.source_uuids
            ):
                consumers.append((obj, pp))
        if not consumers:
            # Leaf deletion: nothing downstream to reconnect.
            return
        # 3. Source S of the deleted object (first source for multi-source ops).
        s_uuid: str | None = None
        if action_a is not None:
            sel = action_a.state.selection.get(pstr, [])
            if sel:
                s_uuid = sel[0]
        alive_ids = set(panel.objmodel.get_object_ids())
        if s_uuid is None or s_uuid not in alive_ids:
            label = (
                action_a.title
                or action_a.func_name
                if action_a is not None
                else x_uuid
            )
            warnings.append(
                _(
                    "“%s” has dependent operations but no valid source to "
                    "reconnect to — downstream results are left unchanged."
                )
                % label
            )
            return
        # 4. Reconnect each consumer: replace x_uuid -> s_uuid in its pp and in
        #    its producing action's recorded inputs, then queue it for recompute.
        for obj, pp in consumers:
            new_source_uuid = (
                s_uuid if pp.source_uuid == x_uuid else pp.source_uuid
            )
            new_source_uuids = pp.source_uuids
            if pp.source_uuids and x_uuid in pp.source_uuids:
                new_source_uuids = [
                    s_uuid if u == x_uuid else u for u in pp.source_uuids
                ]
            insert_processing_parameters(
                obj,
                ProcessingParameters(
                    func_name=pp.func_name,
                    pattern=pp.pattern,
                    param=pp.param,
                    source_uuid=new_source_uuid,
                    source_uuids=new_source_uuids,
                ),
            )
            if pp.func_name:
                action_b = self.find_action_for_output(
                    get_uuid(obj), pp.func_name
                )
                if action_b is not None:
                    self.__rewrite_action_source(
                        action_b, pstr, x_uuid, s_uuid
                    )
                    if action_b not in roots_to_recompute:
                        roots_to_recompute.append(action_b)
        # 5. Drop the deleted node's action if all its outputs are gone.
        if action_a is not None:
            outs = self._action_output_uuids.get(action_a.uuid, [])
            if not any(o in alive_ids for o in outs):
                self.__remove_single_action(action_a)

    def __rewrite_action_source(
        self,
        action: HistoryAction,
        pstr: str,
        old_uuid: str,
        new_uuid: str,
    ) -> None:
        """Replace ``old_uuid`` with ``new_uuid`` in an action's recorded inputs.

        Updates both the captured selection and the ``obj2_uuids`` kwarg (for
        2_to_1 actions) so future replays/recomputes use the new source.
        """
        sel = action.state.selection.get(pstr)
        if sel:
            action.state.selection[pstr] = [
                new_uuid if u == old_uuid else u for u in sel
            ]
        obj2 = action.kwargs.get("obj2_uuids")
        if isinstance(obj2, str):
            if obj2 == old_uuid:
                action.kwargs["obj2_uuids"] = new_uuid
        elif obj2:
            action.kwargs["obj2_uuids"] = [
                new_uuid if u == old_uuid else u for u in obj2
            ]

    def __remove_single_action(self, action: HistoryAction) -> None:
        """Remove a single action from its session (splice, not truncate).

        Also drops the action's entries from the bijective output maps, and
        removes the parent session if it becomes empty.
        """
        for session in self.__history_sessions:
            if action in session.actions:
                session.actions.remove(action)
                outs = self._action_output_uuids.pop(action.uuid, [])
                for out_uuid in outs:
                    if self._output_to_action.get(out_uuid) == action.uuid:
                        self._output_to_action.pop(out_uuid, None)
                if not session.actions:
                    self.__history_sessions.remove(session)
                break

    def add_ui_entry(
        self,
        action_title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
        **kwargs: Any,
    ) -> None:
        """Record a *UI* action in the current history session.

        Args:
            action_title: Title shown in the history tree.
            target: One of ``"mainwindow"``, ``"signalpanel"``, ``"imagepanel"``,
             ``"historypanel"`` -- attribute path on the main window.
            method_name: Method name to call on ``target`` at replay time.
            save_state: If True, capture the workspace state for replay.
            **kwargs: Method keyword arguments. ``DataSet`` instances are
             serialised as JSON; other values must be HDF5-friendly primitives.
        """
        if not self.__record_mode or self.__replaying:
            return
        state = WorkspaceState()
        if save_state:
            state.save(self.mainwindow)
        # Deep-copy kwargs to ensure independent parameter ownership
        # (same rationale as in add_compute_entry).
        action = HistoryAction(
            title=action_title,
            kind=HistoryAction.KIND_UI,
            target=target,
            method_name=method_name,
            kwargs=deepcopy(kwargs),
            state=state,
        )
        self.add_object(action)

    def add_entry(
        self,
        action_title: str,
        save_state: bool,
        func: Callable,
        **kwargs,
    ) -> None:
        """Legacy entry-point kept as a compatibility shim.

        Most call sites have been migrated to :meth:`add_compute_entry` or
        :meth:`add_ui_entry`. The remaining paths -- and the
        :func:`add_to_history` decorator -- still call ``add_entry`` with a
        bound method; we infer the ``(target, method_name)`` from the bound
        ``func.__self__`` and route to :meth:`add_ui_entry`.
        """
        if not self.__record_mode or self.__replaying:
            return
        target = None
        if hasattr(func, "__self__"):
            target = _resolve_self_target(func.__self__)
        if target is None:
            # Cannot route safely -- skip rather than pickle a Callable.
            return
        self.add_ui_entry(
            action_title,
            target=target,
            method_name=func.__name__,
            save_state=save_state,
            **kwargs,
        )

    # ------ AbstractPanel interface ---------------------------------------------------
    def create_object(self) -> HistoryAction:
        """Create and return object"""
        return HistoryAction()

    def add_object(self, obj: HistoryAction) -> None:
        """Add object to panel"""
        if not self.__history_sessions:
            self.create_new_session()
        self.__history_sessions[-1].add_action(obj)
        self.tree.add_action_to_tree(obj)
        self.tree.rearrange_tree()
        self.refresh_compatibility_items()
        self.__update_actions_state()

    def remove_all_objects(self):
        """Remove all objects"""
        super().remove_all_objects()
        self._action_output_uuids.clear()
        self._output_to_action.clear()
        self.__update_actions_state()
