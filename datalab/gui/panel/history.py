# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. History panel (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import functools
import html
import os
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator
from uuid import uuid4

from guidata.configtools import get_icon
from guidata.dataset.conv import dataset_to_json, json_to_dataset
from guidata.dataset.datatypes import DataSet
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.gui import ObjItf
from datalab.gui.panel.base import AbstractPanel
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.processor.base import BaseProcessor
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


# Keys used in the kwargs dict to mark DataSet payloads, so that the
# serialization layer can round-trip them as JSON strings instead of pickling
# arbitrary Python objects.
_DATASET_MARKER = "__dataset_json__"
_DATASET_LIST_MARKER = "__dataset_list_json__"


def _encode_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Encode kwargs for HDF5 storage: replace ``DataSet`` and ``list[DataSet]``
    values with marker dicts holding their JSON representation.

    All other values must already be HDF5-friendly primitives (str, int, float,
    bool, list/tuple of the same).

    Args:
        kwargs: Raw kwargs dict (may contain ``DataSet`` instances).

    Returns:
        A new dict with ``DataSet`` values wrapped in marker dicts.
    """
    encoded: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, DataSet):
            encoded[key] = {_DATASET_MARKER: dataset_to_json(value)}
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


def get_datetime_str() -> str:
    """Return current date and time as a string"""
    return QC.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")


def add_to_history(kwargs_names: list[str] = [], title: str | None = None):
    """Method decorator to add the method call to the history panel as a UI entry.

    Args:
        kwargs_names: List of keyword arguments to add to the history action.
         Defaults to [].
        title: Title of the history action. Defaults to None.
    """

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

    def regenerate_uuid(self):
        """Regenerate UUID after loading from a file (no-op: per-action UUID)."""

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

        For DataSet parameters, uses the dataset titles; otherwise falls back to
        the first non-empty line of the full description.
        """
        titles: list[str] = []
        for param in self.__iter_param_kwargs():
            if isinstance(param, DataSet):
                title = param.get_title()
                if title:
                    titles.append(title)
        if titles:
            return ", ".join(titles)
        for line in self.description.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @property
    def description_html(self) -> str:
        """Return rich-text (HTML) description used for the expanded view."""
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
            try:
                # Lazy import to avoid cycles at module import time.
                import sigima.proc.image as sigimg  # noqa: F401
                import sigima.proc.signal as sigsig  # noqa: F401
            except Exception:  # pylint: disable=broad-except
                return None
            for module in (sigsig, sigimg):
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
        elif self.pattern == "2_to_1":
            uuids = self.kwargs.get("obj2_uuids") or []
            if isinstance(uuids, str):
                uuids = [uuids]
            # Translate captured UUIDs through ``uuid_remap`` (session replay).
            # ``uuid_remap`` keys are ``panel.PANEL_STR`` (matches
            # ``WorkspaceState.selection`` keys), not the
            # ``HistoryAction.panel_str`` (PANEL_STR_ID).
            panel_map = (uuid_remap or {}).get(panel.PANEL_STR, {})
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
        elif self.pattern == "1_to_n":
            params = self.kwargs.get("params") or []
            run_kwargs["params"] = params
        else:
            raise ValueError(f"Unknown compute pattern: {self.pattern!r}")
        processor.run_feature(feature, **run_kwargs)

    def _replay_ui(self, mainwindow: DLMainWindow, edit: bool) -> None:
        """Replay a UI-kind action by calling ``target.method_name(**kwargs)``."""
        target = self._resolve_target(mainwindow)
        method = getattr(target, self.method_name)
        call_kwargs = dict(self.kwargs)
        # Inject edit mode if the method supports it
        try:
            import inspect

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
        with writer.group("state"):
            self.state.serialize(writer)
        with writer.group("dtstr"):
            writer.write(self.dtstr)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this action."""
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
            selection[panel.PANEL_STR] = [
                get_uuid(obj)
                for obj in panel.objview.get_sel_objects(include_groups=True)
            ]
        return selection

    def save(self, mainwindow: DLMainWindow) -> None:
        """Save the current workspace state

        Args:
            mainwindow: DataLab's main window
        """
        self.selection = self.get_current_selection(mainwindow)
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            sel_uuids = self.selection[panel.PANEL_STR]
            self.states[panel.PANEL_STR] = [
                str(obj.data.shape)
                for obj in panel.objmodel
                if get_uuid(obj) in sel_uuids
            ]
            self.titles[panel.PANEL_STR] = [
                obj.title for obj in panel.objmodel if get_uuid(obj) in sel_uuids
            ]

    def is_current_state_compatible(
        self, mainwindow: DLMainWindow, restore_selection: bool
    ) -> bool:
        """Check if the current workspace state is compatible with the saved state.

        Compatibility means that **every** UUID recorded in the saved selection
        still exists in the corresponding panel. The data shape is no longer
        used to discriminate (it is informative only): a missing UUID is the
        only failure mode.

        Args:
            mainwindow: DataLab's main window
            restore_selection: Unused (kept for API symmetry). With UUID-based
             identity, the compatibility check no longer depends on the current
             selection -- it only depends on object existence.

        Returns:
            True if every saved UUID still exists in its panel, False otherwise.
        """
        if not self.selection:
            return True
        for panel in (mainwindow.signalpanel, mainwindow.imagepanel):
            saved_uuids = self.selection.get(panel.PANEL_STR, [])
            existing_uuids = set(panel.objmodel.get_object_ids())
            for uuid in saved_uuids:
                if uuid not in existing_uuids:
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
            uuids = self.selection.get(panel.PANEL_STR, [])
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

    def add_action(self, action: HistoryAction) -> None:
        """Add an action to the history session

        Args:
            action: Action to add
        """
        self.actions.append(action)

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
        # ``panel.PANEL_STR`` (matches ``WorkspaceState.selection`` keys).
        panels = (mainwindow.signalpanel, mainwindow.imagepanel)
        # Map ``HistoryAction.panel_str`` (PANEL_STR_ID, e.g. ``"signal"``)
        # to the corresponding ``panel.PANEL_STR`` used in remap keys.
        id_to_pstr = {
            mainwindow.signalpanel.PANEL_STR_ID: mainwindow.signalpanel.PANEL_STR,
            mainwindow.imagepanel.PANEL_STR_ID: mainwindow.imagepanel.PANEL_STR,
        }
        uuid_remap: dict[str, dict[str, str]] = {p.PANEL_STR: {} for p in panels}
        # FIFO of newly-created UUIDs not yet claimed by a remap entry --
        # required because most creation UI actions (e.g. ``new_signal``)
        # are recorded with ``save_state=False`` (empty captured selection),
        # so we cannot pair captured-vs-new UUIDs by position at UI time.
        # Subsequent compute actions claim from this queue on demand.
        unclaimed: dict[str, list[str]] = {p.PANEL_STR: [] for p in panels}
        for action in self.actions[:]:
            before = {p.PANEL_STR: set(p.objmodel.get_object_ids()) for p in panels}
            if action.kind == HistoryAction.KIND_COMPUTE:
                # Lazy-resolve any captured UUIDs missing from the remap by
                # claiming from ``unclaimed`` (FIFO, panel-local).
                pstr = id_to_pstr.get(action.panel_str or "", "")
                captured = action.state.selection.get(pstr, [])
                for old_uuid in captured:
                    if old_uuid in uuid_remap.get(pstr, {}):
                        continue
                    queue = unclaimed.get(pstr) or []
                    if queue:
                        uuid_remap.setdefault(pstr, {})[old_uuid] = queue.pop(0)
                # 2_to_1: also claim ``obj2_uuids`` from the queue if unknown.
                if action.pattern == "2_to_1":
                    obj2 = action.kwargs.get("obj2_uuids") or []
                    if isinstance(obj2, str):
                        obj2 = [obj2]
                    for old_uuid in obj2:
                        if old_uuid in uuid_remap.get(pstr, {}):
                            continue
                        queue = unclaimed.get(pstr) or []
                        if queue:
                            uuid_remap.setdefault(pstr, {})[old_uuid] = queue.pop(0)
            action.replay(
                mainwindow,
                restore_selection=restore_selection,
                edit=edit,
                uuid_remap=uuid_remap,
            )
            if action.kind == HistoryAction.KIND_UI:
                for panel in panels:
                    pstr = panel.PANEL_STR
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

    def serialize(self, writer: NativeH5Writer) -> None:
        """Serialize this history session

        Args:
            writer: Writer
        """
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

    @staticmethod
    def action_to_tree_item(action: HistoryAction) -> QW.QTreeWidgetItem:
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
        return item

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
        self.__delete_action: QW.QAction | None = None
        self.__menu_actions: list[QW.QAction] = self.__create_menu_actions()

        self.mainwindow = parent
        self.tree = HistoryTree(self)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.replay_restore_actions)

        toolbar = QW.QToolBar(self)
        add_actions(toolbar, self.__menu_actions)
        widget = QW.QWidget(self)
        layout = QW.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        self.addWidget(widget)

        self.__history_sessions: list[HistorySession] = []
        self.__session_increment = 0
        self.__update_actions_state()

    def __update_actions_state(self) -> None:
        """Update the enabled state of menu actions depending on history content."""
        if self.__delete_action is not None:
            self.__delete_action.setEnabled(len(self) > 0)

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
        record_action = create_action(
            self,
            _("Record mode"),
            toggled=self.toggle_record_mode,
            icon=get_icon("record.svg"),
        )
        record_action.setChecked(self.__record_mode)
        self.__delete_action = create_action(
            self,
            _("Delete"),
            self.delete_actions,
            icon=get_icon("delete.svg"),
        )
        return [
            record_action,
            None,
            create_action(
                self,
                _("Replay"),
                lambda: self.replay_restore_actions(restore_selection=False),
                icon=get_icon("replay.svg"),
            ),
            create_action(
                self,
                _("Restore selection"),
                lambda: self.replay_restore_actions(
                    restore_selection=True, replay=False
                ),
                icon=get_icon("restore_selection.svg"),
            ),
            create_action(
                self,
                _("Restore selection and replay"),
                self.replay_restore_actions,
                icon=get_icon("restore_and_replay.svg"),
            ),
            edit_action,
            None,
            self.__delete_action,
        ]

    def toggle_edit_mode(self, checked: bool) -> None:
        """Toggle edit mode

        Args:
            checked: True if the edit mode is checked, False otherwise
        """
        self.__edit_mode = checked

    def toggle_record_mode(self, checked: bool) -> None:
        """Toggle record mode

        Args:
            checked: True if the record mode is checked, False otherwise
        """
        self.__record_mode = checked

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

    def show_context_menu(self, pos: QC.QPoint) -> None:
        """Show the context menu

        Args:
            pos: Position of the context menu
        """
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
        """Replay and/or restore selection for the selected actions"""
        for session_or_action in self.tree.get_selected_actions_or_sessions(
            self.__history_sessions
        ):
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
                with self.replaying():
                    session_or_action.replay(
                        self.mainwindow,
                        restore_selection=restore_selection,
                        edit=self.__edit_mode,
                    )
            elif restore_selection:
                session_or_action.restore(self.mainwindow)

    def delete_actions(self) -> None:
        """Delete the selected actions"""
        # Ask for confirmation as this will delete the action and all subsequent actions
        reply = QW.QMessageBox.question(
            self.mainwindow,
            _("Delete actions"),
            _(
                "Do you really want to delete the selected action "
                "and all the next ones?"
            ),
            QW.QMessageBox.Yes | QW.QMessageBox.No,
            QW.QMessageBox.No,
        )
        if reply == QW.QMessageBox.Yes:
            for action in self.tree.get_selected_actions(self.__history_sessions):
                for session in self.__history_sessions:
                    if action in session.actions:
                        session.remove_action(action)
            self.tree.populate_tree(self.__history_sessions)
            self.__update_actions_state()

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
        self.__history_sessions: list[HistorySession] = (
            reader.read_object_list(self.H5_PREFIX, HistorySession) or []
        )
        if self.__history_sessions:
            self.__session_increment = self.__history_sessions[-1].number
        self.tree.populate_tree(self.__history_sessions)
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
            for action in session.actions:
                yield action

    def create_new_session(self) -> None:
        """Create a new history list"""
        self.__session_increment += 1
        session = HistorySession(number=self.__session_increment)
        self.__history_sessions.append(session)
        self.tree.populate_tree(self.__history_sessions)

    def add_compute_entry(
        self,
        action_title: str,
        panel_str: str,
        func_name: str,
        pattern: str,
        save_state: bool = True,
        **kwargs: Any,
    ) -> None:
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
            **kwargs: Extra primitive kwargs (``param``, ``obj2_uuids``,
             ``obj2_name``, ``pairwise``, ``params`` (list of DataSet),
             ``func_names`` (list of str), ...). ``DataSet`` instances are
             serialised as JSON.
        """
        if not self.__record_mode or self.__replaying:
            return
        state = WorkspaceState()
        if save_state:
            state.save(self.mainwindow)
        action = HistoryAction(
            title=action_title,
            kind=HistoryAction.KIND_COMPUTE,
            panel_str=panel_str,
            func_name=func_name,
            pattern=pattern,
            kwargs=kwargs,
            state=state,
        )
        self.add_object(action)

    def add_compute_entry_from_pp(
        self,
        action_title: str,
        pp: Any,  # ProcessingParameters (avoid circular import)
        panel_str: str,
        save_state: bool = True,
        **extras: Any,
    ) -> None:
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
            **extras: Additional history-only kwargs (``obj2_uuids``,
             ``obj2_name``, ``pairwise``, ``params``, ``func_names``…).
        """
        hist_pattern = pp.pattern.replace("-", "_")
        kwargs: dict[str, Any] = {}
        if pp.param is not None and "param" not in extras and "params" not in extras:
            kwargs["param"] = pp.param
        kwargs.update(extras)
        self.add_compute_entry(
            action_title,
            panel_str=panel_str,
            func_name=pp.func_name,
            pattern=hist_pattern,
            save_state=save_state,
            **kwargs,
        )

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
        action = HistoryAction(
            title=action_title,
            kind=HistoryAction.KIND_UI,
            target=target,
            method_name=method_name,
            kwargs=kwargs,
            state=state,
        )
        self.add_object(action)

    def add_entry(
        self,
        action_title: str,
        save_state: bool,
        func: Callable,
        *args,
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
        self.__update_actions_state()

    def remove_all_objects(self):
        """Remove all objects"""
        super().remove_all_objects()
        self.__update_actions_state()
