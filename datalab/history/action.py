# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""HistoryAction model: serialisable description of one recorded operation."""

from __future__ import annotations

import html
import inspect
import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Optional,
    TypeVar,
    overload,
)
from uuid import uuid4

import sigima.proc.image
import sigima.proc.signal
from guidata.dataset.datatypes import DataSet

from datalab.config import _
from datalab.gui import ObjItf
from datalab.history.core import (
    HISTORY_ACTION_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    copy_history_value,
    decode_kwargs,
    encode_kwargs,
    get_datetime_str,
)
from datalab.history.workspace_state import WorkspaceState
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.h5.native import NativeH5Reader, NativeH5Writer

_logger = logging.getLogger(__name__)

T = TypeVar("T")


class DescriptorField(Generic[T]):
    """Typed descriptor forwarding access to an action descriptor field."""

    def __init__(self, name: str) -> None:
        self.name = name

    @overload
    def __get__(
        self, instance: None, owner: type[HistoryAction]
    ) -> DescriptorField[T]: ...

    @overload
    def __get__(self, instance: HistoryAction, owner: type[HistoryAction]) -> T: ...

    def __get__(
        self, instance: HistoryAction | None, owner: type[HistoryAction]
    ) -> DescriptorField[T] | T:
        if instance is None:
            return self
        return getattr(instance.descriptors, self.name)

    def __set__(self, instance: HistoryAction, value: T) -> None:
        setattr(instance.descriptors, self.name, value)


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
    # UI methods that destroy data objects. Replaying these requires that the
    # captured selection still resolves to existing objects (see ``replay_ui``).
    DESTRUCTIVE_METHODS: frozenset[str] = frozenset(
        {"remove_object", "remove_group", "delete_all_objects"}
    )

    @dataclass
    class Descriptors:
        """Operation descriptors persisted by a history action."""

        kind: str
        panel_str: str | None = None
        func_name: str | None = None
        pattern: str | None = None
        target: str | None = None
        method_name: str | None = None
        plugin_origin: dict[str, Any] | None = None

    kind = DescriptorField[str]("kind")
    panel_str = DescriptorField[Optional[str]]("panel_str")
    func_name = DescriptorField[Optional[str]]("func_name")
    pattern = DescriptorField[Optional[str]]("pattern")
    target = DescriptorField[Optional[str]]("target")
    method_name = DescriptorField[Optional[str]]("method_name")
    plugin_origin = DescriptorField[Optional[Dict[str, Any]]]("plugin_origin")

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
        self.descriptors = self.Descriptors(
            kind=kind,
            panel_str=panel_str,
            func_name=func_name,
            pattern=pattern,
            target=target,
            method_name=method_name,
        )
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
        # patterns, for UI actions, and for sessions loaded without output info
        # loaded from disk (the heuristic fallback then takes over).
        self.output_uuids: list[str] = []
        # Plugin origin descriptor for compute actions (None for built-in
        # Sigima/DataLab features). Populated at registration time by
        # :meth:`BaseProcessor.add_feature` and propagated through
        # ``add_compute_entry_from_pp``. See
        # :func:`datalab.gui.processor.base._detect_plugin_origin` for shape.
        # Persisted as a JSON string in HDF5.
        # Transient flag (NOT serialized): set during a cascade recompute to
        # display a "stale" visual marker in the tree. Cleared once the
        # action has been recomputed.
        self.is_stale: bool = False
        # Snapshot of original kwargs before edit-mode modification.
        # Set lazily when the first edit-mode change touches this action.
        # Persisted to HDF5 so that the Restore
        # action still works after a save/reload cycle while Edit mode is
        # active. Cleared by ``discard_snapshot`` (definitive commit when
        # toggling Edit mode off) or ``restore_kwargs`` (Restore button).
        self.saved_kwargs: dict[str, Any] | None = None

    def snapshot_kwargs(self) -> None:
        """Save a copy of the current kwargs as the pre-edit baseline.

        No-op if a snapshot already exists (preserves the original baseline
        across multiple edit-mode replays).
        """
        if self.saved_kwargs is None:
            self.saved_kwargs = {
                key: copy_history_value(value) for key, value in self.kwargs.items()
            }

    def restore_kwargs(self) -> None:
        """Restore kwargs from the saved snapshot and clear the snapshot."""
        if self.saved_kwargs is not None:
            self.kwargs = self.saved_kwargs
            self.saved_kwargs = None

    def discard_snapshot(self) -> None:
        """Discard the saved snapshot (accept current kwargs as definitive)."""
        self.saved_kwargs = None

    @property
    def has_pending_edits(self) -> bool:
        """Return True if this action has unsaved edit-mode changes."""
        return self.saved_kwargs is not None

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
                key: copy_history_value(value) for key, value in self.kwargs.items()
            },
            state=state,
        )
        new_action.plugin_origin = copy_history_value(self.plugin_origin)
        new_action.output_uuids = list(self.output_uuids)
        # Note: saved_kwargs is intentionally NOT propagated to the copy.
        # Copying an action acts as an implicit commit (no pending edits).
        return new_action

    def effective_panel_str(self) -> str:
        """Return the panel this action operates on ("signal"/"image").

        Falls back to the UI ``target`` when ``panel_str`` is unset (the case for
        creation actions such as ``new_object``, whose ``panel_str`` is ``None``
        but whose ``target`` identifies the panel).
        """
        if self.panel_str:
            return self.panel_str
        return {"imagepanel": "image", "signalpanel": "signal"}.get(self.target, "")

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
            pstr = new_action.effective_panel_str()
            pmap = uuid_remap.get(pstr, {})
            rewritten = [pmap.get(u, u) for u in obj2]
            new_action.kwargs["obj2_uuids"] = (
                rewritten[0] if len(rewritten) == 1 else rewritten
            )
        # Rewrite output_uuids — they reference the target panel.
        if new_action.output_uuids:
            pstr = new_action.effective_panel_str()
            pmap = uuid_remap.get(pstr, {})
            new_action.output_uuids = [pmap.get(u, u) for u in new_action.output_uuids]
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
            func = self.resolve_callable()
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            TypeError,
            ValueError,
        ):
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
                    except (AttributeError, KeyError, TypeError, ValueError):
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

    def resolve_target(self, mainwindow: DLMainWindow) -> Any:
        """Resolve the target object (UI kind) from the mainwindow."""
        attr = self.target or "mainwindow"
        if attr == "mainwindow":
            return mainwindow
        if attr == "signalprocessor":
            return mainwindow.signalpanel.processor
        if attr == "imageprocessor":
            return mainwindow.imagepanel.processor
        return getattr(mainwindow, attr)

    def resolve_panel(self, mainwindow: DLMainWindow):
        """Resolve the data panel for a compute action."""
        if self.panel_str == "signal":
            return mainwindow.signalpanel
        if self.panel_str == "image":
            return mainwindow.imagepanel
        raise ValueError(
            f"Unknown panel_str {self.panel_str!r} for compute history action"
        )

    def resolve_callable(self) -> Callable | None:
        """Best-effort lookup of the underlying callable, for description only."""
        if self.kind == self.KIND_COMPUTE and self.func_name:
            for module in (sigima.proc.signal, sigima.proc.image):
                func = getattr(module, self.func_name, None)
                if callable(func):
                    return func
        return None

    def resolve_obj_by_uuid(self, mainwindow: DLMainWindow, uuid: str) -> Any | None:
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
            ctx = nullcontext()
        with ctx:
            self.replay_inner(mainwindow, restore_selection, edit, uuid_remap)

    def replay_inner(
        self,
        mainwindow: DLMainWindow,
        restore_selection: bool,
        edit: bool,
        uuid_remap: dict[str, dict[str, str]],
    ) -> None:
        """Inner replay logic, always called under the replaying guard."""
        if self.kind == self.KIND_COMPUTE and self.pattern == "1_to_0" and not edit:
            return
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
            translated = self.translate_state(uuid_remap)
            if translated.is_current_state_compatible(mainwindow, False):
                translated.restore(mainwindow)
            self.replay_compute(mainwindow, edit, uuid_remap)
        else:
            if restore_selection:
                self.state.restore(mainwindow)
            self.replay_ui(mainwindow, edit, uuid_remap)

    def translate_state(self, uuid_remap: dict[str, dict[str, str]]) -> WorkspaceState:
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

    def replay_compute(
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
        panel = self.resolve_panel(mainwindow)
        processor = panel.processor
        param = self.kwargs.get("param")
        params = self.kwargs.get("params") or []
        paramclass_name = type(param).__name__ if param is not None else None
        if self.pattern == "1_to_n" and params:
            paramclass_name = type(params[0]).__name__
        feature = processor.get_feature(
            self.func_name,
            plugin_origin=self.plugin_origin,
            paramclass_name=paramclass_name,
        )
        run_kwargs: dict[str, Any] = {self.FUNC_EDIT_MODE: edit}

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
                for obj in (self.resolve_obj_by_uuid(mainwindow, u) for u in uuids)
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
            run_kwargs["params"] = params
        else:
            raise ValueError(f"Unknown compute pattern: {self.pattern!r}")
        processor.run_feature(feature, **run_kwargs)

    def replay_ui(
        self,
        mainwindow: DLMainWindow,
        edit: bool,
        uuid_remap: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Replay a UI-kind action by calling ``target.method_name(**kwargs)``."""
        hpanel = mainwindow.historypanel
        if (
            hpanel is not None
            and hpanel.is_output_suppressed()
            and self.method_name in self.UI_CREATION_METHODS
        ):
            return  # Skip creation UI during non-persistent replay
        target = self.resolve_target(mainwindow)
        # Safety guard for destructive UI actions: if the action would delete
        # objects but the captured selection no longer resolves to existing
        # UUIDs in the target panel, skip the call rather than delete whatever
        # is currently selected (which would silently destroy unrelated data).
        if self.method_name in self.DESTRUCTIVE_METHODS:
            if target is None:
                _logger.warning(
                    "Skipping destructive replay '%s': target '%s' not found",
                    self.method_name,
                    self.target,
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
                if not captured & existing_uuids:
                    _logger.warning(
                        "Skipping destructive replay '%s': none of the captured "
                        "UUIDs %s exist in panel '%s' anymore",
                        self.method_name,
                        list(captured),
                        panel_str,
                    )
                    return
        method = getattr(target, self.method_name)
        call_kwargs = dict(self.kwargs)
        # Translate a recorded source object UUID through the session uuid_remap
        # so deterministic replay after save/reload still targets the right object.
        if uuid_remap and self.panel_str and "source_uuid" in call_kwargs:
            panel_map = uuid_remap.get(self.panel_str, {})
            old = call_kwargs["source_uuid"]
            call_kwargs["source_uuid"] = panel_map.get(old, old)
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
        with writer.group("uuid"):
            writer.write(self.uuid)
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
        encoded = encode_kwargs(self.kwargs)
        if encoded:
            with writer.group("kwargs"):
                writer.write_dict(encoded)
        # ``saved_kwargs``: persisted Edit mode snapshot so the Restore button
        # keeps working after save/reload. Group omitted when there are no
        # pending edits.
        if self.saved_kwargs is not None:
            encoded_saved = encode_kwargs(self.saved_kwargs)
            # Write the group unconditionally (even when empty) so that the
            # round-trip preserves the distinction between None (no pending
            # edits) and {} (degenerate empty snapshot, keeps has_pending_edits).
            with writer.group("saved_kwargs"):
                writer.write_dict(encoded_saved)
        # Only emit ``output_uuids`` when non-empty (empty lists skipped to
        # avoid h5py edge cases with empty arrays).
        if self.output_uuids:
            with writer.group("output_uuids"):
                writer.write(list(self.output_uuids))
        # ``plugin_origin``: stored as a JSON string so the HDF5 schema stays
        # trivially round-trippable. Skipped when None.
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
        deserialize_descriptors(self, reader, current)
        deserialize_kwargs_snapshot(self, reader, current)
        deserialize_outputs_plugin_origin(self, reader, current)
        with reader.group("state"):
            self.state.deserialize(reader)
        with reader.group("dtstr"):
            self.dtstr = reader.read_any()


def deserialize_descriptors(
    action: HistoryAction, reader: NativeH5Reader, current: Any
) -> None:
    """Deserialize optional identity and operation descriptors."""
    # ``uuid`` is present only in files written after UUID persistence was
    # added; keep the freshly generated ``action.uuid`` for older files.
    if "uuid" in current.attrs or "uuid" in current:
        with reader.group("uuid"):
            loaded_uuid = reader.read_any()
        if loaded_uuid:
            action.uuid = str(loaded_uuid)
    for attr in ("panel_str", "func_name", "pattern", "target", "method_name"):
        if attr in current.attrs or attr in current:
            with reader.group(attr):
                setattr(action, attr, reader.read_any())
        else:
            setattr(action, attr, None)


def deserialize_kwargs_snapshot(
    action: HistoryAction, reader: NativeH5Reader, current: Any
) -> None:
    """Deserialize call arguments and the optional edit snapshot."""
    if "kwargs" in current.attrs or "kwargs" in current:
        with reader.group("kwargs"):
            raw = reader.read_dict()
        action.kwargs = decode_kwargs(raw)
    else:
        action.kwargs = {}
    # ``saved_kwargs`` group is present only when an Edit mode snapshot
    # exists; otherwise leave it as ``None``.
    if "saved_kwargs" in current.attrs or "saved_kwargs" in current:
        with reader.group("saved_kwargs"):
            raw_saved = reader.read_dict()
        action.saved_kwargs = decode_kwargs(raw_saved)
    else:
        action.saved_kwargs = None


def deserialize_outputs_plugin_origin(
    action: HistoryAction, reader: NativeH5Reader, current: Any
) -> None:
    """Deserialize optional outputs and plugin provenance."""
    # ``output_uuids`` is present only when the action produced outputs;
    # otherwise leave it empty and consumers fall back to the heuristic
    # matcher.
    if "output_uuids" in current.attrs or "output_uuids" in current:
        with reader.group("output_uuids"):
            raw_outputs = reader.read_any()
        if raw_outputs is None:
            action.output_uuids = []
        else:
            action.output_uuids = [str(u) for u in raw_outputs]
    else:
        action.output_uuids = []
    # ``plugin_origin`` is present only for plugin-originated compute
    # actions; otherwise leave it as ``None`` (a replay of a missing plugin
    # function then surfaces a generic ``FeatureNotFoundError``).
    if "plugin_origin" in current.attrs or "plugin_origin" in current:
        with reader.group("plugin_origin"):
            raw_origin = reader.read_any()
        if raw_origin in (None, ""):
            action.plugin_origin = None
        else:
            try:
                action.plugin_origin = json.loads(raw_origin)
            except (TypeError, ValueError):
                _logger.warning(
                    "Failed to decode plugin_origin for action %s; "
                    "falling back to None.",
                    action.uuid,
                )
                action.plugin_origin = None
    else:
        action.plugin_origin = None
