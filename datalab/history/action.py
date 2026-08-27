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
    List,
    Optional,
    TypeVar,
    overload,
)
from uuid import uuid4

import sigima.proc.image
import sigima.proc.signal
from guidata.dataset.datatypes import DataSet
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui import ObjItf
from datalab.history.core import (
    HISTORY_ACTION_SCHEMA_VERSION,
    HistoryDecodeError,
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
    method on a known target: ``mainwindow``, ``signalpanel``, ``imagepanel``,
    ``historypanel``, ``signalprocessor``, or ``imageprocessor``).

    No Python ``Callable`` is ever pickled: a compute action is identified by
    ``(panel_str, func_name, pattern)`` and a UI action by ``(target,
    method_name)``. ``DataSet`` payloads inside ``kwargs`` are serialised with
    :func:`guidata.dataset.conv.dataset_to_json`.
    """

    KIND_COMPUTE = "compute"
    KIND_UI = "ui"
    # Mutation actions describe in-place modifications of existing data objects
    # (no new objects created), e.g. ROI assignment/removal.
    KIND_MUTATION = "mutation"
    # Mutation keys currently supported by ``replay_mutation``.
    SUPPORTED_MUTATION_KEYS: frozenset[str] = frozenset({"roi"})

    FUNC_EDIT_MODE = "edit"  # Name of the function parameter to enable edit mode
    # Object-creation actions skipped during non-persistent (output-suppressed)
    # replay so the panel object count stays stable.
    UI_CREATION_METHODS: frozenset[str] = frozenset({"new_object"})
    # UI methods that (re)load objects from disk into the workspace.
    UI_LOAD_METHODS: frozenset[str] = frozenset(
        {"load_from_files", "load_from_directory"}
    )
    # UI methods that destroy data objects. Replaying these requires that the
    # captured selection still resolves to existing objects (see ``replay_ui``).
    DESTRUCTIVE_METHODS: frozenset[str] = frozenset(
        {"remove_object", "remove_group", "delete_all_objects"}
    )
    # UI methods that write files on disk. Replaying them silently would
    # overwrite user files, so the replay engine asks for confirmation.
    FILE_OUTPUT_METHODS: frozenset[str] = frozenset(
        {"save_to_h5_file", "save_to_files", "save_to_directory"}
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
        # Mutation-only descriptors (``kind == KIND_MUTATION``):
        mutation_key: str | None = None
        target_uuids: list[str] | None = None

    kind = DescriptorField[str]("kind")
    panel_str = DescriptorField[Optional[str]]("panel_str")
    func_name = DescriptorField[Optional[str]]("func_name")
    pattern = DescriptorField[Optional[str]]("pattern")
    target = DescriptorField[Optional[str]]("target")
    method_name = DescriptorField[Optional[str]]("method_name")
    plugin_origin = DescriptorField[Optional[Dict[str, Any]]]("plugin_origin")
    mutation_key = DescriptorField[Optional[str]]("mutation_key")
    target_uuids = DescriptorField[Optional[List[str]]]("target_uuids")

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
        # --- mutation-only ---------------------------------------------------
        mutation_key: str | None = None,
        target_uuids: list[str] | None = None,
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
            mutation_key=mutation_key,
            target_uuids=target_uuids,
        )
        # Common:
        self.kwargs: dict[str, Any] = (
            {} if kwargs is None else {k: v for k, v in kwargs.items() if v is not None}
        )
        self.state = WorkspaceState() if state is None else state
        self.dtstr: str = get_datetime_str()
        self.uuid: str = str(uuid4())
        self.schema_version: int = HISTORY_ACTION_SCHEMA_VERSION
        # UUIDs of the data objects produced by this action. One action may have
        # multiple outputs; the history runtime also maintains an inverse output
        # lookup. Populated after output-producing compute or UI actions via
        # :meth:`HistoryPanel.register_action_outputs`. Empty for ``1_to_0``
        # patterns, UI actions without new objects, and legacy actions/files
        # lacking output information (the heuristic fallback then takes over).
        self.output_uuids: list[str] = []
        # Plugin origin descriptor for compute actions (None for built-in
        # Sigima/DataLab features). Populated at registration time by
        # :meth:`BaseProcessor.add_feature` and propagated through
        # ``add_compute_entry_from_pp``. See
        # :func:`datalab.gui.processor.base._detect_plugin_origin` for shape.
        # Persisted as a JSON string in HDF5.

        # Analysis effects manifest, keyed by source object UUID, values are
        # ``AnalysisEffects.to_dict()`` payloads. Persisted as a JSON string.
        # Only populated for 1_to_0 compute actions, None otherwise.
        self.effects: dict[str, dict] | None = None
        # Transient flag (NOT serialized): set during a cascade recompute to
        # display a "stale" visual marker in the tree. Cleared once the
        # action has been recomputed.
        self.is_stale: bool = False
        # Transient flag (NOT serialized): set when a persisted kwarg payload
        # could not be decoded at load time (e.g. corrupt or untrusted ROI).
        # A broken action is permanently incompatible (see
        # :meth:`is_current_state_compatible`) so it can never be replayed
        # with altered semantics.
        self.decode_failed: bool = False
        # Snapshot of original kwargs before edit-mode modification.
        # Set lazily when the first edit-mode change touches this action.
        # Persisted to HDF5 so the pre-edit values remain available after a
        # save/reload cycle while Edit mode is active. Cleared by
        # ``discard_snapshot`` (definitive commit when toggling Edit mode off)
        # or ``restore_kwargs`` during parameter rollback.
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
            mutation_key=self.mutation_key,
            target_uuids=list(self.target_uuids) if self.target_uuids else None,
            kwargs={
                key: copy_history_value(value) for key, value in self.kwargs.items()
            },
            state=state,
        )
        new_action.plugin_origin = copy_history_value(self.plugin_origin)
        new_action.output_uuids = list(self.output_uuids)
        new_action.effects = copy_history_value(self.effects)
        # A broken action stays broken: its copy must not become replayable.
        new_action.decode_failed = self.decode_failed
        # Note: saved_kwargs is intentionally NOT propagated to the copy.
        # Copying an action acts as an implicit commit (no pending edits).
        return new_action

    def effective_panel_str(self) -> str:
        """Return the panel this action operates on ("signal"/"image").

        Falls back to the UI ``target`` when ``panel_str`` is unset. This covers
        creation actions targeting a data panel and legacy actions targeting a
        panel processor.
        """
        if self.panel_str:
            return self.panel_str
        return {
            "signalpanel": "signal",
            "signalprocessor": "signal",
            "imagepanel": "image",
            "imageprocessor": "image",
        }.get(self.target, "")

    def copy_with_uuid_remap(
        self, uuid_remap: dict[str, dict[str, str]]
    ) -> HistoryAction:
        """Return a copy with supported object UUID references rewritten.

        State selections and metadata, ``obj2_uuids``, and ``output_uuids`` are
        remapped. Other UI keyword arguments are preserved.

        Args:
            uuid_remap: Per-panel mapping ``{panel_str: {old_uuid: new_uuid}}``
             used to translate captured UUIDs to the cloned objects created by
             the Duplicate operation.

        Returns:
            A new independent :class:`HistoryAction` with supported UUID
             references remapped.
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
        # Rewrite target_uuids — mutated objects live in the target panel.
        if new_action.target_uuids:
            pstr = new_action.effective_panel_str()
            pmap = uuid_remap.get(pstr, {})
            new_action.target_uuids = [pmap.get(u, u) for u in new_action.target_uuids]
        # Rewrite effects keys — they reference source objects in the target panel.
        if new_action.effects:
            pstr = new_action.effective_panel_str()
            pmap = uuid_remap.get(pstr, {})
            new_action.effects = {
                pmap.get(u, u): payload for u, payload in new_action.effects.items()
            }
        return new_action

    @property
    def title(self) -> str:
        """Return object title"""
        return self.__title

    @title.setter
    def title(self, value: str) -> None:
        """Set object title"""
        self.__title = value or ""

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
        for param in self.__iter_param_kwargs():
            if desc:
                desc += os.linesep
            desc += str(param)
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
        if func is None:
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

    def __roi_exclusions(self) -> set[str] | None:
        """Return the target UUIDs to exclude from ROI comparison, or None.

        Mutation states are captured after the mutation was applied, so the
        targets' ROI signatures cannot be expected to match at replay time.
        """
        if self.kind == self.KIND_MUTATION:
            return set(self.target_uuids or [])
        return None

    def is_current_state_compatible(self, mainwindow: DLMainWindow) -> bool:
        """Check if the current workspace state is compatible with the saved state.

        Mutation actions exclude their own targets from the ROI signature
        comparison (the recorded state contains the post-mutation ROI).

        Actions whose persisted kwargs could not be decoded at load time are
        never compatible: replaying them would use degraded parameters.
        """
        if self.decode_failed:
            return False
        return self.state.is_current_state_compatible(
            mainwindow, ignore_roi_uuids=self.__roi_exclusions()
        )

    def restore(self, mainwindow: DLMainWindow) -> None:
        """Restore the associated workspace state."""
        self.state.restore(mainwindow, ignore_roi_uuids=self.__roi_exclusions())

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

    def resolve_callable(self) -> Callable | None:
        """Best-effort lookup of the underlying callable, for description only."""
        if self.kind == self.KIND_COMPUTE and self.func_name:
            for module in (sigima.proc.signal, sigima.proc.image):
                func = getattr(module, self.func_name, None)
                if callable(func):
                    return func
        return None

    def replay(
        self,
        mainwindow: DLMainWindow,
        restore_selection: bool,
        edit: bool,
    ) -> None:
        """Replay a UI-kind or mutation-kind action.

        Compute-kind actions are recomputed in place by the History panel
        engine (see :mod:`datalab.gui.panel.history.recompute`) and must never
        reach this method.

        Args:
            mainwindow: DataLab's main window
            restore_selection: True to restore the captured workspace selection
             before replaying the action.
            edit: If True, request a parameter dialog for supported actions with
             editable parameters. If False, use the captured parameters.

        Raises:
            NotImplementedError: If called on a compute-kind action.
        """
        if self.kind == self.KIND_COMPUTE:
            raise NotImplementedError(
                "Compute actions are recomputed in place and cannot be replayed."
            )
        # Suppress history capture during replay to avoid recording
        # synthetic entries when the target re-executes features.
        # The context manager is reentrant, so nesting with
        # HistoryPanel.replay_restore_actions() is safe.
        hpanel = getattr(mainwindow, "historypanel", None)
        if hpanel is not None:
            ctx = hpanel.replaying()
        else:
            ctx = nullcontext()
        with ctx:
            if restore_selection:
                self.restore(mainwindow)
            if self.kind == self.KIND_MUTATION:
                self.replay_mutation(mainwindow, edit=edit)
            else:
                self.replay_ui(mainwindow, edit)

    def replay_mutation(
        self, mainwindow: DLMainWindow, edit: bool = False, refresh: bool = True
    ) -> list[str]:
        """Replay a mutation-kind action: re-apply the in-place modification.

        Only ``mutation_key == "roi"`` is supported: the ROI payload (or None
        for a deletion) is re-applied to each target object still present in
        the data panel's object model.

        Args:
            mainwindow: DataLab's main window
            edit: If True, the replay was requested in edit mode. Regions of
             interest are defined with the interactive ROI editor, which
             cannot be reopened with the recorded payload, so nothing is
             edited and the recorded ROI is re-applied as is.
            refresh: If True (default), refresh the panel selection and plot
             after applying the mutation. The cascade engine passes False as
             it refreshes each target itself.

        Returns:
            UUIDs of the target objects that were mutated (empty list when
             the mutation could not be applied to any object).
        """
        if self.mutation_key not in self.SUPPORTED_MUTATION_KEYS:
            _logger.warning(
                "Skipping mutation replay: unsupported mutation key %r",
                self.mutation_key,
            )
            return []
        panel_str = self.effective_panel_str()
        if panel_str == "signal":
            panel_data = mainwindow.signalpanel
        elif panel_str == "image":
            panel_data = mainwindow.imagepanel
        else:
            _logger.warning("Skipping mutation replay: unknown panel %r", panel_str)
            return []
        # A missing "payload" kwarg means None (ROI deletion): encode_kwargs
        # skips None values, so deletion payloads are simply not persisted.
        payload = self.kwargs.get("payload")
        targets = [
            uuid
            for uuid in self.target_uuids or []
            if panel_data.objmodel.has_uuid(uuid)
        ]
        if not targets:
            return []
        if edit and payload is not None and not execenv.unattended:
            QW.QMessageBox.information(
                mainwindow,
                _("Recompute regions of interest"),
                _(
                    "Regions of interest cannot be edited from the History "
                    "panel: the ROI editor cannot be reopened with the "
                    "recorded parameters. The recorded regions of interest "
                    "are kept as is."
                ),
            )
        for uuid in targets:
            obj = panel_data.objmodel[uuid]
            obj.roi = payload.copy() if payload is not None else None
            if hasattr(obj, "mark_roi_as_changed"):
                obj.mark_roi_as_changed()
        if refresh:
            panel_data.selection_changed(update_items=True)
            panel_data.refresh_plot(
                "selected", update_items=True, only_visible=False, only_existing=True
            )
        return targets

    def replay_ui(
        self,
        mainwindow: DLMainWindow,
        edit: bool,
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
        if self.mutation_key is not None:
            with writer.group("mutation_key"):
                writer.write(self.mutation_key)
        # Like ``output_uuids``: only emit when non-empty.
        if self.target_uuids:
            with writer.group("target_uuids"):
                writer.write(list(self.target_uuids))
        encoded = encode_kwargs(self.kwargs)
        if encoded:
            with writer.group("kwargs"):
                writer.write_dict(encoded)
        # Persist the Edit mode baseline across save/reload while edits are pending.
        # The group is omitted when there are no pending edits.
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
        # ``effects``: analysis effects manifest (1_to_0 compute actions only),
        # stored as a JSON string. Skipped when None.
        if self.effects is not None:
            with writer.group("effects"):
                writer.write(json.dumps(self.effects))
        with writer.group("state"):
            self.state.serialize(writer)
        with writer.group("dtstr"):
            writer.write(self.dtstr)

    def deserialize(self, reader: NativeH5Reader) -> None:
        """Deserialize this action."""
        # Legacy files predate per-action schema versions: default to 1
        self.schema_version = reader.read("schema_version", default=1)
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
    for attr in (
        "panel_str",
        "func_name",
        "pattern",
        "target",
        "method_name",
        "mutation_key",
    ):
        if attr in current.attrs or attr in current:
            with reader.group(attr):
                setattr(action, attr, reader.read_any())
        else:
            setattr(action, attr, None)
    # ``target_uuids`` is serialized only when non-empty (mutation actions);
    # legacy files and non-mutation actions leave it as ``None``.
    if "target_uuids" in current.attrs or "target_uuids" in current:
        with reader.group("target_uuids"):
            raw_targets = reader.read_any()
        action.target_uuids = (
            [str(u) for u in raw_targets] if raw_targets is not None else None
        )
    else:
        action.target_uuids = None


def deserialize_kwargs_snapshot(
    action: HistoryAction, reader: NativeH5Reader, current: Any
) -> None:
    """Deserialize call arguments and the optional edit snapshot.

    A payload decode failure (:class:`HistoryDecodeError`) is degraded locally
    instead of aborting the whole file load: kwargs are reset to an empty dict,
    a corrupted edit snapshot is dropped (``saved_kwargs = None``, no rollback
    value) and the action is flagged as broken (``decode_failed``), routing it
    into the existing incompatible-action UX.
    """
    if "kwargs" in current.attrs or "kwargs" in current:
        with reader.group("kwargs"):
            raw = reader.read_dict()
        try:
            action.kwargs = decode_kwargs(raw)
        except HistoryDecodeError as exc:
            # ``decode_kwargs`` already emitted a user-visible warning:
            # keep only a debug trace here to avoid double reporting.
            _logger.debug(
                "Failed to decode kwargs for action %s (%s): %s; "
                "marking the action as incompatible.",
                action.uuid,
                action.func_name or action.method_name or action.title,
                exc,
            )
            action.kwargs = {}
            action.decode_failed = True
    else:
        action.kwargs = {}
    # ``saved_kwargs`` group is present only when an Edit mode snapshot
    # exists; otherwise leave it as ``None``.
    if "saved_kwargs" in current.attrs or "saved_kwargs" in current:
        with reader.group("saved_kwargs"):
            raw_saved = reader.read_dict()
        try:
            action.saved_kwargs = decode_kwargs(raw_saved)
        except HistoryDecodeError as exc:
            _logger.debug(
                "Failed to decode saved_kwargs for action %s (%s): %s; "
                "marking the action as incompatible.",
                action.uuid,
                action.func_name or action.method_name or action.title,
                exc,
            )
            # No usable rollback value: drop the snapshot entirely so
            # ``has_pending_edits`` stays False (``decode_failed`` already
            # marks the damage).
            action.saved_kwargs = None
            action.decode_failed = True
    else:
        action.saved_kwargs = None


def deserialize_outputs_plugin_origin(
    action: HistoryAction, reader: NativeH5Reader, current: Any
) -> None:
    """Deserialize optional outputs, plugin provenance and effects manifest."""
    # ``output_uuids`` is serialized only when non-empty. Outputless actions and
    # legacy files without this field leave it empty, so consumers fall back to
    # the heuristic matcher.
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
    # ``effects`` is present only for 1_to_0 compute actions written with
    # action schema v2+; legacy files leave it as ``None``.
    if "effects" in current.attrs or "effects" in current:
        with reader.group("effects"):
            raw_effects = reader.read_any()
        if raw_effects in (None, ""):
            action.effects = None
        else:
            try:
                action.effects = json.loads(raw_effects)
            except (TypeError, ValueError):
                _logger.warning(
                    "Failed to decode effects for action %s; falling back to None.",
                    action.uuid,
                )
                action.effects = None
    else:
        action.effects = None
