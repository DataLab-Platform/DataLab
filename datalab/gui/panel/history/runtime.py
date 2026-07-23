# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Runtime state and registries for the History panel."""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator

from qtpy import QtCore as QC

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history.panel import HistoryPanel
    from datalab.history import HistoryAction


class HistoryExecutionState:
    """Own transient replay modes and re-entrance guards."""

    def __init__(self) -> None:
        self.record_mode = False
        self.edit_mode = False
        self.session_input_pending = False
        self.suppress_session_prompt = False
        self.replaying_active = False
        self.output_suppressed_active = False
        self.cascade_in_progress = False
        self.edit_replay_in_progress = False
        self.cascade_warnings: list[str] = []
        self.broken_actions: set[str] = set()

    @contextmanager
    def replaying(self) -> Generator[None, None, None]:
        """Suppress history capture during the context scope."""
        previous = self.replaying_active
        self.replaying_active = True
        try:
            yield
        finally:
            self.replaying_active = previous

    @contextmanager
    def output_suppressed(self) -> Generator[None, None, None]:
        """Suppress compute outputs during the context scope."""
        previous = self.output_suppressed_active
        self.output_suppressed_active = True
        try:
            yield
        finally:
            self.output_suppressed_active = previous

    @contextmanager
    def session_prompt_suppressed(self) -> Generator[None, None, None]:
        """Suppress the new-session prompt during the context scope."""
        previous = self.suppress_session_prompt
        self.suppress_session_prompt = True
        try:
            yield
        finally:
            self.suppress_session_prompt = previous

    def start_session_input_prompt(self) -> bool:
        """Start the input prompt debounce window if it is not already active."""
        if self.session_input_pending:
            return False
        self.session_input_pending = True
        QC.QTimer.singleShot(0, self.finish_session_input_prompt)
        return True

    def finish_session_input_prompt(self) -> None:
        """End the input prompt debounce window."""
        self.session_input_pending = False

    @contextmanager
    def recomputing_cascade(self) -> Generator[bool, None, None]:
        """Guard a cascade recomputation against re-entrance."""
        if self.cascade_in_progress:
            yield False
            return
        self.broken_actions.clear()
        self.cascade_in_progress = True
        try:
            yield True
        finally:
            self.cascade_in_progress = False

    @contextmanager
    def replaying_edits(self) -> Generator[bool, None, None]:
        """Guard an edit-mode replay against re-entrance."""
        if self.edit_replay_in_progress:
            yield False
            return
        self.edit_replay_in_progress = True
        try:
            yield True
        finally:
            self.edit_replay_in_progress = False


class HistoryObjectIndex:
    """Own object snapshots, output indexes, and panel tracking callbacks."""

    def __init__(
        self,
        panel: HistoryPanel,
        reconnect_after_removal: Callable[[BaseDataPanel], None],
    ) -> None:
        self.panel = panel
        self.reconnect_after_removal = reconnect_after_removal
        self.reconnecting = False
        self.obj_ids_snapshot: dict[str, set[str]] = {}
        self.action_output_uuids: dict[str, list[str]] = {}
        self.output_to_action: dict[str, str] = {}
        self.tracking_enabled = False
        self.object_tracking_connections: list[tuple[Any, Any]] = []
        self.build_tracking_connections()

    def build_tracking_connections(self) -> None:
        """Build callbacks that keep history state aligned with data panels."""
        for data_panel in (
            self.panel.mainwindow.signalpanel,
            self.panel.mainwindow.imagepanel,
        ):
            self.object_tracking_connections.extend(
                (
                    (
                        data_panel.SIG_OBJECT_ADDED,
                        self.panel.refresh_compatibility_items,
                    ),
                    (data_panel.SIG_OBJECT_ADDED, self.refresh_obj_ids_snapshot),
                    (
                        data_panel.SIG_OBJECT_REMOVED,
                        self.panel.refresh_compatibility_items,
                    ),
                    (
                        data_panel.SIG_OBJECT_REMOVED,
                        functools.partial(self.reconnect_after_removal, data_panel),
                    ),
                    (data_panel.SIG_OBJECT_REMOVED, self.prune_output_mapping),
                    (
                        data_panel.SIG_OBJECT_MODIFIED,
                        self.panel.refresh_compatibility_items,
                    ),
                )
            )

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Enable or disable synchronization with data panel object changes."""
        if enabled == self.tracking_enabled:
            return
        for signal, callback in self.object_tracking_connections:
            if enabled:
                signal.connect(callback)
            else:
                signal.disconnect(callback)
        self.tracking_enabled = enabled

    def refresh_obj_ids_snapshot(self) -> None:
        """Cache the current object ids of both data panels."""
        signal_panel = self.panel.mainwindow.signalpanel
        image_panel = self.panel.mainwindow.imagepanel
        self.obj_ids_snapshot = {
            signal_panel.PANEL_STR_ID: set(signal_panel.objmodel.get_object_ids()),
            image_panel.PANEL_STR_ID: set(image_panel.objmodel.get_object_ids()),
        }

    @contextmanager
    def reconnecting_objects(self) -> Generator[bool, None, None]:
        """Guard object reconnection and refresh snapshots when it completes."""
        if self.reconnecting:
            yield False
            return
        self.reconnecting = True
        try:
            yield True
        finally:
            self.reconnecting = False
            self.refresh_obj_ids_snapshot()

    def register_action_outputs(
        self, action: HistoryAction, output_uuids: list[str]
    ) -> None:
        """Register outputs while maintaining both mapping directions."""
        previous = self.action_output_uuids.get(action.uuid, [])
        for previous_uuid in previous:
            if self.output_to_action.get(previous_uuid) == action.uuid:
                self.output_to_action.pop(previous_uuid, None)
        new_outputs = list(output_uuids)
        for output_uuid in new_outputs:
            old_action_uuid = self.output_to_action.get(output_uuid)
            if old_action_uuid is not None and old_action_uuid != action.uuid:
                old_outputs = self.action_output_uuids.get(old_action_uuid)
                if old_outputs is not None and output_uuid in old_outputs:
                    old_outputs.remove(output_uuid)
                    if not old_outputs:
                        del self.action_output_uuids[old_action_uuid]
        action.output_uuids = list(new_outputs)
        self.action_output_uuids[action.uuid] = new_outputs
        for output_uuid in new_outputs:
            self.output_to_action[output_uuid] = action.uuid

    def prune_output_mapping(self) -> None:
        """Drop reverse-index entries for objects that no longer exist."""
        if not self.output_to_action:
            return
        alive: set[str] = set()
        for data_panel in (
            self.panel.mainwindow.signalpanel,
            self.panel.mainwindow.imagepanel,
        ):
            alive.update(data_panel.objmodel.get_object_ids())
        for output_uuid in [
            uuid for uuid in self.output_to_action if uuid not in alive
        ]:
            action_uuid = self.output_to_action.pop(output_uuid)
            outputs = self.action_output_uuids.get(action_uuid)
            if outputs is not None and output_uuid in outputs:
                outputs.remove(output_uuid)
                if not outputs:
                    del self.action_output_uuids[action_uuid]

    def remove_action_outputs(self, action: HistoryAction) -> None:
        """Remove all output-index entries owned by an action."""
        outputs = self.action_output_uuids.pop(action.uuid, [])
        for output_uuid in outputs:
            if self.output_to_action.get(output_uuid) == action.uuid:
                self.output_to_action.pop(output_uuid, None)

    def clear_output_mappings(self) -> None:
        """Clear both output mapping indexes."""
        self.action_output_uuids.clear()
        self.output_to_action.clear()


class HistoryRuntime:
    """Coordinate transient execution state and history object indexes."""

    def __init__(
        self,
        panel: HistoryPanel,
        reconnect_after_removal: Callable[[BaseDataPanel], None],
    ) -> None:
        self.execution = HistoryExecutionState()
        self.objects = HistoryObjectIndex(panel, reconnect_after_removal)
