# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Public History panel facade facets backed by cohesive components."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui import historysession_ops as hsess
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import interactive_replay as hreplay
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.panel.history import reconnection as hconnect
from datalab.h5 import history as hio
from datalab.history import HistoryAction, HistorySession

if TYPE_CHECKING:
    from datalab.h5.native import NativeH5Reader, NativeH5Writer


class HistoryRuntimeFacadeMixin:
    """Expose runtime controls required by panels, processors, and tests."""

    def reconnect_chain_after_removal(self, data_panel: Any) -> None:
        """Reconnect chains after objects are removed from a data panel."""
        hconnect.reconnect_chain_after_removal(self, data_panel)

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Enable or disable synchronization with data panel object changes."""
        self.runtime.objects.set_tracking_enabled(enabled)

    @property
    def record_mode_enabled(self) -> bool:
        """Return whether record mode is enabled."""
        return self.runtime.execution.record_mode

    def toggle_edit_mode(self, checked: bool) -> None:
        """Toggle edit mode, committing pending edits when it is disabled."""
        has_pending_edits = any(
            action.has_pending_edits
            for session in self.history_sessions
            for action in session.actions
        )
        if not checked and has_pending_edits:
            reply = (
                QW.QMessageBox.Yes
                if execenv.unattended
                else QW.QMessageBox.question(
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
            )
            if reply != QW.QMessageBox.Yes:
                return
        self.runtime.execution.edit_mode = checked
        if not checked:
            for session in self.history_sessions:
                for action in session.actions:
                    action.discard_snapshot()
        self.ui.update_actions_state()

    def toggle_record_mode(self, checked: bool) -> None:
        """Toggle record mode."""
        self.runtime.execution.record_mode = checked

    def is_edit_mode(self) -> bool:
        """Return whether the History panel is in edit mode."""
        return self.runtime.execution.edit_mode

    @contextmanager
    def replaying(self) -> Generator[None, None, None]:
        """Suppress history capture during the context scope."""
        with self.runtime.execution.replaying():
            yield

    def is_replaying(self) -> bool:
        """Return whether an external replay or recompute is in progress."""
        return self.runtime.execution.replaying_active

    @contextmanager
    def output_suppressed(self) -> Generator[None, None, None]:
        """Suppress compute outputs during the context scope."""
        with self.runtime.execution.output_suppressed():
            yield

    def is_output_suppressed(self) -> bool:
        """Return whether compute outputs must not be added to panels."""
        return self.runtime.execution.output_suppressed_active


class HistoryReplayFacadeMixin:
    """Expose replay and chain lookups consumed outside the history package."""

    def replay_restore_actions(
        self, replay: bool = True, restore_selection: bool = True
    ) -> None:
        """Replay and/or restore selection for selected actions."""
        hreplay.replay_restore_actions(self, replay, restore_selection)

    def replay_step_by_step(self) -> None:
        """Replay the current selection with parameter dialogs."""
        previous = self.runtime.execution.edit_mode
        self.runtime.execution.edit_mode = True
        try:
            self.replay_restore_actions(replay=True, restore_selection=False)
        finally:
            self.runtime.execution.edit_mode = previous
            for session in self.history_sessions:
                for action in session.actions:
                    action.discard_snapshot()
            self.ui.update_actions_state()

    def find_action_for_output(
        self, output_uuid: str, func_name: str
    ) -> HistoryAction | None:
        """Return the action that produced an output UUID."""
        return hchain.find_action_for_output(self, output_uuid, func_name)

    def find_creation_action_for_output(self, output_uuid: str) -> HistoryAction | None:
        """Return the creation action that produced an output UUID."""
        return hchain.find_creation_action_for_output(self, output_uuid)

    def find_analysis_action(
        self, obj_uuid: str, func_name: str
    ) -> HistoryAction | None:
        """Return the matching analysis action for an object UUID."""
        return hchain.find_analysis_action(self, obj_uuid, func_name)

    def action_output_uuid(self, action: HistoryAction) -> str | None:
        """Return the UUID of the object produced by an action."""
        return hchain.action_output_uuid(self, action)

    def refresh_action(self, action: HistoryAction) -> None:
        """Refresh an action after its arguments are mutated."""
        hrec.refresh_action(self, action)

    def recompute_cascade(
        self,
        root_action: HistoryAction,
        descendants: list[HistoryAction] | None = None,
    ) -> None:
        """Recompute descendants of a root action in place."""
        hrec.recompute_cascade(self, root_action, descendants)

    def on_current_panel_changed(self, panel_str: str) -> None:
        """React to a Signal/Image panel switch."""
        self.navigation.on_current_panel_changed(panel_str)


class HistoryRecordingFacadeMixin:
    """Expose history-session recording operations used by application code."""

    def create_new_session(self, panel_str: str | None = None) -> HistorySession:
        """Create a new history session for a data panel."""
        return hsess.create_new_session(self, panel_str=panel_str)

    def start_new_session_after_workspace_reset(self) -> None:
        """Start a history session after a workspace reset when useful."""
        hsess.start_new_session_after_workspace_reset(self)

    def maybe_start_session_for_input(self, *, load: bool = False) -> None:
        """Offer to start a new session before recording an input."""
        hsess.maybe_start_session_for_input(self, load=load)

    @contextmanager
    def session_prompt_suppressed(self) -> Generator[None, None, None]:
        """Suppress the new-session prompt during a batch load."""
        with self.runtime.execution.session_prompt_suppressed():
            yield

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
        """Add a compute entry to history."""
        return hsess.add_compute_entry(
            self,
            action_title,
            panel_str,
            func_name,
            pattern,
            save_state,
            output_uuids,
            plugin_origin,
            **kwargs,
        )

    def add_compute_entry_from_pp(
        self,
        action_title: str,
        pp: Any,
        panel_str: str,
        save_state: bool = True,
        output_uuids: list[str] | None = None,
        plugin_origin: dict[str, Any] | None = None,
        **extras: Any,
    ) -> HistoryAction | None:
        """Add a compute entry built from processing parameters."""
        return hsess.add_compute_entry_from_pp(
            self,
            action_title,
            pp,
            panel_str,
            save_state,
            output_uuids,
            plugin_origin,
            **extras,
        )

    def register_action_outputs(
        self, action: HistoryAction, output_uuids: list[str]
    ) -> None:
        """Register output UUIDs produced by an action."""
        hsess.register_action_outputs(self, action, output_uuids)

    def capture_outputs(
        self, action: HistoryAction | None
    ) -> Generator[None, None, None]:
        """Return a context manager capturing outputs produced by an action."""
        return hsess.capture_outputs(self, action)

    def add_ui_entry(
        self,
        action_title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
        **kwargs: Any,
    ) -> HistoryAction | None:
        """Add a UI entry to history."""
        return hsess.add_ui_entry(
            self, action_title, target, method_name, save_state, **kwargs
        )


class HistoryPersistenceFacadeMixin:
    """Expose standalone and workspace HDF5 persistence operations."""

    def save_to_dlhist_file(self, filename: str | None = None) -> bool:
        """Save history to a standalone history file."""
        return hio.save_to_dlhist_file(self, filename)

    def open_dlhist_file(self, filename: str | None = None) -> bool:
        """Open history from a standalone history file."""
        return hio.open_dlhist_file(self, filename)

    def import_dlhist_into_new_session(self, reader: NativeH5Reader) -> None:
        """Import standalone history into a new session."""
        hio.import_dlhist_into_new_session(self, reader)

    def refresh_compatibility_items(self, *args: Any) -> None:
        """Refresh compatibility icons in the history tree."""
        hio.refresh_compatibility_items(self, *args)

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize history sessions to HDF5."""
        hio.serialize_to_hdf5(self, writer)

    def deserialize_from_hdf5(
        self, reader: NativeH5Reader, reset_all: bool = False
    ) -> None:
        """Deserialize history sessions from HDF5."""
        hio.deserialize_from_hdf5(self, reader, reset_all)
