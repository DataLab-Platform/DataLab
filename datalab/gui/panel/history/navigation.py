# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Selection, active-session, and step navigation for the History panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from datalab.gui.panel.history import chain as hchain
from datalab.history import HistoryAction, HistorySession

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel


class HistoryNavigation:
    """Coordinate history selection, active sessions, and step navigation."""

    def __init__(self, panel: HistoryPanel) -> None:
        self.panel = panel
        self.syncing = False
        self.active_session_by_panel: dict[str, HistorySession] = {}
        self.session_increment = 0

    def current_action(self) -> HistoryAction | None:
        """Return the action currently selected in the tree."""
        item = self.panel.tree.currentItem()
        if item is None or item.parent() is None:
            return None
        uuid = item.data(0, QC.Qt.UserRole)
        try:
            return self.panel.tree.get_action_from_uuid(
                uuid, self.panel.history_sessions
            )
        except ValueError:
            return None

    def current_panel_str(self) -> str:
        """Return the current data panel identifier, defaulting to signal."""
        panel_str = self.panel.mainwindow.get_current_panel()
        return panel_str if panel_str in ("signal", "image") else "signal"

    def sync_panel_selection(self) -> None:
        """Synchronize data-panel selection from the selected history item."""
        if self.panel.runtime.execution.replaying_active or self.syncing:
            return
        item = self.panel.tree.currentItem()
        if item is None or not item.isSelected():
            return
        if item.parent() is None:
            index = self.panel.tree.indexOfTopLevelItem(item)
            if index < 0 or index >= len(self.panel.history_sessions):
                return
            session = self.panel.history_sessions[index]
            action = next(
                (
                    candidate
                    for candidate in session.actions
                    if candidate.kind == HistoryAction.KIND_COMPUTE
                ),
                None,
            )
        else:
            action = self.current_action()
        if action is None:
            return
        data_panel = hchain.resolve_panel_for_action(self.panel, action)
        if data_panel is None:
            return
        output_uuid = hchain.find_output_object_uuid(self.panel, data_panel, action)
        target_uuids = (
            [output_uuid]
            if output_uuid is not None
            else hchain.existing_input_uuids(data_panel, action)
        )
        if not target_uuids:
            return
        self.syncing = True
        try:
            with QC.QSignalBlocker(data_panel.objview):
                data_panel.objview.select_objects(target_uuids)
            self.panel.mainwindow.set_current_panel(data_panel)
        finally:
            self.syncing = False

    def update_state_widget(self) -> None:
        """Display the workspace state of the selected action."""
        action = self.current_action()
        self.panel.ui.state_widget.update_from_state(
            action.state if action is not None else None
        )

    def session_panel_str(self, session: HistorySession) -> str | None:
        """Return the data panel to which a session belongs."""
        for action in session.actions:
            if action.panel_str:
                return action.panel_str
        for panel_str, active_session in self.active_session_by_panel.items():
            if active_session is session:
                return panel_str
        return None

    def get_active_session(self, panel_str: str) -> HistorySession | None:
        """Return the valid active recording session for a data panel."""
        session = self.active_session_by_panel.get(panel_str)
        if session is not None and session in self.panel.history_sessions:
            return session
        return None

    def set_active_session(
        self, session: HistorySession, panel_str: str | None = None
    ) -> None:
        """Mark a session as active for its data panel."""
        target = panel_str or self.session_panel_str(session)
        if target:
            self.active_session_by_panel[target] = session
            self.refresh_active_session_highlight()

    def refresh_active_session_highlight(self) -> None:
        """Highlight each data panel's active session in the tree."""
        active = {
            session.number: panel_str
            for panel_str, session in self.active_session_by_panel.items()
            if session in self.panel.history_sessions
        }
        self.panel.tree.set_active_sessions(active)

    def set_active_session_from_selection(self) -> None:
        """Make the selected session active while recording."""
        if not self.panel.record_mode_enabled:
            return
        item = self.panel.tree.currentItem()
        if item is None or not item.isSelected():
            return
        if item.parent() is None:
            index = self.panel.tree.indexOfTopLevelItem(item)
            if not 0 <= index < len(self.panel.history_sessions):
                return
            session = self.panel.history_sessions[index]
        else:
            action = self.current_action()
            session = (
                hchain.find_parent_session(self.panel, action)
                if action is not None
                else None
            )
        if session is not None:
            self.set_active_session(session)

    def on_current_panel_changed(self, panel_str: str) -> None:
        """Bring the current data panel's active recording session into view."""
        if panel_str not in ("signal", "image"):
            return
        self.refresh_active_session_highlight()
        session = self.get_active_session(panel_str)
        if session is not None and session in self.panel.history_sessions:
            index = self.panel.history_sessions.index(session)
            item = self.panel.tree.topLevelItem(index)
            if item is not None:
                self.panel.tree.scrollToItem(item)

    def current_session(self) -> HistorySession | None:
        """Return the session relevant for step navigation."""
        item = self.panel.tree.currentItem()
        if item is not None:
            top = item
            while top.parent() is not None:
                top = top.parent()
            index = self.panel.tree.indexOfTopLevelItem(top)
            if 0 <= index < len(self.panel.history_sessions):
                return self.panel.history_sessions[index]
        return self.panel.history_sessions[-1] if self.panel.history_sessions else None

    def can_step_prev(self) -> bool:
        """Return whether a previous action exists in the current session."""
        session = self.current_session()
        action = self.current_action()
        return bool(
            session is not None
            and session.actions
            and action in session.actions
            and session.actions.index(action) > 0
        )

    def can_step_next(self) -> bool:
        """Return whether a next action exists in the current session."""
        session = self.current_session()
        if session is None or not session.actions:
            return False
        action = self.current_action()
        return (
            action not in session.actions
            or session.actions.index(action) < len(session.actions) - 1
        )

    def select_action_in_tree(self, action: HistoryAction) -> None:
        """Select an action in the history tree."""
        iterator = QW.QTreeWidgetItemIterator(self.panel.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QC.Qt.UserRole) == action.uuid:
                self.panel.tree.clearSelection()
                self.panel.tree.setCurrentItem(item)
                item.setSelected(True)
                return
            iterator += 1

    def step_prev(self) -> None:
        """Select the previous action in the current session."""
        if not self.can_step_prev():
            return
        session = self.current_session()
        action = self.current_action()
        self.select_action_in_tree(session.actions[session.actions.index(action) - 1])
        self.panel.ui.update_actions_state()

    def step_next(self) -> None:
        """Select the next action in the current session."""
        if not self.can_step_next():
            return
        session = self.current_session()
        action = self.current_action()
        target = (
            session.actions[0]
            if action not in session.actions
            else session.actions[session.actions.index(action) + 1]
        )
        self.select_action_in_tree(target)
        self.panel.ui.update_actions_state()

    def select_sessions(self, sessions: list[HistorySession]) -> None:
        """Select top-level tree items matching sessions."""
        self.panel.tree.clearSelection()
        for session in sessions:
            index = self.panel.history_sessions.index(session)
            item = self.panel.tree.topLevelItem(index)
            item.setSelected(True)
            self.panel.tree.setCurrentItem(item)
