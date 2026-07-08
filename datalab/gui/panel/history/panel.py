# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. History panel (see parent package :mod:`datalab.gui.panel`)
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from guidata.widgets.dockable import DockableWidgetMixin
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui import historysession_ops as hsess
from datalab.gui import historytools_ops as htools
from datalab.gui.panel.base import AbstractPanel
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import interactive_replay as hreplay
from datalab.gui.panel.history import recompute as hrec
from datalab.h5 import history as hio
from datalab.history import HistoryAction, HistorySession
from datalab.widgets.historytree import HistoryTree
from datalab.widgets.workspacestate_widget import WorkspaceStateWidget

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.h5.native import NativeH5Reader, NativeH5Writer

_logger = logging.getLogger(__name__)


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

        self._record_mode = False
        self.edit_mode = False
        self._replaying = False
        self._output_suppressed = False
        self._syncing = False
        self.cascade_in_progress = False
        self._session_input_pending = False
        self._suppress_session_prompt = False
        self._delete_action: QW.QAction | None = None
        self._duplicate_action: QW.QAction | None = None
        self.step_prev_action: QW.QAction | None = None
        self.step_next_action: QW.QAction | None = None
        self._restore_selection_action: QW.QAction | None = None
        self._edit_action: QW.QAction | None = None
        self._record_action: QW.QAction | None = None
        self._menu_actions: list[QW.QAction] = self.create_menu_actions()

        self.mainwindow = parent
        self.tree = HistoryTree(self)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)
        self.tree.itemDoubleClicked.connect(self.replay_restore_actions)
        self.tree.itemSelectionChanged.connect(self.sync_panel_selection)
        self.tree.itemSelectionChanged.connect(self.update_actions_state)
        self.tree.itemSelectionChanged.connect(self.update_state_widget)
        self.tree.itemSelectionChanged.connect(self.set_active_session_from_selection)

        self._state_widget = WorkspaceStateWidget(self)

        toolbar = QW.QToolBar(self)
        add_actions(toolbar, self._menu_actions)
        widget = QW.QWidget(self)
        layout = QW.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.tree)
        layout.addWidget(self._state_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)

        self.addWidget(widget)

        self.history_sessions: list[HistorySession] = []
        self._active_session_by_panel: dict[str, HistorySession] = {}
        self._session_increment = 0
        self.action_output_uuids: dict[str, list[str]] = {}
        self.output_to_action: dict[str, str] = {}
        self.cascade_warnings: list[str] = []
        self.broken_actions: set[str] = set()
        self.reconnecting = False
        self.obj_ids_snapshot: dict[str, set[str]] = {}
        for panel in (self.mainwindow.signalpanel, self.mainwindow.imagepanel):
            panel.SIG_OBJECT_ADDED.connect(self.refresh_compatibility_items)
            panel.SIG_OBJECT_ADDED.connect(self.refresh_obj_ids_snapshot)
            panel.SIG_OBJECT_REMOVED.connect(self.refresh_compatibility_items)
            panel.SIG_OBJECT_REMOVED.connect(
                functools.partial(self.reconnect_chain_after_removal, panel)
            )
            panel.SIG_OBJECT_REMOVED.connect(self.prune_output_mapping)
            panel.SIG_OBJECT_MODIFIED.connect(self.refresh_compatibility_items)
        self.refresh_obj_ids_snapshot()
        self.update_actions_state()
        self.refresh_compatibility_items()
        if not execenv.unattended and Conf.proc.history_auto_record.get(False):
            self._record_action.setChecked(True)
            self.create_new_session()

    def refresh_obj_ids_snapshot(self) -> None:
        """Cache the current object ids of both data panels."""
        self.obj_ids_snapshot = {
            self.mainwindow.signalpanel.PANEL_STR_ID: set(
                self.mainwindow.signalpanel.objmodel.get_object_ids()
            ),
            self.mainwindow.imagepanel.PANEL_STR_ID: set(
                self.mainwindow.imagepanel.objmodel.get_object_ids()
            ),
        }

    def update_actions_state(self) -> None:
        """Update the enabled state of menu actions depending on history content."""
        has_history = len(self) > 0
        for action in (self._delete_action, self._duplicate_action):
            if action is not None:
                action.setEnabled(has_history)
        if self.step_prev_action is not None:
            self.step_prev_action.setEnabled(self.can_step_prev())
        if self.step_next_action is not None:
            self.step_next_action.setEnabled(self.can_step_next())
        if self._restore_selection_action is not None:
            self._restore_selection_action.setEnabled(
                self.edit_mode or self.has_any_pending_edits()
            )

    @property
    def session_increment(self) -> int:
        """Return the current session counter."""
        return self._session_increment

    @session_increment.setter
    def session_increment(self, value: int) -> None:
        """Set the current session counter."""
        self._session_increment = value

    @property
    def record_mode_enabled(self) -> bool:
        """Return True when record mode is enabled."""
        return self._record_mode

    def has_any_pending_edits(self) -> bool:
        """Return True if any action across all sessions has a pending Edit
        mode snapshot (i.e. uncommitted edits that Restore can revert)."""
        return any(
            action.has_pending_edits
            for session in self.history_sessions
            for action in session.actions
        )

    def update_state_widget(self) -> None:
        """Update the workspace state widget from the currently selected action."""
        action = self.current_action()
        if action is not None:
            self._state_widget.update_from_state(action.state)
        else:
            self._state_widget.update_from_state(None)

    def create_menu_actions(self) -> list[QW.QAction]:
        """Create menu actions for the history panel."""
        edit_action = create_action(
            self,
            _("Edit mode"),
            toggled=self.toggle_edit_mode,
            icon=get_icon("edit_mode.svg"),
        )
        edit_action.setChecked(self.edit_mode)
        self._edit_action = edit_action
        # Temporarily disabled (superseded by the Replay / Step-by-step launch
        # modes): keep the action and its edit-mode logic, but hide it from the
        # toolbar and context menu.
        edit_action.setVisible(False)
        record_action = create_action(
            self,
            _("Record mode"),
            toggled=self.toggle_record_mode,
            icon=get_icon("record.svg"),
        )
        record_action.setChecked(self._record_mode)
        self._record_action = record_action
        new_session_action = create_action(
            self,
            _("New session"),
            lambda checked=False: self.create_new_session(),
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
        self._delete_action = create_action(
            self,
            _("Delete"),
            self.delete_selected,
            icon=get_icon("delete.svg"),
        )
        self._duplicate_action = create_action(
            self,
            _("Duplicate"),
            self.duplicate_selected_entries,
            icon=get_icon("duplicate.svg"),
            tip=_("Duplicate selected history action/session"),
        )
        self.step_prev_action = create_action(
            self,
            _("Previous step"),
            triggered=self.step_prev,
            icon=get_icon("libre-gui-arrow-left.svg"),
            tip=_("Select the previous action in the current session"),
            shortcut=QG.QKeySequence("Ctrl+Left"),
        )
        self.step_next_action = create_action(
            self,
            _("Next step"),
            triggered=self.step_next,
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
        # Temporarily disabled (out of current scope): keep the action and its
        # implementation, but hide it from the toolbar and context menu.
        generate_macro_action.setVisible(False)
        remove_incompatible_action = create_action(
            self,
            _("Remove incompatible"),
            self.remove_incompatible_actions,
            icon=get_icon("edit/delete_all.svg"),
            tip=_("Remove actions incompatible with the current workspace"),
        )
        self._restore_selection_action = create_action(
            self,
            _("Restore parameters"),
            lambda: self.replay_restore_actions(restore_selection=True, replay=False),
            icon=get_icon("restore_selection.svg"),
            tip=_("Restore original parameters (discard edit-mode changes)"),
        )
        # Temporarily disabled (out of current scope): keep the action and its
        # restore logic, but hide it from the toolbar and context menu.
        self._restore_selection_action.setVisible(False)
        replay_action = create_action(
            self,
            _("Replay"),
            lambda: self.replay_restore_actions(restore_selection=False),
            icon=get_icon("replay.svg"),
            tip=_("Replay the selection silently (no parameter dialogs)"),
        )
        step_by_step_action = create_action(
            self,
            _("Step-by-step"),
            triggered=lambda checked=False: self.replay_step_by_step(),
            icon=get_icon("edit_mode.svg"),
            tip=_("Replay the selection step by step, editing parameters at each step"),
        )
        return [
            record_action,
            new_session_action,
            None,
            open_action,
            save_action,
            None,
            self.step_prev_action,
            self.step_next_action,
            None,
            replay_action,
            step_by_step_action,
            self._restore_selection_action,
            edit_action,
            None,
            self._duplicate_action,
            generate_macro_action,
            None,
            remove_incompatible_action,
            self._delete_action,
        ]

    def toggle_edit_mode(self, checked: bool) -> None:
        """Toggle edit mode.

        Toggling Edit mode off is a definitive commit: all parameter
        changes performed during the session become permanent.
        """
        if not checked and self.has_any_pending_edits():
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
                if self._edit_action is not None:
                    self._edit_action.blockSignals(True)
                    self._edit_action.setChecked(True)
                    self._edit_action.blockSignals(False)
                return
        self.edit_mode = checked
        if not checked:
            for session in self.history_sessions:
                for action in session.actions:
                    action.discard_snapshot()
        self.update_actions_state()

    def toggle_record_mode(self, checked: bool) -> None:
        """Toggle record mode."""
        self._record_mode = checked

    def is_edit_mode(self) -> bool:
        """Return True when the History panel is in edit mode."""
        return self.edit_mode

    @contextmanager
    def replaying(self) -> Generator[None, None, None]:
        """Context manager suppressing history capture during its scope."""
        previous = self._replaying
        self._replaying = True
        try:
            yield
        finally:
            self._replaying = previous

    def is_replaying(self) -> bool:
        """Return True when an external replay/recompute is in progress."""
        return self._replaying

    @contextmanager
    def output_suppressed(self) -> Generator[None, None, None]:
        """Context manager suppressing compute outputs during its scope."""
        previous = self._output_suppressed
        self._output_suppressed = True
        try:
            yield
        finally:
            self._output_suppressed = previous

    def is_output_suppressed(self) -> bool:
        """Return True when compute outputs must not be added to panels."""
        return self._output_suppressed

    def show_context_menu(self, pos: QC.QPoint) -> None:
        """Show the context menu."""
        self.refresh_compatibility_items()
        menu = QW.QMenu()
        add_actions(menu, self._menu_actions)
        menu.exec_(self.tree.mapToGlobal(pos))

    def get_action_from_uuid(self, uuid: str) -> HistoryAction:
        """Get the action from its UUID."""
        for session in self.history_sessions:
            for action in session.actions:
                if action.uuid == uuid:
                    return action
        raise ValueError("Action not found")

    # ------------------------------------------------------------------
    # Interactive replay delegations
    # ------------------------------------------------------------------

    def replay_restore_actions(
        self, replay: bool = True, restore_selection: bool = True
    ) -> None:
        """Replay and/or restore selection for the selected actions."""
        return hreplay.replay_restore_actions(self, replay, restore_selection)

    def replay_step_by_step(self) -> None:
        """Replay the current selection step by step, prompting for parameters.

        Dialog-driven launch mode: each replayed action opens its parameter
        dialog (reusing the edit-mode machinery), then recomputes. Edits are
        committed immediately -- there is no persistent edit session.
        """
        previous = self.edit_mode
        self.edit_mode = True
        try:
            self.replay_restore_actions(replay=True, restore_selection=False)
        finally:
            self.edit_mode = previous
            # Commit step-by-step edits immediately (no persistent edit session).
            for session in self.history_sessions:
                for action in session.actions:
                    action.discard_snapshot()
            self.update_actions_state()

    def prompt_edit_action_params(self, action: HistoryAction) -> bool | None:
        """Open the parameter dialog for *action* according to its pattern."""
        return hreplay.prompt_edit_action_params(self, action)

    def restore_action_params(self, item: HistoryAction | HistorySession) -> None:
        """Restore original kwargs from snapshot and recompute in-place."""
        return hreplay.restore_action_params(self, item)

    # ------------------------------------------------------------------
    # Chain delegations
    # ------------------------------------------------------------------

    def find_parent_session(self, action: HistoryAction) -> HistorySession | None:
        """Return the session that contains ``action``, or None."""
        return hchain.find_parent_session(self, action)

    def resolve_panel_for_action(self, action: HistoryAction) -> BaseDataPanel | None:
        """Return the data panel targeted by ``action``, or ``None``."""
        return hchain.resolve_panel_for_action(self, action)

    def find_output_object_uuid(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> str | None:
        """Find the UUID of the output object produced by ``action``."""
        return hchain.find_output_object_uuid(self, panel, action)

    def find_action_for_output(
        self, output_uuid: str, func_name: str
    ) -> HistoryAction | None:
        """Find the action that produced ``output_uuid``."""
        return hchain.find_action_for_output(self, output_uuid, func_name)

    def find_creation_action_for_output(self, output_uuid: str) -> HistoryAction | None:
        """Find the creation action that produced ``output_uuid``."""
        return hchain.find_creation_action_for_output(self, output_uuid)

    def find_analysis_action(
        self, obj_uuid: str, func_name: str
    ) -> HistoryAction | None:
        """Find the 1-to-0 analysis action for ``obj_uuid`` with ``func_name``."""
        return hchain.find_analysis_action(self, obj_uuid, func_name)

    def get_session_of(self, action: HistoryAction) -> HistorySession | None:
        """Return the session that contains ``action``, or None."""
        return hchain.get_session_of(self, action)

    def action_output_uuid(self, action: HistoryAction) -> str | None:
        """Return the UUID of the object produced by ``action``, or ``None``."""
        return hchain.action_output_uuid(self, action)

    def get_downstream_actions(self, action: HistoryAction) -> list[HistoryAction]:
        """Return the actions of the current session that depend on ``action``."""
        return hchain.get_downstream_actions(self, action)

    def resolve_target_outputs(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> tuple[list[str], list[str]]:
        """Return ``(existing, missing)`` UUIDs registered for ``action``."""
        return hchain.resolve_target_outputs(self, panel, action)

    def existing_input_uuids(
        self, panel: BaseDataPanel, action: HistoryAction
    ) -> list[str]:
        """Return recorded input UUIDs that still exist in ``panel``."""
        return hchain.existing_input_uuids(panel, action)

    def prune_output_mapping(self) -> None:
        """Drop entries of :attr:`output_to_action` whose object no longer exists."""
        return hchain.prune_output_mapping(self)

    def remove_single_action(self, action: HistoryAction) -> None:
        """Remove a single action from its session (splice, not truncate)."""
        return hchain.remove_single_action(self, action)

    def reconnect_chain_after_removal(self, panel: BaseDataPanel) -> None:
        """Reconnect the processing chain after object(s) were deleted."""
        return hchain.reconnect_chain_after_removal(self, panel)

    # ------------------------------------------------------------------
    # Recompute delegations
    # ------------------------------------------------------------------

    def refresh_action(self, action: HistoryAction) -> None:
        """Refresh the tree display for ``action`` after its kwargs were mutated."""
        return hrec.refresh_action(self, action)

    def recompute_cascade(
        self,
        root_action: HistoryAction,
        descendants: list[HistoryAction] | None = None,
    ) -> None:
        """Recompute ``root_action``'s descendants in the current session."""
        return hrec.recompute_cascade(self, root_action, descendants)

    def flush_cascade_warnings(self) -> None:
        """Show + clear accumulated cascade warnings (no-op when empty)."""
        return hrec.flush_cascade_warnings(self)

    # ------------------------------------------------------------------
    # Sync History tree selection → Signal/Image panel
    # ------------------------------------------------------------------

    def sync_panel_selection(self) -> None:
        """Sync data panel selection from the currently selected tree item."""
        if self._replaying or self._syncing:
            return
        item = self.tree.currentItem()
        if item is None or not item.isSelected():
            return
        if item.parent() is None:
            index = self.tree.indexOfTopLevelItem(item)
            if index < 0 or index >= len(self.history_sessions):
                return
            session = self.history_sessions[index]
            action = next(
                (a for a in session.actions if a.kind == HistoryAction.KIND_COMPUTE),
                None,
            )
        else:
            uuid = item.data(0, QC.Qt.UserRole)
            try:
                action = self.tree.get_action_from_uuid(uuid, self.history_sessions)
            except ValueError:
                action = None
        if action is None:
            return

        panel = self.resolve_panel_for_action(action)
        if panel is None:
            return

        target_uuids: list[str] = []
        output_uuid = self.find_output_object_uuid(panel, action)
        if output_uuid is not None:
            target_uuids = [output_uuid]
        else:
            target_uuids = self.existing_input_uuids(panel, action)

        if not target_uuids:
            return

        self._syncing = True
        try:
            with QC.QSignalBlocker(panel.objview):
                panel.objview.select_objects(target_uuids)
            self.mainwindow.set_current_panel(panel)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Step-by-step navigation
    # ------------------------------------------------------------------

    def current_action(self) -> HistoryAction | None:
        """Return the action currently selected in the tree, or ``None``."""
        item = self.tree.currentItem()
        if item is None or item.parent() is None:
            return None
        uuid = item.data(0, QC.Qt.UserRole)
        try:
            return self.tree.get_action_from_uuid(uuid, self.history_sessions)
        except ValueError:
            return None

    def _current_panel_str(self) -> str:
        """Return the current data panel id ('signal'/'image'); default 'signal'."""
        pstr = self.mainwindow.get_current_panel()
        return pstr if pstr in ("signal", "image") else "signal"

    def on_current_panel_changed(self, panel_str: str) -> None:
        """React to a Signal/Image panel switch: bring this panel's active
        recording session into view and refresh the active-session highlight.
        """
        if panel_str not in ("signal", "image"):
            return
        self.refresh_active_session_highlight()
        session = self.get_active_session(panel_str)
        if session is not None and session in self.history_sessions:
            index = self.history_sessions.index(session)
            item = self.tree.topLevelItem(index)
            if item is not None:
                self.tree.scrollToItem(item)

    def session_panel_str(self, session: HistorySession) -> str | None:
        """Return the panel a session belongs to.

        Derived from the panel_str of its first tagged action; falls back to
        the active-session tracking for freshly created (empty) sessions.
        """
        for action in session.actions:
            if action.panel_str:
                return action.panel_str
        for pstr, sess in self._active_session_by_panel.items():
            if sess is session:
                return pstr
        return None

    def get_active_session(self, panel_str: str) -> HistorySession | None:
        """Return the active recording session for ``panel_str``, if still valid."""
        session = self._active_session_by_panel.get(panel_str)
        if session is not None and session in self.history_sessions:
            return session
        return None

    def set_active_session(
        self, session: HistorySession, panel_str: str | None = None
    ) -> None:
        """Mark ``session`` as the active recording session for its panel."""
        pstr = panel_str or self.session_panel_str(session)
        if pstr:
            self._active_session_by_panel[pstr] = session
            self.refresh_active_session_highlight()

    def refresh_active_session_highlight(self) -> None:
        """Update the tree highlight to mark each panel's active session."""
        active: dict[int, str] = {}
        for pstr, session in self._active_session_by_panel.items():
            if session in self.history_sessions:
                active[session.number] = pstr
        self.tree.set_active_sessions(active)

    def set_active_session_from_selection(self) -> None:
        """When recording, make the selected session the active one for its panel.

        Lets the user resume recording into any session by selecting it (or one
        of its actions) in the tree.
        """
        if not self.record_mode_enabled:
            return
        item = self.tree.currentItem()
        if item is None or not item.isSelected():
            return
        if item.parent() is None:
            index = self.tree.indexOfTopLevelItem(item)
            if not 0 <= index < len(self.history_sessions):
                return
            session = self.history_sessions[index]
        else:
            action = self.current_action()
            if action is None:
                return
            session = self.find_parent_session(action)
        if session is not None:
            self.set_active_session(session)

    def current_session(self) -> HistorySession | None:
        """Return the session relevant for step navigation."""
        item = self.tree.currentItem()
        if item is not None:
            top = item
            while top.parent() is not None:
                top = top.parent()
            index = self.tree.indexOfTopLevelItem(top)
            if 0 <= index < len(self.history_sessions):
                return self.history_sessions[index]
        if self.history_sessions:
            return self.history_sessions[-1]
        return None

    def can_step_prev(self) -> bool:
        """Return True if a previous action exists in the current session."""
        session = self.current_session()
        if session is None or not session.actions:
            return False
        action = self.current_action()
        if action is None or action not in session.actions:
            return False
        return session.actions.index(action) > 0

    def can_step_next(self) -> bool:
        """Return True if a next action exists in the current session."""
        session = self.current_session()
        if session is None or not session.actions:
            return False
        action = self.current_action()
        if action is None or action not in session.actions:
            return True
        return session.actions.index(action) < len(session.actions) - 1

    def select_action_in_tree(self, action: HistoryAction) -> None:
        """Select ``action`` in the tree (triggers ``sync_panel_selection``)."""
        iterator = QW.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, QC.Qt.UserRole) == action.uuid:
                self.tree.clearSelection()
                self.tree.setCurrentItem(item)
                item.setSelected(True)
                return
            iterator += 1

    def step_prev(self) -> None:
        """Select the previous action in the current session."""
        if not self.can_step_prev():
            return
        session = self.current_session()
        action = self.current_action()
        idx = session.actions.index(action)
        self.select_action_in_tree(session.actions[idx - 1])
        self.update_actions_state()

    def step_next(self) -> None:
        """Select the next action in the current session."""
        if not self.can_step_next():
            return
        session = self.current_session()
        action = self.current_action()
        if action is None or action not in session.actions:
            target = session.actions[0]
        else:
            target = session.actions[session.actions.index(action) + 1]
        self.select_action_in_tree(target)
        self.update_actions_state()

    # ------------------------------------------------------------------
    # History tools delegations
    # ------------------------------------------------------------------

    def duplicate_selected_entries(self) -> None:
        """Duplicate selected entries."""
        return htools.duplicate_selected_entries(self)

    def generate_macro(self) -> None:
        """Generate a Python macro script from history."""
        return htools.generate_macro(self)

    def select_sessions(self, sessions: list[HistorySession]) -> None:
        """Select top-level tree items matching ``sessions``."""
        self.tree.clearSelection()
        for session in sessions:
            index = self.history_sessions.index(session)
            item = self.tree.topLevelItem(index)
            item.setSelected(True)
            self.tree.setCurrentItem(item)

    def delete_selected(self) -> None:
        """Delete the currently selected entries."""
        return htools.delete_selected(self)

    def remove_incompatible_actions(self) -> None:
        """Remove actions incompatible with the current workspace."""
        return htools.remove_incompatible_actions(self)

    # ------------------------------------------------------------------
    # HDF5 / .dlhist I/O delegations
    # ------------------------------------------------------------------

    def save_to_dlhist_file(self, filename: str | None = None) -> bool:
        """Save history to a standalone .dlhist file."""
        return hio.save_to_dlhist_file(self, filename)

    def open_dlhist_file(self, filename: str | None = None) -> bool:
        """Open history from a standalone .dlhist file."""
        return hio.open_dlhist_file(self, filename)

    def import_dlhist_into_new_session(self, reader: NativeH5Reader) -> None:
        """Import a .dlhist into a new session."""
        return hio.import_dlhist_into_new_session(self, reader)

    def refresh_compatibility_items(self, *args: Any) -> None:
        """Refresh compatibility icons in the history tree."""
        return hio.refresh_compatibility_items(self, *args)

    def serialize_to_hdf5(self, writer: NativeH5Writer) -> None:
        """Serialize the history to HDF5."""
        return hio.serialize_to_hdf5(self, writer)

    def deserialize_from_hdf5(
        self, reader: NativeH5Reader, reset_all: bool = False
    ) -> None:
        """Deserialize the history from HDF5."""
        return hio.deserialize_from_hdf5(self, reader, reset_all)

    def __len__(self) -> int:
        """Return number of objects."""
        return sum(len(session.actions) for session in self.history_sessions)

    def __getitem__(self, nb: int) -> HistoryAction:
        """Return object from its number (1 to N)."""
        for session in self.history_sessions:
            if nb <= len(session.actions):
                return session.actions[nb - 1]
            nb -= len(session.actions)
        raise IndexError("Index out of range")

    def __iter__(self) -> Generator[HistoryAction, None, None]:
        """Iterate over objects."""
        for session in self.history_sessions:
            yield from session.actions

    # ------------------------------------------------------------------
    # Session operations delegations
    # ------------------------------------------------------------------

    def create_new_session(self, panel_str: str | None = None) -> HistorySession:
        """Create a new history session (active for the given/current panel)."""
        return hsess.create_new_session(self, panel_str=panel_str)

    def start_new_session_after_workspace_reset(self) -> None:
        """Start a new history session after a workspace reset."""
        return hsess.start_new_session_after_workspace_reset(self)

    def maybe_start_session_for_input(self, *, load: bool = False) -> None:
        """Offer to start a new history session before a creation/load is recorded.

        Args:
            load: True when triggered by a file/workspace load, False for an
             object creation. Only affects the prompt wording.
        """
        return hsess.maybe_start_session_for_input(self, load=load)

    @contextmanager
    def session_prompt_suppressed(self) -> Generator[None, None, None]:
        """Context manager suppressing the new-session prompt during a batch load."""
        previous = self._suppress_session_prompt
        self._suppress_session_prompt = True
        try:
            yield
        finally:
            self._suppress_session_prompt = previous

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
        """Add a compute entry to the history."""
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
        """Add a compute entry built from a :class:`ProcessingParameters`."""
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
        """Register the output UUIDs produced by ``action``."""
        return hsess.register_action_outputs(self, action, output_uuids)

    def capture_outputs(
        self, action: HistoryAction | None
    ) -> Generator[None, None, None]:
        """Context manager capturing outputs produced by ``action``."""
        return hsess.capture_outputs(self, action)

    def add_ui_entry(
        self,
        action_title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
        **kwargs: Any,
    ) -> HistoryAction | None:
        """Add a UI entry to the history."""
        return hsess.add_ui_entry(
            self, action_title, target, method_name, save_state, **kwargs
        )

    # ------ AbstractPanel interface ---------------------------------------------------
    def create_object(self) -> HistoryAction:
        """Create and return object."""
        return HistoryAction()

    def add_object(self, obj: HistoryAction) -> None:
        """Add an object to the history."""
        return hsess.add_object(self, obj)

    def remove_all_objects(self) -> None:
        """Remove all objects."""
        super().remove_all_objects()
        self.action_output_uuids.clear()
        self.output_to_action.clear()
