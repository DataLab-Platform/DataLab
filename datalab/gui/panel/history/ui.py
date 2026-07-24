# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Qt action and widget setup for the History panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.qthelpers import add_actions, create_action
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.gui import historytools_ops as htools
from datalab.widgets.workspacestate_widget import WorkspaceStateWidget

if TYPE_CHECKING:
    from qtpy import QtCore as QC

    from datalab.gui.panel.history.panel import HistoryPanel


class HistoryPanelUI:
    """Build and own History panel widgets and actions."""

    def __init__(self, panel: HistoryPanel) -> None:
        self.panel = panel
        self.state_widget = WorkspaceStateWidget(panel)
        self.actions = self.create_actions()
        self.menu_actions = self.create_menu_actions()
        self.setup_connections()
        self.setup_layout()

    def create_actions(self) -> dict[str, QW.QAction]:
        """Create named actions used by history menus and toolbar."""
        panel = self.panel
        actions = {
            "record": create_action(
                panel,
                _("Record mode"),
                toggled=panel.toggle_record_mode,
                icon=get_icon("record.svg"),
            ),
            "new_session": create_action(
                panel,
                _("New session"),
                lambda checked=False: panel.create_new_session(),
                icon=get_icon("libre-gui-add.svg"),
                tip=_("Start a new history session"),
            ),
            "open": create_action(
                panel,
                _("Open history file..."),
                triggered=lambda checked=False: panel.open_dlhist_file(),
                icon=get_icon("fileopen_h5.svg"),
                tip=_("Open history from a standalone .dlhist file"),
            ),
            "save": create_action(
                panel,
                _("Save history file..."),
                triggered=lambda checked=False: panel.save_to_dlhist_file(),
                icon=get_icon("filesave_h5.svg"),
                tip=_("Save history to a standalone .dlhist file"),
            ),
            "delete": create_action(
                panel,
                _("Delete"),
                lambda: htools.delete_selected(panel),
                icon=get_icon("delete.svg"),
            ),
            "duplicate": create_action(
                panel,
                _("Duplicate"),
                lambda: htools.duplicate_selected_entries(panel),
                icon=get_icon("duplicate.svg"),
                tip=_("Duplicate selected history action/session"),
            ),
            "step_prev": create_action(
                panel,
                _("Previous step"),
                triggered=panel.navigation.step_prev,
                icon=get_icon("libre-gui-arrow-left.svg"),
                tip=_("Select the previous action in the current session"),
                shortcut=QG.QKeySequence("Ctrl+Left"),
            ),
            "step_next": create_action(
                panel,
                _("Next step"),
                triggered=panel.navigation.step_next,
                icon=get_icon("libre-gui-arrow-right.svg"),
                tip=_("Select the next action in the current session"),
                shortcut=QG.QKeySequence("Ctrl+Right"),
            ),
            "remove_incompatible": create_action(
                panel,
                _("Remove incompatible"),
                lambda: htools.remove_incompatible_actions(panel),
                icon=get_icon("edit/delete_all.svg"),
                tip=_("Remove actions incompatible with the current workspace"),
            ),
            "replay": create_action(
                panel,
                _("Replay"),
                lambda: panel.replay_restore_actions(restore_selection=False),
                icon=get_icon("replay.svg"),
                tip=_("Replay the selection silently (no parameter dialogs)"),
            ),
            "step_by_step": create_action(
                panel,
                _("Step-by-step"),
                triggered=lambda checked=False: panel.replay_step_by_step(),
                icon=get_icon("edit_mode.svg"),
                tip=_(
                    "Replay the selection step by step, editing parameters at each step"
                ),
            ),
        }
        actions["record"].setChecked(panel.runtime.execution.record_mode)
        return actions

    def create_menu_actions(self) -> list[QW.QAction | None]:
        """Return ordered actions and separators for menus and toolbar."""
        action = self.actions
        return [
            action["record"],
            action["new_session"],
            None,
            action["open"],
            action["save"],
            None,
            action["step_prev"],
            action["step_next"],
            None,
            action["replay"],
            action["step_by_step"],
            None,
            action["duplicate"],
            None,
            action["remove_incompatible"],
            action["delete"],
        ]

    def setup_connections(self) -> None:
        """Connect history-tree interactions to their owning components."""
        tree = self.panel.tree
        tree.customContextMenuRequested.connect(self.show_context_menu)
        tree.itemDoubleClicked.connect(self.panel.replay_restore_actions)
        tree.itemSelectionChanged.connect(self.panel.navigation.sync_panel_selection)
        tree.itemSelectionChanged.connect(self.update_actions_state)
        tree.itemSelectionChanged.connect(self.panel.navigation.update_state_widget)
        tree.itemSelectionChanged.connect(
            self.panel.navigation.set_active_session_from_selection
        )

    def setup_layout(self) -> None:
        """Install the toolbar, history tree, and workspace-state widget."""
        toolbar = QW.QToolBar(self.panel)
        add_actions(toolbar, self.menu_actions)
        widget = QW.QWidget(self.panel)
        layout = QW.QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.panel.tree)
        layout.addWidget(self.state_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(layout)
        self.panel.addWidget(widget)

    def update_actions_state(self) -> None:
        """Update action availability from history and step state."""
        has_history = len(self.panel) > 0
        self.actions["delete"].setEnabled(has_history)
        self.actions["duplicate"].setEnabled(has_history)
        self.actions["step_prev"].setEnabled(self.panel.navigation.can_step_prev())
        self.actions["step_next"].setEnabled(self.panel.navigation.can_step_next())

    def show_context_menu(self, pos: QC.QPoint) -> None:
        """Show the history context menu at a tree position."""
        self.panel.refresh_compatibility_items()
        menu = QW.QMenu()
        add_actions(menu, self.menu_actions)
        menu.exec_(self.panel.tree.mapToGlobal(pos))
