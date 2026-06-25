# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive (dialog-driven) replay helpers for the History panel."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import guidata.dataset as gds
from guidata.dataset.qtwidgets import DataSetEditDialog, DataSetGroupEditDialog
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import recompute as hrec
from datalab.history import HistoryAction, HistorySession

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel

_logger = logging.getLogger(__name__)


def replay_restore_actions(
    panel: HistoryPanel, replay: bool = True, restore_selection: bool = True
) -> None:
    """Replay and/or restore selection for the selected actions."""
    panel.refresh_compatibility_items()
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        if not panel.history_sessions:
            return
        selected = [panel.history_sessions[-1]]
    for session_or_action in selected:
        if isinstance(session_or_action, HistoryAction) and session_or_action.is_stale:
            hrec.recompute_cascade(panel, session_or_action)
            continue
        if not session_or_action.is_current_state_compatible(
            panel.mainwindow, restore_selection=restore_selection
        ):
            if not execenv.unattended:
                QW.QMessageBox.critical(
                    panel.mainwindow,
                    _("Error"),
                    _("The current workspace state is not compatible with the action."),
                )
            return
        if replay:
            if panel.edit_mode and isinstance(session_or_action, HistoryAction):
                edit_mode_replay(panel, session_or_action)
            elif panel.edit_mode and isinstance(session_or_action, HistorySession):
                view_only_session_replay(panel, session_or_action, restore_selection)
            else:
                with panel.replaying(), panel.output_suppressed():
                    session_or_action.replay(
                        panel.mainwindow,
                        restore_selection=restore_selection,
                        edit=panel.edit_mode,
                    )
        elif restore_selection:
            if panel.edit_mode or panel.has_any_pending_edits():
                restore_action_params(panel, session_or_action)
            else:
                session_or_action.restore(panel.mainwindow)


def prompt_edit_action_params(
    panel: HistoryPanel, action: HistoryAction
) -> bool | None:
    """Open the parameter dialog for *action* according to its pattern."""
    pattern = action.pattern
    if pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
        param = action.kwargs.get("param")
        if param is None:
            return None
        edited = copy.deepcopy(param)
        dialog_target: gds.DataSet | gds.DataSetGroup = edited
        new_kwargs = {"param": edited}
    elif pattern == "1_to_n":
        params = action.kwargs.get("params") or []
        if not params:
            return None
        edited_params = [copy.deepcopy(p) for p in params]
        dialog_target = gds.DataSetGroup(edited_params, title=_("Parameters"))
        new_kwargs = {"params": edited_params}
    else:
        return None
    if not dialog_target.edit(parent=panel.mainwindow):
        return False
    action.snapshot_kwargs()
    action.kwargs.update(new_kwargs)
    return True


def edit_mode_replay(panel: HistoryPanel, action: HistoryAction) -> None:
    """Replay a single action in edit mode: open param dialog, update kwargs."""
    if action.kind != HistoryAction.KIND_COMPUTE or action.pattern is None:
        with panel.replaying(), panel.output_suppressed():
            action.replay(panel.mainwindow, restore_selection=True, edit=True)
        return

    chain: list[HistoryAction] = [action] + panel.get_downstream_actions(action)
    edited_actions: list[HistoryAction] = []
    for a in chain:
        result = prompt_edit_action_params(panel, a)
        if result is False:
            for done in edited_actions:
                done.restore_kwargs()
                panel.tree.refresh_action_item(done)
            return
        if result is True:
            edited_actions.append(a)

    for a in edited_actions:
        panel.tree.refresh_action_item(a)

    downstream = chain[1:]
    hrec.recompute_action_in_place(panel, action)
    hrec.recompute_cascade(panel, action, descendants=downstream)

    for a in chain:
        panel.tree.refresh_action_item(a)
    QW.QApplication.processEvents()


def show_readonly_param_dialog(
    panel: HistoryPanel, dataset: gds.DataSet | gds.DataSetGroup
) -> None:
    """Show a parameter dialog identical to the edit dialog but read-only."""
    if isinstance(dataset, gds.DataSetGroup):
        dialog = DataSetGroupEditDialog(dataset, parent=panel.mainwindow)
    else:
        dialog = DataSetEditDialog(dataset, parent=panel.mainwindow)
    for edl in dialog.edit_layout:
        for widget in edl.widgets:
            if widget.group is not None:
                widget.group.setEnabled(False)
            if widget.label is not None:
                widget.label.setEnabled(False)
    dialog.exec()


def view_only_session_replay(
    panel: HistoryPanel,
    session: HistorySession,
    restore_selection: bool,
) -> None:
    """Replay a session in edit mode with read-only parameter dialogs."""
    for action in session.actions:
        if action.kind != HistoryAction.KIND_COMPUTE:
            continue
        pattern = action.pattern
        panel.select_action_in_tree(action)
        QW.QApplication.processEvents()
        if pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
            param = action.kwargs.get("param")
            if param is not None:
                show_readonly_param_dialog(panel, copy.deepcopy(param))
        elif pattern == "1_to_n":
            params = action.kwargs.get("params") or []
            if params:
                group = gds.DataSetGroup(
                    [copy.deepcopy(p) for p in params],
                    title=_("Parameters"),
                )
                show_readonly_param_dialog(panel, group)

    with panel.replaying(), panel.output_suppressed():
        session.replay(
            panel.mainwindow,
            restore_selection=restore_selection,
            edit=False,
        )


def restore_action_params(
    panel: HistoryPanel, item: HistoryAction | HistorySession
) -> None:
    """Restore original kwargs from snapshot and recompute in-place."""
    actions: list[HistoryAction]
    if isinstance(item, HistorySession):
        actions = [a for a in item.actions if a.kind == HistoryAction.KIND_COMPUTE]
    else:
        actions = [item]
    for action in actions:
        if not action.has_pending_edits:
            continue
        action.restore_kwargs()
        panel.tree.refresh_action_item(action)
        hrec.recompute_action_in_place(panel, action)
        hrec.recompute_cascade(panel, action)
    panel.update_actions_state()
