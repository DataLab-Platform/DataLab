# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive (dialog-driven) replay helpers for the History panel."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import recompute as hrec
from datalab.history import HistoryAction, HistorySession
from datalab.history.core import copy_history_value

if TYPE_CHECKING:
    from datalab.gui.panel.history.panel import HistoryPanel


@dataclass
class ActionParamEdit:
    """Parameter dialog target and action kwargs to update after acceptance."""

    dialog_target: gds.DataSet | gds.DataSetGroup
    new_kwargs: dict[str, Any]


def replay_restore_actions(
    panel: HistoryPanel, replay: bool = True, restore_selection: bool = True
) -> None:
    """Replay and/or restore selection for the selected actions.

    Entry point of the Replay, Step-by-step and double-click commands. When
    nothing is selected in the tree, the last session is targeted (no-op if
    the history is empty). Each selected session or action is first checked
    against the current workspace state: any incompatibility vetoes the whole
    command with an error dialog (skipped in unattended mode).

    When ``replay`` is enabled, the selected actions (sessions contribute all
    of their actions) are forwarded to :func:`replay_actions`, with parameter
    dialogs when the panel's edit mode is active. When ``replay`` is disabled
    and ``restore_selection`` is enabled, each selected entry is restored
    instead: if edit mode is active or any action has pending parameter
    edits, :func:`restore_action_params` rolls back the edited parameters and
    recomputes in place; otherwise the recorded workspace selection is simply
    restored.

    Args:
        panel: History panel instance
        replay: Replay the selected actions through the in-place recompute
         engine
        restore_selection: When not replaying, restore the recorded workspace
         selection (or the original parameters when edits are pending)
    """
    panel.refresh_compatibility_items()
    selected = panel.tree.get_selected_actions_or_sessions(panel.history_sessions)
    if not selected:
        if not panel.history_sessions:
            return
        selected = [panel.history_sessions[-1]]
    edit_mode = panel.runtime.execution.edit_mode
    actions_to_replay: list[HistoryAction] = []
    for session_or_action in selected:
        if not session_or_action.is_current_state_compatible(panel.mainwindow):
            if not execenv.unattended:
                QW.QMessageBox.critical(
                    panel.mainwindow,
                    _("Error"),
                    _("The current workspace state is not compatible with the action."),
                )
            return
        if replay:
            if isinstance(session_or_action, HistorySession):
                actions_to_replay.extend(session_or_action.actions)
            else:
                actions_to_replay.append(session_or_action)
        elif restore_selection:
            if edit_mode or any(
                action.has_pending_edits
                for session in panel.history_sessions
                for action in session.actions
            ):
                restore_action_params(panel, session_or_action)
            else:
                session_or_action.restore(panel.mainwindow)
    if actions_to_replay:
        replay_actions(panel, actions_to_replay, prompt=edit_mode)


def prepare_action_param_edit(action: HistoryAction) -> ActionParamEdit | None:
    """Prepare the editable parameter copy for ``action``."""
    result = None
    if (
        action.kind == HistoryAction.KIND_UI
        and action.method_name in HistoryAction.UI_CREATION_METHODS
    ):
        param = action.kwargs.get("param")
        if param is not None:
            edited = copy.deepcopy(param)
            result = ActionParamEdit(edited, {"param": edited})
    elif action.pattern in {"1_to_1", "1_to_0", "n_to_1", "2_to_1"}:
        param = action.kwargs.get("param")
        if param is not None:
            edited = copy.deepcopy(param)
            result = ActionParamEdit(edited, {"param": edited})
    elif action.pattern == "1_to_n":
        params = action.kwargs.get("params") or []
        if params:
            edited_params = [copy.deepcopy(p) for p in params]
            dialog_target = gds.DataSetGroup(edited_params, title=_("Parameters"))
            result = ActionParamEdit(dialog_target, {"params": edited_params})
    return result


def prompt_edit_action_params(
    panel: HistoryPanel, action: HistoryAction
) -> bool | None:
    """Open the parameter dialog for *action* according to its pattern."""
    edit = prepare_action_param_edit(action)
    if edit is None:
        return None
    if not edit.dialog_target.edit(parent=panel.mainwindow):
        return False
    action.snapshot_kwargs()
    action.kwargs.update(edit.new_kwargs)
    return True


def _load_outputs_still_exist(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Return True if all recorded load outputs still exist in a data panel.

    Args:
        panel: History panel instance
        action: Load action (``UI_LOAD_METHODS``) to check

    Returns:
        True if the action recorded at least one output UUID and every one of
        them still exists in either the signal or the image panel.
    """
    output_uuids = hchain.recorded_action_output_uuids(panel, action)
    if not output_uuids:
        return False
    panels = (panel.mainwindow.signalpanel, panel.mainwindow.imagepanel)
    return all(any(p.objmodel.has_uuid(uid) for p in panels) for uid in output_uuids)


def confirm_file_output_replay(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Ask the user to confirm the replay of a file-save action.

    Replaying a file-output action (``FILE_OUTPUT_METHODS``) overwrites the
    files recorded in the action kwargs, so one confirmation question is asked
    per action. In unattended mode no dialog is shown: the action is skipped
    by default and replayed only when ``execenv.accept_dialogs`` is set.

    Args:
        panel: History panel instance
        action: File-output action (``FILE_OUTPUT_METHODS``) to confirm

    Returns:
        True if the action should be replayed.
    """
    if execenv.unattended:
        return bool(execenv.accept_dialogs)
    names: list[str] = []
    filename = action.kwargs.get("filename")
    if isinstance(filename, str):
        names.append(filename)
    filenames = action.kwargs.get("filenames")
    if isinstance(filenames, (list, tuple)):
        names.extend(str(fname) for fname in filenames)
    if not names:
        # ``save_to_directory``: the destination is carried by the recorded
        # parameter object (inspected defensively, format may evolve).
        param = action.kwargs.get("param")
        directory = getattr(param, "directory", None)
        if directory:
            pattern = getattr(param, "basename", None)
            extension = getattr(param, "extension", None)
            if pattern:
                names.append(f"{directory} ({pattern}{extension or ''})")
            else:
                names.append(str(directory))
    if not names:
        names.append(action.title or action.uuid)
    answer = QW.QMessageBox.question(
        panel.mainwindow,
        _("Replay file save"),
        _("This action will overwrite the following file(s):\n%s\n\nReplay it?")
        % "\n".join(names),
        QW.QMessageBox.Yes | QW.QMessageBox.No,
        QW.QMessageBox.No,
    )
    return answer == QW.QMessageBox.Yes


def _recompute_stale_actions(panel: HistoryPanel, ordered: list[HistoryAction]) -> None:
    """Recompute stale actions in place after a dialog rollback.

    Args:
        panel: History panel instance
        ordered: Selected actions in session order
    """
    stale_actions = [a for a in ordered if a.is_stale]
    if not stale_actions:
        return
    try:
        for stale_action in stale_actions:
            success = hrec.recompute_action_in_place(panel, stale_action)
            stale_action.is_stale = not success
            panel.tree.refresh_action_item(stale_action)
    finally:
        hrec.flush_cascade_warnings(panel)


def replay_actions(
    panel: HistoryPanel, actions: list[HistoryAction], prompt: bool = True
) -> None:
    """Replay selected actions through the in-place recompute engine.

    When ``prompt`` is enabled, each selected action gets exactly one
    parameter dialog. Recomputable selected actions are always included,
    while accepted parameter edits also include all downstream dependent
    actions. When ``prompt`` is disabled, actions are recomputed silently
    with their current parameters (no dialogs anywhere). The resulting
    global plan is deduplicated and executed in session order. A re-entrance
    guard prevents nested prompt loops.

    Args:
        panel: History panel instance
        actions: Selected actions to replay
        prompt: Open parameter dialogs before recomputing
    """
    # Deduplicate and sort the selected actions in their session order
    ordered = order_selected_actions(panel, actions)
    if not ordered:
        return
    with panel.runtime.execution.replaying_edits() as started:
        if not started:
            return
        entry_states = (
            {
                action.uuid: (
                    copy_history_value(action.kwargs),
                    copy_history_value(action.saved_kwargs),
                )
                for action in ordered
            }
            if prompt
            else {}
        )
        edited_actions: list[HistoryAction] = []
        recomputable: list[HistoryAction] = []
        deferred_actions: list[HistoryAction] = []
        for action in ordered:
            is_creation = (
                action.kind == HistoryAction.KIND_UI
                and action.method_name in HistoryAction.UI_CREATION_METHODS
            )
            is_compute = (
                action.kind == HistoryAction.KIND_COMPUTE and action.pattern is not None
            )
            if action.kind == HistoryAction.KIND_COMPUTE and not is_compute:
                name = action.func_name or action.title or action.uuid
                panel.runtime.execution.cascade_warnings.append(
                    _("Action %s has no recorded pattern and cannot be replayed.")
                    % name
                )
                continue
            if not is_creation and not is_compute:
                deferred_actions.append(action)
                continue
            if prompt:
                result = prompt_edit_action_params(panel, action)
                if result is False:
                    for selected_action in ordered:
                        kwargs, saved_kwargs = entry_states[selected_action.uuid]
                        selected_action.kwargs = kwargs
                        selected_action.saved_kwargs = saved_kwargs
                        panel.tree.refresh_action_item(selected_action)
                    _recompute_stale_actions(panel, ordered)
                    return
                if result is True:
                    edited_actions.append(action)
            recomputable.append(action)

        for action in edited_actions:
            panel.tree.refresh_action_item(action)
        planned = list(recomputable)
        for action in edited_actions:
            planned.extend(hchain.get_downstream_actions(panel, action))
        planned = order_selected_actions(panel, planned)
        execution_plan = order_selected_actions(panel, deferred_actions + planned)
        for action in planned:
            action.is_stale = True
            panel.tree.refresh_action_item(action)
        QW.QApplication.processEvents()
        blocked_outputs: set[str] = set()
        try:
            for action in execution_plan:
                if action in deferred_actions:
                    if hchain.action_mutates_any(action, blocked_outputs):
                        # Mutation targeting an object whose recompute failed
                        # upstream: skip it like a blocked compute.
                        continue
                    if (
                        action.kind == HistoryAction.KIND_UI
                        and action.method_name in HistoryAction.DESTRUCTIVE_METHODS
                        and not action.is_current_state_compatible(panel.mainwindow)
                    ):
                        # Destructive action whose captured selection no longer
                        # resolves (e.g. recorded deletion of objects that a
                        # replayed load just re-created under new UUIDs): skip
                        # it instead of failing the whole plan on restore.
                        name = action.title or action.method_name or action.uuid
                        panel.runtime.execution.cascade_warnings.append(
                            _(
                                "Action %s targets objects that no longer exist "
                                "and was skipped."
                            )
                            % name
                        )
                        continue
                    is_load_action = (
                        action.kind == HistoryAction.KIND_UI
                        and action.method_name in HistoryAction.UI_LOAD_METHODS
                    )
                    if is_load_action:
                        if _load_outputs_still_exist(panel, action):
                            # All loaded objects still exist: replaying would
                            # duplicate them, so skip the load action.
                            continue
                        if action.kwargs.get("add_objects") is False:
                            # Legacy entries recorded by ``load_from_directory``
                            # with ``add_objects=False``: self-heal so replay
                            # actually adds the loaded objects.
                            action.kwargs["add_objects"] = True
                    if (
                        action.kind == HistoryAction.KIND_UI
                        and action.method_name in HistoryAction.FILE_OUTPUT_METHODS
                        and not confirm_file_output_replay(panel, action)
                    ):
                        # User declined (or unattended default): skip the
                        # file-save action cleanly, the plan continues.
                        continue
                    data_panels = (
                        panel.mainwindow.signalpanel,
                        panel.mainwindow.imagepanel,
                    )
                    before_ids = (
                        {
                            p.PANEL_STR_ID: set(p.objmodel.get_object_ids())
                            for p in data_panels
                        }
                        if is_load_action
                        else None
                    )
                    payload_before = action.kwargs.get("payload")
                    with panel.replaying(), panel.output_suppressed():
                        action.replay(
                            panel.mainwindow, restore_selection=True, edit=prompt
                        )
                    if before_ids is not None:
                        new_uuids = [
                            uid
                            for p in data_panels
                            for uid in p.objmodel.get_object_ids()
                            if uid not in before_ids[p.PANEL_STR_ID]
                        ]
                        if new_uuids:
                            # Re-bind the load action to the freshly loaded
                            # objects: replay assigns new UUIDs, and stale
                            # recorded outputs would break duplicate detection
                            # and downstream reconnection.
                            panel.register_action_outputs(action, new_uuids)
                    if (
                        prompt
                        and action.kind == HistoryAction.KIND_MUTATION
                        and action.kwargs.get("payload") is not payload_before
                    ):
                        # The mutation payload was edited in the dialog:
                        # recompute the downstream closure (seeded from the
                        # mutation targets, see ``get_downstream_actions``).
                        panel.tree.refresh_action_item(action)
                        hrec.recompute_cascade(panel, action)
                    continue
                if hchain.action_consumes_any(action, blocked_outputs):
                    blocked_outputs.update(
                        hchain.recorded_action_output_uuids(panel, action)
                    )
                    continue
                success = hrec.recompute_action_in_place(panel, action)
                action.is_stale = not success
                panel.tree.refresh_action_item(action)
                if not success:
                    blocked_outputs.update(
                        hchain.recorded_action_output_uuids(panel, action)
                    )
        finally:
            hrec.flush_cascade_warnings(panel)
        QW.QApplication.processEvents()


def order_selected_actions(
    panel: HistoryPanel, actions: list[HistoryAction]
) -> list[HistoryAction]:
    """Deduplicate ``actions`` and sort them by (session, position) order."""
    rank: dict[str, int] = {}
    pos = 0
    for session in panel.history_sessions:
        for action in session.actions:
            rank[action.uuid] = pos
            pos += 1
    seen: set[str] = set()
    unique: list[HistoryAction] = []
    for action in actions:
        if action.uuid in seen:
            continue
        seen.add(action.uuid)
        unique.append(action)
    unique.sort(key=lambda a: rank.get(a.uuid, 0))
    return unique


def restore_action_params(
    panel: HistoryPanel, item: HistoryAction | HistorySession
) -> None:
    """Restore original kwargs from snapshot and recompute in-place.

    Every targeted action is recomputed unconditionally, even when it has no
    pending parameter edits, so that stale markers are cleared on success.
    """
    actions: list[HistoryAction]
    if isinstance(item, HistorySession):
        actions = [
            a
            for a in item.actions
            if a.kind in (HistoryAction.KIND_COMPUTE, HistoryAction.KIND_MUTATION)
            or (
                a.kind == HistoryAction.KIND_UI
                and a.method_name in HistoryAction.UI_CREATION_METHODS
            )
        ]
    else:
        actions = [item]
    try:
        for action in actions:
            action.restore_kwargs()
            panel.tree.refresh_action_item(action)
            success = hrec.recompute_action_in_place(panel, action)
            action.is_stale = not success
            panel.tree.refresh_action_item(action)
            if not success:
                break
            if not isinstance(item, HistorySession):
                hrec.recompute_cascade(panel, action)
    finally:
        hrec.flush_cascade_warnings(panel)
        panel.ui.update_actions_state()
