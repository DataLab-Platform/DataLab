# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Interactive (dialog-driven) replay helpers for the History panel."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import guidata.dataset as gds
from qtpy import QtWidgets as QW
from sigima.objects.base import BaseROIParam

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
    against the current workspace state: any incompatibility (e.g. an input
    object was deleted, breaking the chain) triggers a resolution dialog
    proposing to repair the history — remove the broken actions and their
    downstream dependents, then continue with the remaining actions — or to
    cancel the whole command (see :func:`confirm_broken_chain_repair`).

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
    broken = [
        entry
        for entry in selected
        if not entry.is_current_state_compatible(panel.mainwindow)
    ]
    if broken:
        if not confirm_broken_chain_repair(panel, broken):
            return
        repair_broken_entries(panel, broken)
        hchain.refresh_reconnected_history(panel)
        selected = [
            entry
            for entry in selected
            if _entry_still_in_history(panel, entry)
            and entry.is_current_state_compatible(panel.mainwindow)
        ]
        if not selected:
            return
    edit_mode = panel.runtime.execution.edit_mode
    actions_to_replay: list[HistoryAction] = []
    for session_or_action in selected:
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


def confirm_broken_chain_repair(
    panel: HistoryPanel, broken: list[HistorySession | HistoryAction]
) -> bool:
    """Propose repairing the history when the selected chain is broken.

    One or more selected sessions or actions reference objects that no longer
    exist in the workspace (deleted inputs), so their processing chain is no
    longer guaranteed. The user may repair the history — remove the broken
    actions and their downstream dependents, then continue with the remaining
    actions — or cancel (nothing is modified). In unattended mode no dialog
    is shown: repair is chosen when ``execenv.accept_dialogs`` is set,
    otherwise the command is cancelled.

    Args:
        panel: History panel instance
        broken: Selected sessions or actions whose recorded workspace state
         no longer matches the current workspace

    Returns:
        True if the history should be repaired and the command continued.
    """
    if execenv.unattended:
        return bool(execenv.accept_dialogs)
    names: list[str] = []
    for entry in broken:
        if isinstance(entry, HistorySession):
            names.append(entry.title)
        else:
            names.append(entry.title or entry.func_name or entry.uuid)
    msgbox = QW.QMessageBox(panel.mainwindow)
    msgbox.setWindowTitle(_("Broken processing chain"))
    msgbox.setIcon(QW.QMessageBox.Icon.Warning)
    msgbox.setText(
        _(
            "One or more objects used by the selected processing chain were "
            "deleted: the chain is broken and the following entries can no "
            "longer be replayed:\n%s\n\n"
            "Repair the history by removing the broken actions (and the "
            "actions depending on them), then continue with the remaining "
            "actions?"
        )
        % "\n".join("• " + name for name in names)
    )
    repair_button = msgbox.addButton(
        _("Repair and continue"), QW.QMessageBox.ButtonRole.YesRole
    )
    cancel_button = msgbox.addButton(QW.QMessageBox.Cancel)
    msgbox.setDefaultButton(cancel_button)
    msgbox.exec()
    return msgbox.clickedButton() is repair_button


def collect_broken_chain_actions(
    panel: HistoryPanel, broken: list[HistorySession | HistoryAction]
) -> list[HistoryAction]:
    """Return the incompatible actions of ``broken`` plus their dependents.

    For a selected session, every action whose recorded workspace state is
    incompatible with the current workspace is included; a directly selected
    action is included as is. The downstream closure of each broken action
    (actions consuming its outputs, transitively) is appended through
    :func:`datalab.gui.panel.history.chain.get_downstream_actions`.

    Args:
        panel: History panel instance
        broken: Selected sessions or actions flagged as incompatible

    Returns:
        Actions to remove from the history, in discovery order
    """
    roots: list[HistoryAction] = []
    for entry in broken:
        candidates = (
            [
                action
                for action in entry.actions
                if not action.is_current_state_compatible(panel.mainwindow)
            ]
            if isinstance(entry, HistorySession)
            else [entry]
        )
        for action in candidates:
            if action not in roots:
                roots.append(action)
    to_remove = list(roots)
    for action in roots:
        # Same-session closure only: later sessions survive (objects not deleted)
        for downstream in hchain.get_downstream_actions(panel, action):
            if downstream not in to_remove:
                to_remove.append(downstream)
    return to_remove


def repair_broken_entries(
    panel: HistoryPanel, broken: list[HistorySession | HistoryAction]
) -> None:
    """Remove broken actions and their downstream dependents from the history.

    Args:
        panel: History panel instance
        broken: Selected sessions or actions flagged as incompatible
    """
    for action in collect_broken_chain_actions(panel, broken):
        hchain.remove_single_action(panel, action)


def _entry_still_in_history(
    panel: HistoryPanel, entry: HistorySession | HistoryAction
) -> bool:
    """Return True if ``entry`` survived the history repair."""
    if isinstance(entry, HistorySession):
        return entry in panel.history_sessions and bool(entry.actions)
    return hchain.find_parent_session(panel, entry) is not None


def action_has_roi_params(action: HistoryAction) -> bool:
    """Return whether ``action`` was recorded with region-of-interest parameters.

    Args:
        action: Recorded action to inspect

    Returns:
        True if at least one recorded parameter is a ROI parameter.
    """
    values: list[Any] = []
    for key in ("param", "params"):
        value = action.kwargs.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(value)
        elif value is not None:
            values.append(value)
    return any(isinstance(value, BaseROIParam) for value in values)


def inform_roi_edit_unsupported(panel: HistoryPanel) -> None:
    """Tell the user that ROI parameters cannot be edited from the history.

    Args:
        panel: History panel instance
    """
    if execenv.unattended:
        return
    QW.QMessageBox.information(
        panel.mainwindow,
        _("Recompute regions of interest"),
        _(
            "Regions of interest cannot be edited from the History panel: "
            "the ROI editor cannot be reopened with the recorded parameters. "
            "The recorded regions of interest are kept as is."
        ),
    )


def prepare_action_param_edit(action: HistoryAction) -> ActionParamEdit | None:
    """Prepare the editable parameter copy for ``action``."""
    result = None
    if action_has_roi_params(action):
        # ROIs are defined with the interactive ROI editor, which cannot be
        # reopened with the recorded parameters.
        return None
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


def destructive_replay_skip_reason(
    panel: HistoryPanel,
    action: HistoryAction,
    recreated_outputs: set[str],
) -> str | None:
    """Return a warning when a destructive action must not be replayed.

    A destructive UI action (``DESTRUCTIVE_METHODS``) is skipped when:

    1. Its captured selection no longer resolves (e.g. recorded deletion of
       objects that a replayed load just re-created under new UUIDs).
    2. It targets objects that were (re)computed earlier in the same
       execution plan: replaying the recorded deletion would immediately
       destroy the outputs this replay just restored.
    3. It targets outputs produced by another session's actions: replaying
       one session must never delete objects owned by a different session
       (e.g. a deletion of the original chain recorded while a duplicated
       session was active).

    Args:
        panel: History panel instance
        action: Destructive UI action about to be replayed
        recreated_outputs: Output UUIDs already (re)computed by this plan

    Returns:
        The warning message when the action must be skipped, ``None`` when
        the action may be replayed.
    """
    name = action.title or action.method_name or action.uuid
    if not action.is_current_state_compatible(panel.mainwindow):
        return (
            _(
                "Action %s was skipped: its recorded workspace state no "
                "longer matches the current workspace."
            )
            % name
        )
    captured = {uuid for uuids in action.state.selection.values() for uuid in uuids}
    if captured & recreated_outputs:
        return (
            _(
                "Action %s would delete objects that this replay just "
                "recomputed and was skipped."
            )
            % name
        )
    own_session = hchain.find_parent_session(panel, action)
    if own_session is not None:
        own_action_uuids = {act.uuid for act in own_session.actions}
        for uuid in captured:
            producer_uuid = panel.runtime.objects.output_to_action.get(uuid)
            if producer_uuid is not None and producer_uuid not in own_action_uuids:
                return (
                    _(
                        "Action %s targets objects produced by another "
                        "session and was skipped."
                    )
                    % name
                )
    return None


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
    try:
        run_replay_actions(panel, actions, prompt)
    finally:
        panel.ui.update_actions_state()


def run_replay_actions(
    panel: HistoryPanel, actions: list[HistoryAction], prompt: bool
) -> None:
    """Execute the replay plan (engine core of :func:`replay_actions`).

    Args:
        panel: History panel instance
        actions: Selected actions to replay
        prompt: Open parameter dialogs before recomputing
    """
    # Deduplicate and sort the selected actions in their session order
    ordered = order_selected_actions(panel, actions)
    if not ordered:
        return
    # Non-editable actions (ROIs, interactive fits) only report their refusal
    # when a single action was selected: replaying a session or a batch keeps
    # their recorded parameters silently.
    report_non_editable = prompt and len(ordered) == 1
    with panel.runtime.execution.replaying_edits() as started:
        if not started:
            return
        panel.ui.update_actions_state()
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
                if (
                    result is None
                    and report_non_editable
                    and action_has_roi_params(action)
                ):
                    inform_roi_edit_unsupported(panel)
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
        recreated_outputs: set[str] = set()
        try:
            for action in execution_plan:
                if action in deferred_actions:
                    if getattr(action, "decode_failed", False):
                        # Deferred actions bypass ``recompute_action_in_place``
                        # (direct ``action.replay``): apply the same guard so
                        # broken persisted parameters are never executed.
                        panel.runtime.execution.cascade_warnings.append(
                            _(
                                "Action %s was skipped: its recorded "
                                "parameters could not be read from the "
                                "history file."
                            )
                            % (action.title or action.method_name or action.uuid)
                        )
                        continue
                    if hchain.action_mutates_any(action, blocked_outputs):
                        # Mutation targeting an object whose recompute failed
                        # upstream: skip it like a blocked compute.
                        continue
                    if (
                        action.kind == HistoryAction.KIND_UI
                        and action.method_name in HistoryAction.DESTRUCTIVE_METHODS
                    ):
                        skip_reason = destructive_replay_skip_reason(
                            panel, action, recreated_outputs
                        )
                        if skip_reason is not None:
                            panel.runtime.execution.cascade_warnings.append(skip_reason)
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
                    with panel.replaying(), panel.output_suppressed():
                        action.replay(
                            panel.mainwindow,
                            restore_selection=True,
                            edit=report_non_editable,
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
                else:
                    recreated_outputs.update(
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
