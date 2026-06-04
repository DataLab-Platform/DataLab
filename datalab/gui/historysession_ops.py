# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Helpers for History panel session recording and indexing."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Generator

from datalab.history import HistoryAction, HistorySession, WorkspaceState
from datalab.history.core import _resolve_self_target

if TYPE_CHECKING:
    from datalab.gui.panel.history import HistoryPanel

_logger = logging.getLogger(__name__)


def create_new_session(panel: HistoryPanel) -> None:
    """Create a new history list"""
    panel.session_increment += 1
    session = HistorySession(number=panel.session_increment)
    panel.history_sessions.append(session)
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()


def start_new_session_after_workspace_reset(panel: HistoryPanel) -> None:
    """Start a new history session after a workspace reset, when useful."""
    if panel.history_sessions and panel.history_sessions[-1].actions:
        panel.create_new_session()


def add_compute_entry(
    panel: HistoryPanel,
    action_title: str,
    panel_str: str,
    func_name: str,
    pattern: str,
    save_state: bool = True,
    output_uuids: list[str] | None = None,
    plugin_origin: dict[str, Any] | None = None,
    **kwargs: Any,
) -> HistoryAction | None:
    """Record a *compute* action in the current history session.

    Args:
        action_title: Title shown in the history tree.
        panel_str: ``"signal"`` or ``"image"``.
        func_name: Sigima feature name (resolvable via
         :meth:`BaseProcessor.get_feature`).
        pattern: One of ``"1_to_1"``, ``"1_to_0"``, ``"n_to_1"``, ``"2_to_1"``,
         ``"1_to_n"``, ``"multiple_1_to_1"`` (the latter is recorded for
         traceability but not replayable).
        save_state: If True, capture the workspace state for replay.
        output_uuids: Optional list of UUIDs of the data objects produced by
         this action. When known at call time, prefer passing it here so the
         bijective mapping is initialised in one step. Most callers do not
         know the outputs yet and instead wrap the compute call with
         :meth:`capture_outputs` (or call :meth:`register_action_outputs`
         explicitly afterwards) using the returned action.
        plugin_origin: Optional plugin origin descriptor (see
         :func:`datalab.gui.processor.base._detect_plugin_origin`). ``None``
         for built-in Sigima/DataLab features.
        **kwargs: Extra primitive kwargs (``param``, ``obj2_uuids``,
         ``obj2_name``, ``pairwise``, ``params`` (list of DataSet),
         ``func_names`` (list of str), ...). ``DataSet`` instances are
         serialised as JSON.

    Returns:
        The created :class:`HistoryAction`, or ``None`` if recording is
        disabled (record mode off or replay in progress).
    """
    if not panel.record_mode_enabled or panel.is_replaying():
        return None
    state = WorkspaceState()
    if save_state:
        state.save(panel.mainwindow)
    # Deep-copy kwargs so each action owns independent parameter
    # instances. Without this, consecutive applications of the same
    # function (e.g. two gaussian_filter calls with different sigma)
    # would share a single DataSet object and editing one action's
    # parameters would silently mutate the other.
    action = HistoryAction(
        title=action_title,
        kind=HistoryAction.KIND_COMPUTE,
        panel_str=panel_str,
        func_name=func_name,
        pattern=pattern,
        kwargs=deepcopy(kwargs),
        state=state,
        plugin_origin=plugin_origin,
    )
    panel.add_object(action)
    if output_uuids is not None:
        panel.register_action_outputs(action, output_uuids)
    return action


def add_compute_entry_from_pp(
    panel: HistoryPanel,
    action_title: str,
    pp: Any,  # ProcessingParameters (avoid circular import)
    panel_str: str,
    save_state: bool = True,
    output_uuids: list[str] | None = None,
    plugin_origin: dict[str, Any] | None = None,
    **extras: Any,
) -> HistoryAction | None:
    """Record a *compute* action derived from a ``ProcessingParameters``.

    Bridges the dash-form pattern used in object metadata
    (``"1-to-1"`` …) with the underscore form expected by
    :class:`HistoryAction` (``"1_to_1"`` …) so that both sides share
    a single identity (``func_name`` / ``pattern`` / ``param``).

    Args:
        action_title: Title shown in the history tree.
        pp: :class:`~datalab.gui.processor.base.ProcessingParameters`
         instance describing the operation.
        panel_str: ``"signal"`` or ``"image"``.
        save_state: If True, capture the workspace state for replay.
        output_uuids: Optional list of UUIDs of the data objects produced
         by this action (see :meth:`add_compute_entry`).
        plugin_origin: Optional plugin origin descriptor (see
         :meth:`add_compute_entry`).
        **extras: Additional history-only kwargs (``obj2_uuids``,
         ``obj2_name``, ``pairwise``, ``params``, ``func_names``…).

    Returns:
        The created :class:`HistoryAction`, or ``None`` if recording is
        disabled.
    """
    hist_pattern = pp.pattern.replace("-", "_")
    kwargs: dict[str, Any] = {}
    if pp.param is not None and "param" not in extras and "params" not in extras:
        kwargs["param"] = pp.param
    kwargs.update(extras)
    return panel.add_compute_entry(
        action_title,
        panel_str=panel_str,
        func_name=pp.func_name,
        pattern=hist_pattern,
        save_state=save_state,
        output_uuids=output_uuids,
        plugin_origin=plugin_origin,
        **kwargs,
    )


def register_action_outputs(
    panel: HistoryPanel, action: HistoryAction, output_uuids: list[str]
) -> None:
    """Register the data objects produced by ``action``.

    Maintains the bijective ``action → outputs`` and ``output → action``
    mappings. May be called multiple times for a given action (later calls
    replace earlier ones, e.g. after a cascade recompute).

    Args:
        action: The history action that produced the outputs.
        output_uuids: UUIDs of the produced data objects (empty for
         ``1_to_0`` analysis patterns and for UI actions that did not
         create new objects).
    """
    # Drop previous outputs for this action from the reverse index.
    previous = panel._action_output_uuids.get(action.uuid, [])
    for prev_uuid in previous:
        if panel._output_to_action.get(prev_uuid) == action.uuid:
            panel._output_to_action.pop(prev_uuid, None)
    new_outputs = list(output_uuids)
    # Ownership transfer: if an output_uuid already belongs to a
    # *different* action, remove it from that action's output list so the
    # forward mapping stays consistent.  The HistoryAction object's
    # ``output_uuids`` attribute is NOT updated here because traversing all
    # sessions to locate the object would be expensive; the panel-level
    # dicts are the source of truth.
    for out_uuid in new_outputs:
        old_action_uuid = panel._output_to_action.get(out_uuid)
        if old_action_uuid is not None and old_action_uuid != action.uuid:
            old_list = panel._action_output_uuids.get(old_action_uuid)
            if old_list is not None:
                try:
                    old_list.remove(out_uuid)
                except ValueError:
                    pass
                if not old_list:
                    del panel._action_output_uuids[old_action_uuid]
            _logger.debug(
                "Output %s transferred from action %s to %s",
                out_uuid,
                old_action_uuid,
                action.uuid,
            )
    action.output_uuids = list(new_outputs)
    panel._action_output_uuids[action.uuid] = new_outputs
    for out_uuid in new_outputs:
        panel._output_to_action[out_uuid] = action.uuid


@contextmanager
def capture_outputs(
    panel: HistoryPanel, action: HistoryAction | None
) -> Generator[None, None, None]:
    """Context manager: snapshot panel object IDs and record diffs as outputs.

    Use around any compute call when the produced UUIDs are not known
    upfront. On exit, every newly-added object (signal or image) is
    registered as an output of ``action`` via
    :meth:`register_action_outputs`. No-op when ``action`` is ``None``
    (recording disabled).

    Args:
        action: The history action being processed, or ``None``.
    """
    if action is None:
        yield
        return
    panels = (panel.mainwindow.signalpanel, panel.mainwindow.imagepanel)
    before = {p.PANEL_STR_ID: set(p.objmodel.get_object_ids()) for p in panels}
    try:
        yield
    finally:
        new_uuids: list[str] = []
        for p in panels:
            before_p = before[p.PANEL_STR_ID]
            for uid in p.objmodel.get_object_ids():
                if uid not in before_p:
                    new_uuids.append(uid)
        panel.register_action_outputs(action, new_uuids)


def add_ui_entry(
    panel: HistoryPanel,
    action_title: str,
    target: str,
    method_name: str,
    save_state: bool = True,
    **kwargs: Any,
) -> None:
    """Record a *UI* action in the current history session.

    Args:
        action_title: Title shown in the history tree.
        target: One of ``"mainwindow"``, ``"signalpanel"``, ``"imagepanel"``,
         ``"historypanel"`` -- attribute path on the main window.
        method_name: Method name to call on ``target`` at replay time.
        save_state: If True, capture the workspace state for replay.
        **kwargs: Method keyword arguments. ``DataSet`` instances are
         serialised as JSON; other values must be HDF5-friendly primitives.
    """
    if not panel.record_mode_enabled or panel.is_replaying():
        return
    state = WorkspaceState()
    if save_state:
        state.save(panel.mainwindow)
    # Deep-copy kwargs to ensure independent parameter ownership
    # (same rationale as in add_compute_entry).
    action = HistoryAction(
        title=action_title,
        kind=HistoryAction.KIND_UI,
        target=target,
        method_name=method_name,
        kwargs=deepcopy(kwargs),
        state=state,
    )
    panel.add_object(action)


def add_entry(
    panel: HistoryPanel,
    action_title: str,
    save_state: bool,
    func: Callable,
    **kwargs,
) -> None:
    """Legacy entry-point kept as a compatibility shim.

    Most call sites have been migrated to :meth:`add_compute_entry` or
    :meth:`add_ui_entry`. The remaining paths -- and the
    :func:`add_to_history` decorator -- still call ``add_entry`` with a
    bound method; we infer the ``(target, method_name)`` from the bound
    ``func.__self__`` and route to :meth:`add_ui_entry`.
    """
    if not panel.record_mode_enabled or panel.is_replaying():
        return
    target = None
    if hasattr(func, "__self__"):
        target = _resolve_self_target(func.__self__)
    if target is None:
        # Cannot route safely -- skip rather than pickle a Callable.
        return
    panel.add_ui_entry(
        action_title,
        target=target,
        method_name=func.__name__,
        save_state=save_state,
        **kwargs,
    )


def add_object(panel: HistoryPanel, obj: HistoryAction) -> None:
    """Add object to panel"""
    if not panel.history_sessions:
        panel.create_new_session()
    panel.history_sessions[-1].add_action(obj)
    panel.tree.add_action_to_tree(obj)
    panel.tree.rearrange_tree()
    panel.refresh_compatibility_items()
    panel.update_actions_state()
