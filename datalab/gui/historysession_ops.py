# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Helpers for History panel session recording and indexing."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generator, Literal

from qtpy import QtWidgets as QW

from datalab.config import Conf, _
from datalab.env import execenv
from datalab.gui.panel.history import chain as hchain
from datalab.history import HistoryAction, HistorySession, WorkspaceState

if TYPE_CHECKING:
    from datalab.gui.panel.history import HistoryPanel


SessionBehavior = Literal["ask", "yes", "no"]
SESSION_BEHAVIORS: tuple[SessionBehavior, ...] = ("ask", "yes", "no")


def create_new_session(panel: HistoryPanel) -> HistorySession:
    """Create a new history session and make it the active recording session.

    Returns:
        The newly created session.
    """
    panel.navigation.session_increment += 1
    session = HistorySession(number=panel.navigation.session_increment)
    panel.history_sessions.append(session)
    panel.navigation.set_active_session(session)
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    return session


def start_new_session_after_workspace_reset(panel: HistoryPanel) -> None:
    """Start a new history session after a workspace reset, when useful."""
    if panel.history_sessions and panel.history_sessions[-1].actions:
        panel.create_new_session()


def maybe_start_session_for_input(
    panel: HistoryPanel,
    *,
    load: bool = False,
    behavior: SessionBehavior | None = None,
) -> bool:
    # pylint: disable=too-many-return-statements
    """Offer to start a new history session before a creation/load is recorded.

    When the active recording session already contains actions, prompt the user
    to start a fresh session so the new creation/load becomes the root of a
    clean, self-contained pipeline. A new session is opened *before* the action
    is recorded when the user accepts.

    Args:
        load: True when triggered by a file/workspace load, False for an object
         creation. Only affects the prompt wording.
        behavior: Session creation policy: ask, always create ("yes"), or keep
         the current session ("no"). Defaults to the live general policy.

    Returns:
        True if a new session was created.

    Raises:
        ValueError: If ``behavior`` is unsupported.
    """
    if behavior is None:
        behavior = Conf.history_new_session_behavior.get()
    if behavior not in SESSION_BEHAVIORS:
        raise ValueError(f"Invalid session behavior: {behavior!r}")
    if not panel.record_mode_enabled or panel.is_replaying():
        return False
    if panel.runtime.execution.suppress_session_prompt:
        return False
    active_session = panel.navigation.get_active_session()
    if active_session is None or not active_session.actions:
        return False
    if behavior == "no":
        return False
    if behavior == "yes":
        panel.create_new_session()
        return True
    # Debounce: a synchronous burst of creations (plugin/macro) must prompt only
    # once. The guard is reset on the next event-loop turn.
    if not panel.runtime.execution.start_session_input_prompt():
        return False
    if execenv.unattended:
        # Headless runs: honor the accept_dialogs flag (default False -> "No"),
        # so tests can drive the behavior without a real modal dialog.
        if execenv.accept_dialogs:
            panel.create_new_session()
            return True
        return False
    if load:
        message = _("A new object was loaded. Start a new history session?")
    else:
        message = _("A new object was created. Start a new history session?")
    answer = QW.QMessageBox.question(
        panel.mainwindow,
        _("New history session"),
        message,
        QW.QMessageBox.Yes | QW.QMessageBox.No,
    )
    if answer == QW.QMessageBox.Yes:
        panel.create_new_session()
        return True
    return False


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
         ``"1_to_n"``, ``"multiple_1_to_1"`` (the latter is replayable via
         the generic compute replay, like the other compute patterns).
        save_state: If True, capture the workspace state for replay.
        output_uuids: Optional list of UUIDs of the data objects produced by
         this action. When known at call time, prefer passing it here so the
         action-to-outputs mapping and inverse output lookup are initialised in
         one step. Most callers do not know the outputs yet and instead wrap
         the compute call with :meth:`capture_outputs` (or call
         :meth:`register_action_outputs` explicitly afterwards) using the
         returned action.
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
        state.save(panel.mainwindow, panel_str=panel_str)
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
    )
    action.plugin_origin = deepcopy(plugin_origin)
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

    Maintains the ``action → outputs`` mapping and inverse ``output → action``
    lookup. One action may produce multiple outputs. May be called multiple
    times for a given action (later calls replace earlier ones, e.g. after a
    cascade recompute).

    Args:
        action: The history action that produced the outputs.
        output_uuids: UUIDs of the produced data objects (empty for
         ``1_to_0`` analysis patterns and UI actions without new objects;
         output-producing UI actions may provide one or more UUIDs).
    """
    panel.runtime.objects.register_action_outputs(action, output_uuids)


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
        if not new_uuids:
            no_output_compute = action.kind == HistoryAction.KIND_COMPUTE and (
                action.pattern
                in {"1_to_1", "multiple_1_to_1", "1_to_n", "n_to_1", "2_to_1"}
            )
            no_output_load = (
                action.kind == HistoryAction.KIND_UI
                and action.method_name in HistoryAction.UI_LOAD_METHODS
            )
            if no_output_compute or no_output_load:
                # The action produced no output object: either the compute
                # failed (or was a full no-op), or the load found nothing
                # readable for the panel. Do not keep a misleading entry in
                # the history.
                discard_empty_output_action(panel, action)


def discard_empty_output_action(panel: HistoryPanel, action: HistoryAction) -> None:
    """Remove a just-recorded action that produced no output object.

    Removes the action from the session chain and refreshes the tree so the
    panel stays consistent.

    Args:
        action: The history action to discard.
    """
    hchain.remove_single_action(panel, action)
    panel.tree.populate_tree(panel.history_sessions)
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()


def add_ui_entry(
    panel: HistoryPanel,
    action_title: str,
    target: str,
    method_name: str,
    save_state: bool = True,
    **kwargs: Any,
) -> HistoryAction | None:
    """Record a *UI* action in the current history session.

    Args:
        action_title: Title shown in the history tree.
        target: One of ``"mainwindow"``, ``"signalpanel"``, ``"imagepanel"``,
         ``"historypanel"``, ``"signalprocessor"``, or ``"imageprocessor"``.
        method_name: Method name to call on ``target`` at replay time.
        save_state: If True, capture the workspace state for replay.
        **kwargs: Method keyword arguments. ``DataSet`` instances are
         serialised as JSON; other values must be HDF5-friendly primitives.

    Returns:
        The created :class:`HistoryAction`, or ``None`` if recording is
        disabled (record mode off or replay in progress).
    """
    if not panel.record_mode_enabled or panel.is_replaying():
        return None
    # Derive the action's panel from the UI target so prompting and captured
    # state concern the panel the action actually operates on.
    target_panel_str = {
        "signalpanel": "signal",
        "imagepanel": "image",
        "signalprocessor": "signal",
        "imageprocessor": "image",
    }.get(target)
    # When the entry is an object creation, offer to start a fresh history
    # session first so the creation becomes the root of a clean pipeline.
    if method_name in HistoryAction.UI_CREATION_METHODS:
        panel.maybe_start_session_for_input(load=False)
    state = WorkspaceState()
    if save_state:
        state.save(panel.mainwindow, panel_str=target_panel_str)
    # Deep-copy kwargs to ensure independent parameter ownership
    # (same rationale as in add_compute_entry).
    action = HistoryAction(
        title=action_title,
        kind=HistoryAction.KIND_UI,
        target=target,
        method_name=method_name,
        kwargs=deepcopy(kwargs),
        state=state,
        panel_str=target_panel_str,
    )
    panel.add_object(action)
    return action


def add_mutation_entry(
    panel: HistoryPanel,
    action_title: str,
    panel_str: str,
    mutation_key: str,
    target_uuids: list[str],
    payload: Any = None,
    save_state: bool = True,
) -> HistoryAction | None:
    """Record a *mutation* action in the current history session.

    Mutation actions describe in-place modifications of existing data objects
    (no new objects created), e.g. ROI assignment or removal. At replay time,
    the payload is re-applied to each target object still present in the data
    panel's object model.

    Args:
        action_title: Title shown in the history tree.
        panel_str: Data panel the mutation operates on ("signal" or "image").
        mutation_key: Mutation identifier (currently only "roi" is supported).
        target_uuids: UUIDs of the data objects modified in place.
        payload: Mutation payload (e.g. a sigima ROI object). ``None`` means
         the attribute is removed at replay time (e.g. ROI deletion).
        save_state: If True, capture the workspace state for replay.

    Returns:
        The created :class:`HistoryAction`, or ``None`` if recording is
        disabled (record mode off or replay in progress).
    """
    if not panel.record_mode_enabled or panel.is_replaying():
        return None
    state = WorkspaceState()
    if save_state:
        state.save(panel.mainwindow, panel_str=panel_str)
    # Deep-copy the payload to ensure independent ownership (same rationale
    # as in add_compute_entry). A None payload is encoded as a missing kwarg.
    action = HistoryAction(
        title=action_title,
        kind=HistoryAction.KIND_MUTATION,
        panel_str=panel_str,
        mutation_key=mutation_key,
        target_uuids=list(target_uuids),
        kwargs={"payload": deepcopy(payload)} if payload is not None else {},
        state=state,
    )
    panel.add_object(action)
    return action


def add_object(panel: HistoryPanel, obj: HistoryAction) -> None:
    """Add an action to the single active recording session.

    Actions from both the signal and image panels are chained into the same
    active recording session, creating one on first use, so mixed-panel
    pipelines stay together and recording resumes in the user-selected session.
    """
    session = panel.navigation.get_active_session()
    if session is None:
        session = panel.create_new_session()
    session.add_action(obj)
    session_index = panel.history_sessions.index(session)
    panel.tree.rebuild_session(session_index)
    panel.tree.rearrange_tree()
    panel.refresh_compatibility_items()
    panel.ui.update_actions_state()
