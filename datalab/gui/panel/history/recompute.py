# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""In-place recompute helpers for the History panel cascade."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj
from sigima.objects.base import ROI_KEY

from datalab.config import _
from datalab.env import execenv
from datalab.gui.creation import (
    create_image_from_param,
    create_signal_from_param,
    insert_creation_parameters,
    prepare_signal_parameters,
)
from datalab.gui.panel.history import chain as hchain
from datalab.gui.processor.base import (
    FeatureNotFoundError,
    ProcessingParameters,
    extract_analysis_parameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.history import HistoryAction
from datalab.history.core import decode_roi
from datalab.history.effects import AnalysisEffects, capture_effects, merge_effects
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.panel.history.panel import HistoryPanel

_logger = logging.getLogger(__name__)


def refresh_action(panel: HistoryPanel, action: HistoryAction) -> None:
    """Refresh the tree display for ``action`` after its kwargs were mutated.

    Used by :meth:`ObjectProp.apply_processing_parameters` to update the
    Description column when the user edits a ``param`` from the Processing
    tab of the Signal/Image panel.
    """
    panel.tree.refresh_action_item(action)


def update_obj_in_place(
    target_obj: SignalObj | ImageObj,
    new_obj: SignalObj | ImageObj,
) -> None:
    """Copy data + title + metadata from ``new_obj`` onto ``target_obj``.

    Preserves the target's identity (UUID, panel position, references)
    while reflecting all user-visible changes produced by a recompute.
    """
    target_obj.title = new_obj.title
    if isinstance(target_obj, SignalObj):
        target_obj.xydata = new_obj.xydata
    else:
        target_obj.data = new_obj.data
        target_obj.invalidate_maskdata_cache()
    # Read everything that may raise AttributeError (missing/None metadata)
    # before wiping the target, so a failure cannot leave metadata half-updated.
    try:
        saved_uuid = target_obj.metadata.get("__uuid")
        saved_number = target_obj.metadata.get("__number")
        # Align with the 1_to_1 path: keep the target's user ROI when the
        # freshly computed object does not carry one.
        saved_roi = target_obj.metadata.get(ROI_KEY)
        new_metadata = dict(new_obj.metadata)
    except AttributeError:
        return
    target_obj.metadata.clear()
    target_obj.metadata.update(new_metadata)
    if saved_uuid is not None:
        target_obj.metadata["__uuid"] = saved_uuid
    if saved_number is not None:
        target_obj.metadata["__number"] = saved_number
    if saved_roi is not None and ROI_KEY not in target_obj.metadata:
        target_obj.metadata[ROI_KEY] = saved_roi


def refresh_target(panel_data: BaseDataPanel, output_uuid: str) -> None:
    """Refresh tree item + plot for ``output_uuid`` in ``panel_data``.

    Also updates the Properties panel when the refreshed object is
    currently selected, marks the object as freshly processed so the
    Processing tab is shown, and emits ``SIG_OBJECT_MODIFIED``.
    """
    panel_data.objview.update_item(output_uuid)
    panel_data.refresh_plot(output_uuid, update_items=True, force=True)
    obj = (
        panel_data.objmodel[output_uuid]
        if panel_data.objmodel.has_uuid(output_uuid)
        else None
    )
    if obj is not None:
        if obj is panel_data.objview.get_current_object():
            panel_data.objprop.update_properties_from(obj, force_tab="processing")
        else:
            panel_data.objprop.mark_as_freshly_processed(obj)
    panel_data.SIG_OBJECT_MODIFIED.emit()


def resolve_output_panel(
    panel: HistoryPanel,
    out_uuid: str,
    new_obj: SignalObj | ImageObj | None,
    fallback: BaseDataPanel,
) -> BaseDataPanel:
    """Return the data panel that owns (or must own) an action output.

    Cross-panel features (e.g. an image line profile producing a signal)
    store their output in the panel matching the output type, not in the
    panel of the action. Resolution order: the panel whose object model
    currently owns ``out_uuid``, then the panel matching the type of
    ``new_obj``, then ``fallback`` (the action's panel).

    Args:
        panel: History panel instance.
        out_uuid: Recorded UUID of the output object.
        new_obj: Freshly recomputed output object, or ``None`` when the
         output has not been recomputed yet.
        fallback: Panel to return when neither the UUID nor the object
         type resolves to a panel.

    Returns:
        Data panel that owns (or must own) the output object.
    """
    # Stub panels in unit tests have no mainwindow: no cross-panel routing
    mainwindow = getattr(panel, "mainwindow", None)
    if mainwindow is None:
        return fallback
    signalpanel = mainwindow.signalpanel
    imagepanel = mainwindow.imagepanel
    if signalpanel.objmodel.has_uuid(out_uuid):
        return signalpanel
    if imagepanel.objmodel.has_uuid(out_uuid):
        return imagepanel
    if isinstance(new_obj, SignalObj):
        return signalpanel
    if isinstance(new_obj, ImageObj):
        return imagepanel
    return fallback


def apply_output_in_place_or_recreate(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    action: HistoryAction,
    out_uuid: str,
    new_obj: SignalObj | ImageObj,
    pparams: ProcessingParameters | None = None,
    group_id: str | None = None,
) -> None:
    """Update the recorded output in place, re-creating it if it was deleted.

    When the recorded output object no longer exists in ``panel_data``, the
    freshly computed ``new_obj`` is inserted back **under its original
    recorded UUID** so that downstream references (``source_uuid`` /
    ``source_uuids`` metadata and ``action.output_uuids``) remain valid
    without any remapping. The runtime action→outputs mapping (pruned when
    the object was deleted) is re-registered.

    This only commits the data: callers are responsible for calling
    :func:`refresh_target` once all outputs have been committed.

    Args:
        panel: History panel instance.
        panel_data: Data panel that owns (or owned) the output object.
        action: History action that produced the output.
        out_uuid: Recorded UUID of the output object.
        new_obj: Freshly recomputed object providing title, data and metadata.
        pparams: Processing parameters to store on the output, or ``None``.
        group_id: Group for a re-created output (``None`` = default group).
    """
    if panel_data.objmodel.has_uuid(out_uuid):
        target_obj = panel_data.objmodel[out_uuid]
        update_obj_in_place(target_obj, new_obj)
    else:
        new_obj.set_metadata_option("uuid", out_uuid)
        # The ``replaying()`` guard suppresses history capture and session
        # prompts while the deleted output is re-inserted in the data panel.
        with panel.replaying():
            panel_data.add_object(new_obj, group_id=group_id, set_current=False)
        target_obj = panel_data.objmodel[out_uuid]
        panel.runtime.objects.register_action_outputs(
            action, hchain.recorded_action_output_uuids(panel, action)
        )
    if pparams is not None:
        insert_processing_parameters(target_obj, pparams)


def recompute_mutation_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Re-apply a mutation action to its target object(s) during a cascade.

    The recorded payload is re-applied through
    :meth:`HistoryAction.replay_mutation`; targets that were deleted are
    skipped with a cascade warning.

    Args:
        panel: History panel instance.
        action: Mutation-kind history action to re-apply.

    Returns:
        True when at least one target object was mutated.
    """
    name = action.title or action.uuid
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: target panel not found — skipping.") % name
        )
        return False
    targets = action.target_uuids or []
    if not targets:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: no recorded mutation target — skipping.") % name
        )
        return False
    missing = [uuid for uuid in targets if not panel_data.objmodel.has_uuid(uuid)]
    if len(missing) == len(targets):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: target object(s) no longer exist — skipping.") % name
        )
        return False
    if missing:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: %d target object(s) were deleted — applying to the rest.")
            % (name, len(missing))
        )
    # ``replaying()`` is reentrant: suppress history capture while the
    # payload is re-applied to the data panel objects. ``refresh=False``
    # because each mutated target is refreshed individually below.
    with panel.replaying():
        mutated = action.replay_mutation(panel.mainwindow, refresh=False)
    for uuid in mutated:
        refresh_target(panel_data, uuid)
    return bool(mutated)


def recompute_action_in_place(  # pylint: disable=too-many-return-statements
    panel: HistoryPanel, action: HistoryAction
) -> bool:
    """Re-run ``action`` on the existing output object(s) (same UUIDs)."""
    if getattr(action, "decode_failed", False):
        # Broken persisted parameters: executing would silently change
        # semantics (e.g. a mutation payload degraded to None deletes ROIs).
        # Not marked stale: the action is permanently non-recomputable.
        panel.runtime.execution.cascade_warnings.append(
            _(
                "Action %s was skipped: its recorded parameters could not "
                "be read from the history file."
            )
            % (action.title or action.func_name or action.uuid)
        )
        panel.tree.refresh_action_item(action)
        return False
    if (
        action.kind == HistoryAction.KIND_UI
        and action.method_name in HistoryAction.UI_CREATION_METHODS
    ):
        return recompute_creation_in_place(panel, action)
    if action.kind == HistoryAction.KIND_MUTATION:
        return recompute_mutation_in_place(panel, action)
    if action.kind != HistoryAction.KIND_COMPUTE:
        return False
    method = {
        "1_to_1": recompute_compute_in_place,
        "multiple_1_to_1": recompute_compute_in_place,
        "1_to_n": recompute_compute_in_place,
        "n_to_1": recompute_compute_in_place,
        "2_to_1": recompute_compute_in_place,
        "1_to_0": recompute_1_to_0_in_place,
    }.get(action.pattern or "")
    if method is None:
        _logger.warning(
            "Cascade recompute: unsupported pattern %r for action %s.",
            action.pattern,
            action.uuid,
        )
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s uses pattern %r which is not recomputable yet.")
            % (action.func_name or action.uuid, action.pattern)
        )
        return False
    try:
        warning_count = len(panel.runtime.execution.cascade_warnings)
        success = method(panel, action)
        if (
            not success
            and len(panel.runtime.execution.cascade_warnings) == warning_count
        ):
            panel.runtime.execution.cascade_warnings.append(
                _("Action %s could not be fully recomputed.")
                % (action.func_name or action.uuid)
            )
        return success
    except FeatureNotFoundError as exc:
        handle_missing_feature(panel, action, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        action.is_stale = True
        _logger.exception(
            "Cascade recompute failed for action %s (%s): %s",
            action.uuid,
            action.func_name,
            exc,
        )
        panel.runtime.execution.cascade_warnings.append(
            _("Recompute failed for action %s: %s")
            % (action.func_name or action.uuid, exc)
        )
    return False


def handle_missing_feature(
    panel: HistoryPanel, action: HistoryAction, exc: FeatureNotFoundError
) -> None:
    """Flag ``action`` as broken (missing plugin) and queue a user warning."""
    action.is_stale = True
    plugin_origin = action.plugin_origin or exc.plugin_origin or {}
    directory = (plugin_origin.get("directory") if plugin_origin else None) or "?"
    param = action.kwargs.get("param")
    paramclass = exc.paramclass_name or (
        type(param).__name__ if param is not None else "—"
    )
    func_name = action.func_name or exc.func_name or action.uuid
    location = f"{directory}/plugins:{func_name}"
    _logger.warning(
        "Cascade recompute: plugin missing for action %s (%s) — %s.",
        action.uuid,
        func_name,
        location,
    )
    panel.runtime.execution.cascade_warnings.append(
        _(
            "Action %(name)s skipped: plugin '%(loc)s' is missing.\n"
            "Required parameter class: %(param)s\n"
            "Reinstall the plugin to re-enable this action."
        )
        % {"name": func_name, "loc": location, "param": paramclass}
    )


def recompute_creation_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute a creation (``new_object``) action in place.

    Rebuild the object from the edited ``param`` and copy it onto the
    existing output object so its UUID (and downstream references) are kept.
    If the output object was deleted, it is re-created under its recorded
    UUID so the downstream chain remains valid.

    Synthetic session heads (e.g. produced by *Duplicate chain*) carry no
    creation ``param``: they represent a pre-existing object. As long as
    their recorded outputs still exist, they are a no-op success; if the
    object was deleted, it cannot be re-created and a warning is queued.
    """
    name = action.title or action.uuid
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: target panel not found — skipping.") % name
        )
        return False
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: no recorded output object — skipping.") % name
        )
        return False
    output_uuid = recorded[0]
    param = action.kwargs.get("param")
    if param is None:
        # Outputs may live in either panel (cross-panel routing): resolve
        # each one before checking existence.
        if all(
            resolve_output_panel(panel, uuid, None, panel_data).objmodel.has_uuid(uuid)
            for uuid in recorded
        ):
            return True
        panel.runtime.execution.cascade_warnings.append(
            _(
                "Action %s: the initial object was deleted and cannot be "
                "re-created (no creation parameters)."
            )
            % name
        )
        return False
    if action.target == "signalpanel":
        prepared = prepare_signal_parameters(param, edit=False)
        if prepared is None:
            panel.runtime.execution.cascade_warnings.append(
                _("Action %s: creation parameters could not be prepared — skipping.")
                % name
            )
            return False
        new_obj = create_signal_from_param(prepared)
    else:
        new_obj = create_image_from_param(param)
    # Creation parameters are carried by ``new_obj`` so both the in-place
    # update (metadata copy) and the recreation path preserve them.
    insert_creation_parameters(new_obj, param)
    apply_output_in_place_or_recreate(panel, panel_data, action, output_uuid, new_obj)
    refresh_target(panel_data, output_uuid)
    return True


def execute_compute_via_ui(
    panel_data: BaseDataPanel,
    action: HistoryAction,
    obj2_uuids: list[str],
) -> None:
    """Invoke the public processor entry point matching ``action``'s pattern.

    This is the same code path the menus use (``compute_1_to_1``,
    ``compute_multiple_1_to_1``, ``compute_1_to_n``, ``compute_n_to_1``,
    ``compute_2_to_1``), so multi-selection batching, group creation,
    pairwise mode, X-array compatibility, progress bars and error handling
    are reused by construction. The recorded parameters are deep-copied so
    replay never mutates (nor shares) the action's own kwargs, and
    ``edit=False`` suppresses parameter dialogs (edit-mode prompting is
    handled upstream by the interactive replay).

    Args:
        panel_data: Data panel targeted by the action.
        action: Compute-kind history action to re-execute.
        obj2_uuids: Recorded second-operand UUIDs (2-to-1 pattern only).

    Raises:
        FeatureNotFoundError: If the feature is not registered (missing
         plugin), propagated to :func:`recompute_action_in_place`.
    """
    processor = panel_data.processor
    title = action.title or action.func_name
    if action.pattern == "multiple_1_to_1":
        func_names = action.kwargs.get("func_names") or (
            [action.func_name] if action.func_name else []
        )
        funcs = [
            processor.get_feature(
                func_name, plugin_origin=action.plugin_origin
            ).function
            for func_name in func_names
        ]
        params = action.kwargs.get("params")
        processor.compute_multiple_1_to_1(
            funcs,
            params=copy.deepcopy(params) if params is not None else None,
            title=title,
            edit=False,
        )
        return
    if action.pattern == "1_to_n":
        params = [copy.deepcopy(param) for param in action.kwargs.get("params") or []]
        feature = processor.get_feature(
            action.func_name,
            plugin_origin=action.plugin_origin,
            paramclass_name=type(params[0]).__name__ if params else None,
        )
        processor.compute_1_to_n(
            feature.function, params=params, title=title, edit=False
        )
        return
    param = copy.deepcopy(action.kwargs.get("param"))
    feature = processor.get_feature(
        action.func_name,
        plugin_origin=action.plugin_origin,
        paramclass_name=type(param).__name__ if param is not None else None,
    )
    if action.pattern == "1_to_1":
        processor.compute_1_to_1(feature.function, param=param, title=title, edit=False)
    elif action.pattern == "n_to_1":
        processor.compute_n_to_1(
            feature.function,
            param=param,
            title=title,
            edit=False,
            pairwise=bool(action.kwargs.get("pairwise")),
        )
    elif action.pattern == "2_to_1":
        objs2 = [panel_data.objmodel[uuid] for uuid in obj2_uuids]
        pairwise = bool(action.kwargs.get("pairwise"))
        processor.compute_2_to_1(
            objs2 if pairwise else objs2[0],
            action.kwargs.get("obj2_name") or feature.obj2_name or _("Second operand"),
            feature.function,
            param=param,
            title=title,
            edit=False,
            skip_xarray_compat=feature.skip_xarray_compat,
            pairwise=pairwise,
            pre_execute_hook=feature.pre_execute_hook,
        )
    else:
        raise ValueError(f"Unsupported compute pattern: {action.pattern!r}")


def _detach_object(panel_data: BaseDataPanel, obj_uuid: str) -> SignalObj | ImageObj:
    """Remove ``obj_uuid`` from ``panel_data`` and return the live instance.

    Low-level counterpart of ``BaseDataPanel.remove_object`` used for the
    temporary objects created by an execute-via-UI replay: no history entry,
    no removal signal (the object was never a real workspace output).
    """
    obj = panel_data.objmodel[obj_uuid]
    panel_data.plothandler.remove_item(obj_uuid)
    panel_data.objview.remove_item(obj_uuid, refresh=False)
    panel_data.objmodel.remove_object(obj)
    panel_data.objview.update_tree()
    return obj


def _discard_new_empty_groups(
    data_panels: tuple[BaseDataPanel, ...],
    before_groups: dict[str, set[str]],
) -> None:
    """Remove empty groups created by an execute-via-UI replay batch."""
    for panel_data in data_panels:
        removed = False
        for group in list(panel_data.objmodel.get_groups()):
            group_uuid = get_uuid(group)
            if group_uuid in before_groups[panel_data.PANEL_STR_ID]:
                continue
            if group.get_object_ids():
                continue
            panel_data.objview.remove_item(group_uuid, refresh=False)
            panel_data.objmodel.remove_group(group)
            removed = True
        if removed:
            panel_data.objview.update_tree()


def _restore_selection(
    data_panels: tuple[BaseDataPanel, ...],
    saved_selection: dict[str, list[str]],
) -> None:
    """Best-effort restore of the pre-replay object and group selection."""
    for panel_data in data_panels:
        group_uuids = {get_uuid(grp) for grp in panel_data.objmodel.get_groups()}
        uuids = [
            uuid
            for uuid in saved_selection[panel_data.PANEL_STR_ID]
            if panel_data.objmodel.has_uuid(uuid) or uuid in group_uuids
        ]
        for idx, uuid in enumerate(uuids):
            panel_data.objview.set_current_item_id(uuid, extend=idx > 0)


def _commit_outputs(
    panel: HistoryPanel,
    panel_data: BaseDataPanel,
    action: HistoryAction,
    recorded: list[str],
    detached: list[tuple[BaseDataPanel, SignalObj | ImageObj]],
) -> None:
    """Commit fresh outputs onto the recorded outputs (index-aligned).

    Recorded outputs that still exist are updated in place; deleted outputs
    are re-created under their original recorded UUIDs, preferring the group
    of a surviving sibling output, then the first source's group. For 1-to-1
    family patterns, existing outputs keep their own metadata (ROIs,
    annotations, analysis results...) and only their processing parameters
    are refreshed, matching the behavior of a manual re-processing. On
    failure, previously existing outputs are restored from snapshots before
    the exception is propagated.

    Args:
        panel: History panel instance.
        panel_data: Data panel targeted by the action.
        action: Compute-kind history action being reconciled.
        recorded: Recorded output UUIDs, in recording order.
        detached: Freshly computed objects (with their creation panel), in
         creation order, index-aligned with ``recorded``.
    """
    sources = action.state.selection.get(panel_data.PANEL_STR_ID, [])
    fallback_gid = None
    if sources and panel_data.objmodel.has_uuid(sources[0]):
        fallback_gid = panel_data.objmodel.get_object_group_id(
            panel_data.objmodel[sources[0]]
        )
    preserve_metadata = action.pattern in {"1_to_1", "multiple_1_to_1"}
    plans: list[tuple[str, BaseDataPanel, SignalObj | ImageObj]] = []
    snapshots: dict[str, tuple[BaseDataPanel, SignalObj | ImageObj]] = {}
    for out_uuid, (fresh_panel, fresh_obj) in zip(recorded, detached):
        output_panel = resolve_output_panel(panel, out_uuid, fresh_obj, fresh_panel)
        plans.append((out_uuid, output_panel, fresh_obj))
        if output_panel.objmodel.has_uuid(out_uuid):
            snapshots[out_uuid] = (
                output_panel,
                copy.deepcopy(output_panel.objmodel[out_uuid]),
            )
    sibling_gid = next(
        (
            output_panel.objmodel.get_object_group_id(output_panel.objmodel[out_uuid])
            for out_uuid, output_panel, _fresh_obj in plans
            if output_panel is panel_data and out_uuid in snapshots
        ),
        None,
    )
    try:
        for out_uuid, output_panel, fresh_obj in plans:
            pparams = extract_processing_parameters(fresh_obj)
            existing_pp = (
                extract_processing_parameters(output_panel.objmodel[out_uuid])
                if out_uuid in snapshots
                else None
            )
            if pparams is not None:
                # The freshly registered feature may not carry the plugin
                # origin (or a param): fall back to the recorded action, then
                # to the metadata stored on the target, as the legacy
                # per-pattern engines did.
                if pparams.plugin_origin is None:
                    pparams.plugin_origin = action.plugin_origin or (
                        existing_pp.plugin_origin if existing_pp else None
                    )
                if pparams.param is None and existing_pp is not None:
                    pparams.param = existing_pp.param
            if out_uuid in snapshots and preserve_metadata and pparams is not None:
                # Preserve the existing output's own metadata (ROIs,
                # annotations, analysis results...)
                output_panel.objprop.apply_recomputed_object_in_place(
                    output_panel.objmodel[out_uuid], fresh_obj, pparams
                )
                continue
            group_id = None
            if output_panel is panel_data and out_uuid not in snapshots:
                group_id = sibling_gid or fallback_gid
            apply_output_in_place_or_recreate(
                panel,
                output_panel,
                action,
                out_uuid,
                fresh_obj,
                pparams,
                group_id=group_id,
            )
        for out_uuid, output_panel, _fresh_obj in plans:
            refresh_target(output_panel, out_uuid)
    except Exception:
        # Outputs re-created during this failed batch (absent from
        # ``snapshots``) must not survive the rollback: detach them.
        for out_uuid, output_panel, _fresh_obj in plans:
            if out_uuid not in snapshots and output_panel.objmodel.has_uuid(out_uuid):
                _detach_object(output_panel, out_uuid)
        for out_uuid, (output_panel, snapshot) in snapshots.items():
            update_obj_in_place(output_panel.objmodel[out_uuid], snapshot)
        for out_uuid, (output_panel, _snapshot) in snapshots.items():
            try:
                refresh_target(output_panel, out_uuid)
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.exception(
                    "Cascade recompute rollback refresh failed for output %s.",
                    out_uuid,
                )
        raise


def recompute_compute_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:  # pylint: disable=too-many-return-statements
    """Replay a compute action through the real UI entry point, then reconcile.

    **Execute**: the recorded selection is restored and the same public
    processor method the menus use is invoked with the recorded parameters
    (see :func:`execute_compute_via_ui`), under the ``replaying()`` guard so
    no new history entry is recorded.

    **Reconcile**: the freshly created objects — diffed from both data
    panels, in recording order (cross-panel outputs land in the panel
    matching their type) — are aligned by index with the recorded output
    UUIDs and committed via :func:`_commit_outputs`. The temporary objects
    (and any temporary group created by the batch, e.g. pairwise ``dst_gname``
    groups) are then removed so no duplicates remain.

    If the number of fresh outputs differs from the number of recorded
    outputs, the temporary objects are discarded, a warning is queued and the
    action is flagged stale.

    Args:
        panel: History panel instance.
        action: Compute-kind history action to recompute.

    Returns:
        True when every recorded output was reconciled.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded:
        return False
    name = action.func_name or action.title or action.uuid
    sources = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    if not sources:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: no recorded source object — skipping.") % name
        )
        return False
    obj2_uuids = action.kwargs.get("obj2_uuids") or []
    if isinstance(obj2_uuids, str):
        obj2_uuids = [obj2_uuids]
    if action.pattern == "2_to_1" and not obj2_uuids:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: no recorded second operand — skipping.") % name
        )
        return False
    required = sources + (obj2_uuids if action.pattern == "2_to_1" else [])
    if any(not panel_data.objmodel.has_uuid(uuid) for uuid in required):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object(s) were deleted — skipping.") % name
        )
        return False
    data_panels = (panel.mainwindow.signalpanel, panel.mainwindow.imagepanel)
    before_objs = {
        p.PANEL_STR_ID: set(p.objmodel.get_object_ids()) for p in data_panels
    }
    before_grps = {
        p.PANEL_STR_ID: {get_uuid(grp) for grp in p.objmodel.get_groups()}
        for p in data_panels
    }
    saved_selection = {
        p.PANEL_STR_ID: p.objview.get_sel_object_uuids()
        + p.objview.get_sel_group_uuids()
        for p in data_panels
    }
    # ``replaying()`` suppresses history capture and session prompts for the
    # whole execute + reconcile scope (temporary insertions included).
    with panel.replaying():
        try:
            panel_data.objview.select_objects(sources)
            try:
                execute_compute_via_ui(panel_data, action, obj2_uuids)
            except Exception:
                # A compute failing mid-batch may already have inserted some
                # fresh temporaries: detach them so no duplicates remain.
                for p in data_panels:
                    for uid in list(p.objmodel.get_object_ids()):
                        if uid not in before_objs[p.PANEL_STR_ID]:
                            _detach_object(p, uid)
                _discard_new_empty_groups(data_panels, before_grps)
                raise
            fresh = [
                (p, uid)
                for p in data_panels
                for uid in p.objmodel.get_object_ids()
                if uid not in before_objs[p.PANEL_STR_ID]
            ]
            if len(fresh) != len(recorded):
                for fresh_panel, fresh_uuid in fresh:
                    _detach_object(fresh_panel, fresh_uuid)
                _discard_new_empty_groups(data_panels, before_grps)
                action.is_stale = True
                _logger.warning(
                    "Cascade recompute: cardinality changed for action %s: "
                    "%d output(s), %d recorded.",
                    action.uuid,
                    len(fresh),
                    len(recorded),
                )
                panel.runtime.execution.cascade_warnings.append(
                    _("Action %s: recompute returned %d output(s), expected %d.")
                    % (name, len(fresh), len(recorded))
                )
                return False
            # Detach the fresh objects: they are temporary carriers whose
            # content is committed onto the recorded outputs below.
            detached = [
                (fresh_panel, _detach_object(fresh_panel, fresh_uuid))
                for fresh_panel, fresh_uuid in fresh
            ]
            _discard_new_empty_groups(data_panels, before_grps)
            _commit_outputs(panel, panel_data, action, recorded, detached)
        finally:
            _restore_selection(data_panels, saved_selection)
    return True


def _snapshot_analysis_source(
    obj: SignalObj | ImageObj, effects_dict: dict | None
) -> tuple[dict[str, Any], list[str] | None]:
    """Snapshot the metadata of one analysis source before a recompute.

    When an effects manifest is available, only the keys it lists are deep
    copied (targeted snapshot) and the manifest keys currently absent are
    recorded so a failed attempt that recreates them can be rolled back by
    deletion. Without a manifest (legacy action), the whole metadata
    dictionary is deep copied.

    Args:
        obj: Source object about to be recomputed.
        effects_dict: Serialized :class:`AnalysisEffects` manifest, or None.

    Returns:
        Tuple ``(saved, absent)`` where ``saved`` maps keys to deep-copied
        values and ``absent`` lists manifest keys missing before the
        recompute. ``absent`` is None for the legacy full-metadata snapshot.
    """
    if effects_dict is None:
        return copy.deepcopy(obj.metadata), None
    manifest = AnalysisEffects.from_dict(effects_dict)
    keys = manifest.metadata_added + manifest.metadata_replaced
    saved = {
        key: copy.deepcopy(obj.metadata[key]) for key in keys if key in obj.metadata
    }
    absent = [key for key in keys if key not in obj.metadata]
    return saved, absent


def _restore_analysis_source(
    obj: SignalObj | ImageObj,
    saved: dict[str, Any],
    absent: list[str] | None,
    attempt_effects: AnalysisEffects | None,
) -> None:
    """Restore a source's metadata from its snapshot after a failed recompute.

    Args:
        obj: Source object to restore.
        saved: Snapshotted metadata values (full metadata for legacy actions).
        absent: Manifest keys absent before the recompute (delete them if the
         failed attempt recreated them), or None for a legacy full restore.
        attempt_effects: Effects captured during the failed attempt, used to
         delete keys it created outside the manifest (targeted mode only).
    """
    if absent is None:
        obj.metadata.clear()
        obj.metadata.update(saved)
        # Drop any ROI created during the failed recompute so the cache stays
        # consistent with the restored metadata
        obj.invalidate_roi_cache()
        return
    touched = set(saved) | set(absent)
    if attempt_effects is not None:
        for key in attempt_effects.metadata_added:
            obj.metadata.pop(key, None)
        touched.update(attempt_effects.metadata_added)
    obj.metadata.update(saved)
    for key in absent:
        obj.metadata.pop(key, None)
    if ROI_KEY in touched and hasattr(obj, "invalidate_roi_cache"):
        # Align with the legacy full-restore path: a restored/removed ROI
        # entry must not leave a stale cached ROI object behind.
        obj.invalidate_roi_cache()


def recompute_1_to_0_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute a 1-to-0 analysis on each source object in place.

    Sources are snapshotted before the recompute. When the action carries an
    effects manifest, the snapshot is targeted: only manifest keys are deep
    copied and a failed attempt rolls back exactly those keys (plus any key
    the attempt created), leaving unrelated metadata untouched. Legacy
    actions without a manifest fall back to a full-metadata snapshot.
    Sources whose manifest records a pre-analysis ROI (``roi_before``) are
    restored to it and re-run with first-run semantics, regenerating
    detection ROIs; user ROI edits recorded as later mutation actions are
    then re-applied by the replay/cascade sequence.
    On success, the freshly captured effects are merged into the manifest.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    sources = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    if not sources:
        return False
    param = copy.deepcopy(action.kwargs.get("param"))
    missing = [uuid for uuid in sources if not panel_data.objmodel.has_uuid(uuid)]
    if missing:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: %d analysed object(s) were deleted — skipping.")
            % (action.func_name or action.uuid, len(missing))
        )
        return False
    source_objs = [panel_data.objmodel[uuid] for uuid in sources]
    snapshots = [
        _snapshot_analysis_source(obj, (action.effects or {}).get(uuid))
        for uuid, obj in zip(sources, source_objs)
    ]
    captured: dict[str, AnalysisEffects] = {}
    roi_snapshots: dict[str, Any] = {}

    def rollback() -> None:
        for uuid, obj, (saved, absent) in zip(sources, source_objs, snapshots):
            _restore_analysis_source(obj, saved, absent, captured.get(uuid))
        # Undo the pre-analysis ROI restoration after the metadata restore so
        # the ROI setter leaves both metadata and cache consistent.
        for uuid, obj in zip(sources, source_objs):
            if uuid in roi_snapshots:
                obj.roi = roi_snapshots[uuid]

    try:
        for uuid, src_obj in zip(sources, source_objs):
            analysis_parameters = extract_analysis_parameters(src_obj)
            plugin_origin = action.plugin_origin or (
                analysis_parameters.plugin_origin if analysis_parameters else None
            )
            # Restore the recorded pre-analysis ROI so the detection re-runs
            # on the same region as the first run, with ROI creation enabled
            # ("" encodes "no ROI before", None means legacy/not recorded).
            roi_before = AnalysisEffects.from_dict(
                (action.effects or {}).get(uuid) or {}
            ).roi_before
            if roi_before is not None:
                roi_snapshots[uuid] = (
                    src_obj.roi.copy() if src_obj.roi is not None else None
                )
                src_obj.roi = decode_roi(roi_before) if roi_before else None
            with capture_effects(src_obj) as effects:
                # Register the (mutable) effects before running so rollback
                # sees them even when the recompute raises
                captured[uuid] = effects
                success = panel_data.processor.recompute_1_to_0(
                    action.func_name,
                    src_obj,
                    param,
                    plugin_origin=plugin_origin,
                    first_run_side_effects=roi_before is not None,
                )
            if not success:
                rollback()
                return False
    except Exception:
        rollback()
        raise
    if action.effects is None:
        action.effects = {}
    for uuid in sources:
        prev_dict = action.effects.get(uuid)
        previous = AnalysisEffects.from_dict(prev_dict) if prev_dict else None
        action.effects[uuid] = merge_effects(previous, captured[uuid]).to_dict()
    for uuid in sources:
        refresh_target(panel_data, uuid)
    return True


def recompute_cascade(
    panel: HistoryPanel,
    root_action: HistoryAction,
    descendants: list[HistoryAction] | None = None,
) -> None:
    """Recompute ``root_action``'s descendants in the current session in place."""
    if descendants is None:
        descendants = hchain.get_downstream_actions(panel, root_action)
    if root_action.is_stale:
        descendants = [root_action] + descendants
    if panel.runtime.execution.cascade_in_progress:
        flush_cascade_warnings(panel)
        return
    if not descendants:
        flush_cascade_warnings(panel)
        return
    try:
        with panel.runtime.execution.recomputing_cascade():
            panel.ui.update_actions_state()
            for action in descendants:
                action.is_stale = True
                panel.tree.refresh_action_item(action)
            QW.QApplication.processEvents()
            for action in descendants:
                success = recompute_action_in_place(panel, action)
                if success:
                    action.is_stale = False
                panel.tree.refresh_action_item(action)
                QW.QApplication.processEvents()
                if not success:
                    break
    finally:
        panel.ui.update_actions_state()
    flush_cascade_warnings(panel)


def flush_cascade_warnings(panel: HistoryPanel) -> None:
    """Show + clear accumulated cascade warnings (no-op when empty)."""
    if panel.runtime.execution.cascade_warnings and not execenv.unattended:
        QW.QMessageBox.warning(
            panel.mainwindow,
            _("Cascade recompute"),
            _("Some downstream actions could not be recomputed:")
            + "\n\n• "
            + "\n• ".join(panel.runtime.execution.cascade_warnings),
        )
    panel.runtime.execution.cascade_warnings.clear()
