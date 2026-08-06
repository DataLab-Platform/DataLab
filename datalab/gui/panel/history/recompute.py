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


def record_missing_outputs(
    panel: HistoryPanel, action: HistoryAction, missing: list[str]
) -> None:
    """Log + queue a user-facing warning for deleted output objects.

    Only used for genuinely irrecoverable cases (e.g. recorded outputs that
    cannot be aligned with the action's parameters); recoverable deletions
    are handled by :func:`apply_output_in_place_or_recreate`.
    """
    if not missing:
        return
    name = action.func_name or action.title or action.uuid
    _logger.warning(
        "Cascade recompute: %d output(s) missing for action %s (%s).",
        len(missing),
        action.uuid,
        name,
    )
    panel.runtime.execution.cascade_warnings.append(
        _(
            "Action %s has been edited but its target output object(s) "
            "no longer exist — skipping."
        )
        % name
    )


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


def recompute_action_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Re-run ``action`` on the existing output object(s) (same UUIDs)."""
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
        "1_to_1": recompute_1_to_1_in_place,
        "1_to_n": recompute_1_to_n_in_place,
        "n_to_1": recompute_n_to_1_in_place,
        "2_to_1": recompute_2_to_1_in_place,
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


def recompute_1_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute a single 1-to-1 action in place.

    If the output object was deleted, it is re-created under its recorded
    UUID; the source is then resolved from the action's captured state.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded:
        return False
    output_uuid = recorded[0]
    output_panel = resolve_output_panel(panel, output_uuid, None, panel_data)
    output_obj = (
        output_panel.objmodel[output_uuid]
        if output_panel.objmodel.has_uuid(output_uuid)
        else None
    )
    pp = extract_processing_parameters(output_obj) if output_obj is not None else None
    source_uuid = pp.source_uuid if pp is not None and pp.source_uuid else None
    if source_uuid is None:
        selection = action.state.selection.get(panel_data.PANEL_STR_ID, [])
        source_uuid = selection[0] if selection else None
    if source_uuid is None:
        return False
    if not panel_data.objmodel.has_uuid(source_uuid):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object was deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return False
    source_obj = panel_data.objmodel[source_uuid]
    param = action.kwargs.get("param")
    plugin_origin = action.plugin_origin or (pp.plugin_origin if pp else None)
    compout = panel_data.processor.recompute_1_to_1(
        action.func_name,
        source_obj,
        param,
        plugin_origin=plugin_origin,
    )
    if compout.cancelled:
        return False
    if compout.error_msg:
        panel.runtime.execution.cascade_warnings.append(
            _("Recompute failed for action %s: %s")
            % (action.func_name or action.uuid, compout.error_msg)
        )
        return False
    new_obj = compout.result
    if not isinstance(new_obj, (SignalObj, ImageObj)):
        return False
    pp_new = ProcessingParameters(
        func_name=pp.func_name if pp is not None else action.func_name,
        pattern=pp.pattern if pp is not None else "1-to-1",
        param=param if param is not None else (pp.param if pp is not None else None),
        source_uuid=source_uuid,
        plugin_origin=plugin_origin,
    )
    output_panel = resolve_output_panel(panel, output_uuid, new_obj, panel_data)
    if output_obj is not None:
        # Preserve the existing output's own metadata (ROIs, annotations...)
        output_panel.objprop.apply_recomputed_object_in_place(
            output_obj, new_obj, pp_new
        )
    else:
        apply_output_in_place_or_recreate(
            panel,
            output_panel,
            action,
            output_uuid,
            new_obj,
            pp_new,
            group_id=(
                panel_data.objmodel.get_object_group_id(source_obj)
                if output_panel is panel_data
                else None
            ),
        )
    refresh_target(output_panel, output_uuid)
    return True


def recompute_1_to_n_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute a 1-to-n action in place: replace each of the N outputs.

    Deleted outputs are re-created under their recorded UUIDs; indices of
    ``action.output_uuids`` align with the recorded ``params`` list.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    params = action.kwargs.get("params") or []
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded or not params:
        return False
    # Resolve against the first recorded output still owned by a panel, so a
    # deleted recorded[0] does not fall back to the wrong (action) panel.
    resolve_uuid = next(
        (
            u
            for u in recorded
            if resolve_output_panel(panel, u, None, panel_data).objmodel.has_uuid(u)
        ),
        recorded[0],
    )
    output_panel = resolve_output_panel(panel, resolve_uuid, None, panel_data)
    if len(recorded) != len(params):
        # Legacy or inconsistent recording: outputs cannot be aligned with
        # the parameter list, so recreation is impossible.
        record_missing_outputs(
            panel,
            action,
            [u for u in recorded if not output_panel.objmodel.has_uuid(u)],
        )
        return False
    existing = [u for u in recorded if output_panel.objmodel.has_uuid(u)]
    pp = (
        extract_processing_parameters(output_panel.objmodel[existing[0]])
        if existing
        else None
    )
    source_uuid = pp.source_uuid if pp is not None and pp.source_uuid else None
    if source_uuid is None:
        selection = action.state.selection.get(panel_data.PANEL_STR_ID, [])
        source_uuid = selection[0] if selection else None
    if source_uuid is None:
        return False
    if not panel_data.objmodel.has_uuid(source_uuid):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object was deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return False
    source_obj = panel_data.objmodel[source_uuid]
    plugin_origin = action.plugin_origin or (pp.plugin_origin if pp else None)
    new_objs = panel_data.processor.recompute_1_to_n(
        action.func_name,
        source_obj,
        params,
        plugin_origin=plugin_origin,
    )
    if not new_objs:
        return False
    if len(new_objs) != len(recorded) or not all(
        isinstance(obj, (SignalObj, ImageObj)) for obj in new_objs
    ):
        _logger.warning(
            "1-to-n cardinality changed for action %s: %d outputs, %d recorded.",
            action.uuid,
            len(new_objs),
            len(recorded),
        )
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: recompute returned %d output(s), expected %d.")
            % (action.func_name or action.uuid, len(new_objs), len(recorded))
        )
        return False
    output_panel = resolve_output_panel(panel, recorded[0], new_objs[0], panel_data)
    group_id = (
        panel_data.objmodel.get_object_group_id(source_obj)
        if output_panel is panel_data
        else None
    )
    snapshots = {
        out_uuid: copy.deepcopy(output_panel.objmodel[out_uuid])
        for out_uuid in existing
        if output_panel.objmodel.has_uuid(out_uuid)
    }
    try:
        for idx, out_uuid in enumerate(recorded):
            apply_output_in_place_or_recreate(
                panel,
                output_panel,
                action,
                out_uuid,
                new_objs[idx],
                ProcessingParameters(
                    func_name=action.func_name,
                    pattern="1-to-n",
                    param=params[idx],
                    source_uuid=source_uuid,
                    plugin_origin=plugin_origin,
                ),
                group_id=group_id,
            )
        for out_uuid in recorded:
            refresh_target(output_panel, out_uuid)
    except Exception:
        for out_uuid, snapshot in snapshots.items():
            update_obj_in_place(output_panel.objmodel[out_uuid], snapshot)
        for out_uuid in snapshots:
            try:
                refresh_target(output_panel, out_uuid)
            except Exception:
                _logger.exception(
                    "Cascade recompute rollback refresh failed for output %s.",
                    out_uuid,
                )
        raise
    return True


def recompute_n_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute an n-to-1 action in place.

    If the output object was deleted, it is re-created under its recorded
    UUID; sources are then resolved from the action's captured state.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded:
        return False
    output_uuid = recorded[0]
    output_panel = resolve_output_panel(panel, output_uuid, None, panel_data)
    output_obj = (
        output_panel.objmodel[output_uuid]
        if output_panel.objmodel.has_uuid(output_uuid)
        else None
    )
    pp = extract_processing_parameters(output_obj) if output_obj is not None else None
    source_uuids: list[str] = []
    if pp is not None and pp.source_uuids:
        source_uuids = list(pp.source_uuids)
    else:
        source_uuids = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    if not source_uuids or not all(
        panel_data.objmodel.has_uuid(uuid) for uuid in source_uuids
    ):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object(s) were deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return False
    src_objs = [panel_data.objmodel[uuid] for uuid in source_uuids]
    param = action.kwargs.get("param")
    plugin_origin = action.plugin_origin or (pp.plugin_origin if pp else None)
    new_obj = panel_data.processor.recompute_n_to_1(
        action.func_name,
        src_objs,
        param,
        plugin_origin=plugin_origin,
    )
    if not isinstance(new_obj, (SignalObj, ImageObj)):
        return False
    output_panel = resolve_output_panel(panel, output_uuid, new_obj, panel_data)
    apply_output_in_place_or_recreate(
        panel,
        output_panel,
        action,
        output_uuid,
        new_obj,
        ProcessingParameters(
            func_name=action.func_name,
            pattern="n-to-1",
            param=param,
            source_uuids=[get_uuid(o) for o in src_objs],
            plugin_origin=plugin_origin,
        ),
        group_id=(
            panel_data.objmodel.get_object_group_id(src_objs[0])
            if output_panel is panel_data
            else None
        ),
    )
    refresh_target(output_panel, output_uuid)
    return True


def recompute_2_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> bool:
    """Recompute a 2-to-1 action in place (single or pairwise).

    Deleted outputs are re-created under their recorded UUIDs; sources are
    then resolved from the action's captured inputs (indices of
    ``action.output_uuids`` align with the recorded input/operand lists).
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return False
    recorded = hchain.recorded_action_output_uuids(panel, action)
    if not recorded:
        return False
    output_panel = resolve_output_panel(panel, recorded[0], None, panel_data)
    param = action.kwargs.get("param")
    obj2_uuids = action.kwargs.get("obj2_uuids") or []
    if isinstance(obj2_uuids, str):
        obj2_uuids = [obj2_uuids]
    pairwise = bool(action.kwargs.get("pairwise"))
    recorded_inputs = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    resolved: list[
        tuple[
            str,
            SignalObj | ImageObj,
            SignalObj | ImageObj,
            dict | None,
        ]
    ] = []
    for idx, out_uuid in enumerate(recorded):
        pp = (
            extract_processing_parameters(output_panel.objmodel[out_uuid])
            if output_panel.objmodel.has_uuid(out_uuid)
            else None
        )
        src_uuids = (
            list(pp.source_uuids)
            if pp is not None and pp.source_uuids
            else (
                recorded_inputs[idx : idx + 1] + obj2_uuids[idx : idx + 1]
                if pairwise
                else recorded_inputs[idx : idx + 1] + obj2_uuids[:1]
            )
        )
        if len(src_uuids) < 2:
            panel.runtime.execution.cascade_warnings.append(
                _("Action %s: missing source(s) for output #%d — skipping.")
                % (action.func_name or action.uuid, idx + 1)
            )
            return False
        if not (
            panel_data.objmodel.has_uuid(src_uuids[0])
            and panel_data.objmodel.has_uuid(src_uuids[1])
        ):
            panel.runtime.execution.cascade_warnings.append(
                _("Action %s: source object(s) were deleted — skipping.")
                % (action.func_name or action.uuid)
            )
            return False
        obj1 = panel_data.objmodel[src_uuids[0]]
        obj2 = panel_data.objmodel[src_uuids[1]]
        plugin_origin = action.plugin_origin or (pp.plugin_origin if pp else None)
        resolved.append((out_uuid, obj1, obj2, plugin_origin))

    paramclass_name = type(param).__name__ if param is not None else None
    feature = panel_data.processor.get_feature(
        action.func_name,
        plugin_origin=resolved[0][3],
        paramclass_name=paramclass_name,
    )
    preparation = panel_data.processor.prepare_2_to_1_pairs(
        [(obj1, obj2) for _out_uuid, obj1, obj2, _origin in resolved],
        feature.skip_xarray_compat,
        feature.pre_execute_hook,
    )
    if preparation is None:
        return False
    prepared_pairs, source_transaction = preparation
    staged: list[
        tuple[
            str,
            SignalObj | ImageObj,
            SignalObj | ImageObj,
            SignalObj | ImageObj,
            dict | None,
        ]
    ] = []
    for resolved_item, prepared_pair in zip(resolved, prepared_pairs):
        out_uuid, obj1, obj2, plugin_origin = resolved_item
        new_obj = panel_data.processor.recompute_2_to_1(
            action.func_name,
            obj1,
            obj2,
            param,
            plugin_origin=plugin_origin,
            prepared_pair=prepared_pair,
        )
        if not isinstance(new_obj, (SignalObj, ImageObj)):
            return False
        staged.append((out_uuid, new_obj, obj1, obj2, plugin_origin))
    output_panel = resolve_output_panel(panel, recorded[0], staged[0][1], panel_data)
    snapshots = {
        out_uuid: copy.deepcopy(output_panel.objmodel[out_uuid])
        for out_uuid, *_rest in staged
        if output_panel.objmodel.has_uuid(out_uuid)
    }
    try:
        for out_uuid, new_obj, obj1, obj2, plugin_origin in staged:
            apply_output_in_place_or_recreate(
                panel,
                output_panel,
                action,
                out_uuid,
                new_obj,
                ProcessingParameters(
                    func_name=action.func_name,
                    pattern="2-to-1",
                    param=param,
                    source_uuids=[get_uuid(obj1), get_uuid(obj2)],
                    plugin_origin=plugin_origin,
                ),
                group_id=(
                    panel_data.objmodel.get_object_group_id(obj1)
                    if output_panel is panel_data
                    else None
                ),
            )
        for out_uuid, *_rest in staged:
            refresh_target(output_panel, out_uuid)
        if source_transaction is not None:
            for _out_uuid, _new_obj, obj1, _obj2, _origin in staged:
                source_transaction.commit(obj1)
    except Exception:
        for out_uuid, snapshot in snapshots.items():
            update_obj_in_place(output_panel.objmodel[out_uuid], snapshot)
        for out_uuid in snapshots:
            try:
                refresh_target(output_panel, out_uuid)
            except Exception:
                _logger.exception(
                    "Cascade recompute rollback refresh failed for output %s.",
                    out_uuid,
                )
        raise
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

    def rollback() -> None:
        for uuid, obj, (saved, absent) in zip(sources, source_objs, snapshots):
            _restore_analysis_source(obj, saved, absent, captured.get(uuid))

    try:
        for uuid, src_obj in zip(sources, source_objs):
            analysis_parameters = extract_analysis_parameters(src_obj)
            plugin_origin = action.plugin_origin or (
                analysis_parameters.plugin_origin if analysis_parameters else None
            )
            with capture_effects(src_obj) as effects:
                # Register the (mutable) effects before running so rollback
                # sees them even when the recompute raises
                captured[uuid] = effects
                success = panel_data.processor.recompute_1_to_0(
                    action.func_name,
                    src_obj,
                    param,
                    plugin_origin=plugin_origin,
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
    with panel.runtime.execution.recomputing_cascade():
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
