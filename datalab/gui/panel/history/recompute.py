# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""In-place recompute helpers for the History panel cascade."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj

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
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.history import HistoryAction
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
    try:
        saved_uuid = target_obj.metadata.get("__uuid")
        saved_number = target_obj.metadata.get("__number")
        target_obj.metadata.clear()
        target_obj.metadata.update(new_obj.metadata)
        if saved_uuid is not None:
            target_obj.metadata["__uuid"] = saved_uuid
        if saved_number is not None:
            target_obj.metadata["__number"] = saved_number
    except AttributeError:
        pass


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


def record_missing_outputs(
    panel: HistoryPanel, action: HistoryAction, missing: list[str]
) -> None:
    """Log + queue a user-facing warning for deleted output objects."""
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


def recompute_action_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Re-run ``action`` on the existing output object(s) (same UUIDs)."""
    if (
        action.kind == HistoryAction.KIND_UI
        and action.method_name in HistoryAction.UI_CREATION_METHODS
    ):
        recompute_creation_in_place(panel, action)
        return
    if action.kind != HistoryAction.KIND_COMPUTE:
        return
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
        return
    try:
        method(panel, action)
    except FeatureNotFoundError as exc:
        handle_missing_feature(panel, action, exc)
    except (RuntimeError, ValueError, AttributeError, KeyError, TypeError) as exc:
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


def handle_missing_feature(
    panel: HistoryPanel, action: HistoryAction, exc: FeatureNotFoundError
) -> None:
    """Flag ``action`` as broken (missing plugin) and queue a user warning."""
    action.is_stale = True
    panel.runtime.execution.broken_actions.add(action.uuid)
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


def recompute_creation_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute a creation (``new_object``) action in place.

    Rebuild the object from the edited ``param`` and copy it onto the
    existing output object so its UUID (and downstream references) are kept.
    """
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    existing, missing = hchain.resolve_target_outputs(panel, panel_data, action)
    record_missing_outputs(panel, action, missing)
    if not existing:
        return
    output_uuid = existing[0]
    if not panel_data.objmodel.has_uuid(output_uuid):
        return
    output_obj = panel_data.objmodel[output_uuid]
    param = action.kwargs.get("param")
    if param is None:
        return
    if action.target == "signalpanel":
        prepared = prepare_signal_parameters(param, edit=False)
        if prepared is None:
            return
        new_obj = create_signal_from_param(prepared)
    else:
        new_obj = create_image_from_param(param)
    update_obj_in_place(output_obj, new_obj)
    insert_creation_parameters(output_obj, param)
    refresh_target(panel_data, output_uuid)


def recompute_1_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute a single 1-to-1 action in place."""
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    existing, missing = hchain.resolve_target_outputs(panel, panel_data, action)
    if not existing and not missing:
        legacy = hchain.find_output_object_uuid(panel, panel_data, action)
        if legacy is not None:
            existing = [legacy]
    record_missing_outputs(panel, action, missing)
    if not existing:
        return
    output_uuid = existing[0]
    if not panel_data.objmodel.has_uuid(output_uuid):
        return
    output_obj = panel_data.objmodel[output_uuid]
    pp = extract_processing_parameters(output_obj)
    if pp is None or pp.source_uuid is None:
        return
    if not panel_data.objmodel.has_uuid(pp.source_uuid):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object was deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return
    source_obj = panel_data.objmodel[pp.source_uuid]
    param = action.kwargs.get("param")
    compout = panel_data.processor.recompute_1_to_1(
        action.func_name,
        source_obj,
        param,
        plugin_origin=action.plugin_origin,
    )
    if compout.cancelled:
        return
    if compout.error_msg:
        panel.runtime.execution.cascade_warnings.append(
            _("Recompute failed for action %s: %s")
            % (action.func_name or action.uuid, compout.error_msg)
        )
        return
    new_obj = compout.result
    if not isinstance(new_obj, (SignalObj, ImageObj)):
        return
    panel_data.objprop.apply_recomputed_object_in_place(
        output_obj,
        new_obj,
        ProcessingParameters(
            func_name=pp.func_name,
            pattern=pp.pattern,
            param=param if param is not None else pp.param,
            source_uuid=pp.source_uuid,
        ),
    )
    refresh_target(panel_data, output_uuid)


def recompute_1_to_n_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute a 1-to-n action in place: replace each of the N outputs."""
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    existing, missing = hchain.resolve_target_outputs(panel, panel_data, action)
    record_missing_outputs(panel, action, missing)
    if not existing or not panel_data.objmodel.has_uuid(existing[0]):
        return
    first_obj = panel_data.objmodel[existing[0]]
    pp = extract_processing_parameters(first_obj)
    if pp is None or pp.source_uuid is None:
        return
    if not panel_data.objmodel.has_uuid(pp.source_uuid):
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: source object was deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return
    source_obj = panel_data.objmodel[pp.source_uuid]
    params = action.kwargs.get("params") or []
    if not params:
        return
    new_objs = panel_data.processor.recompute_1_to_n(
        action.func_name,
        source_obj,
        params,
        plugin_origin=action.plugin_origin,
    )
    if not new_objs:
        return
    n = min(len(existing), len(new_objs))
    for idx in range(n):
        out_uuid = existing[idx]
        if not panel_data.objmodel.has_uuid(out_uuid):
            continue
        out_obj = panel_data.objmodel[out_uuid]
        new_obj = new_objs[idx]
        update_obj_in_place(out_obj, new_obj)
        new_param = params[idx] if idx < len(params) else None
        insert_processing_parameters(
            out_obj,
            ProcessingParameters(
                func_name=action.func_name,
                pattern="1-to-n",
                param=new_param,
                source_uuid=pp.source_uuid,
            ),
        )
        refresh_target(panel_data, out_uuid)
    if len(new_objs) != len(existing):
        _logger.warning(
            "1-to-n cardinality changed for action %s: %d outputs, %d existing.",
            action.uuid,
            len(new_objs),
            len(existing),
        )


def recompute_n_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute an n-to-1 action in place."""
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    existing, missing = hchain.resolve_target_outputs(panel, panel_data, action)
    record_missing_outputs(panel, action, missing)
    if not existing:
        return
    output_uuid = existing[0]
    if not panel_data.objmodel.has_uuid(output_uuid):
        return
    output_obj = panel_data.objmodel[output_uuid]
    pp = extract_processing_parameters(output_obj)
    source_uuids: list[str] = []
    if pp is not None and pp.source_uuids:
        source_uuids = list(pp.source_uuids)
    else:
        source_uuids = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    src_objs: list[SignalObj | ImageObj] = []
    for uuid in source_uuids:
        if panel_data.objmodel.has_uuid(uuid):
            src_objs.append(panel_data.objmodel[uuid])
    if not src_objs:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: all source objects were deleted — skipping.")
            % (action.func_name or action.uuid)
        )
        return
    param = action.kwargs.get("param")
    new_obj = panel_data.processor.recompute_n_to_1(
        action.func_name,
        src_objs,
        param,
        plugin_origin=action.plugin_origin,
    )
    if new_obj is None:
        return
    update_obj_in_place(output_obj, new_obj)
    insert_processing_parameters(
        output_obj,
        ProcessingParameters(
            func_name=action.func_name,
            pattern="n-to-1",
            param=param,
            source_uuids=[get_uuid(o) for o in src_objs],
        ),
    )
    refresh_target(panel_data, output_uuid)


def recompute_2_to_1_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute a 2-to-1 action in place (single or pairwise)."""
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    existing, missing = hchain.resolve_target_outputs(panel, panel_data, action)
    record_missing_outputs(panel, action, missing)
    if not existing:
        return
    param = action.kwargs.get("param")
    obj2_uuids = action.kwargs.get("obj2_uuids") or []
    if isinstance(obj2_uuids, str):
        obj2_uuids = [obj2_uuids]
    pairwise = bool(action.kwargs.get("pairwise"))
    recorded_inputs = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    for idx, out_uuid in enumerate(existing):
        if not panel_data.objmodel.has_uuid(out_uuid):
            continue
        output_obj = panel_data.objmodel[out_uuid]
        pp = extract_processing_parameters(output_obj)
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
            continue
        if not (
            panel_data.objmodel.has_uuid(src_uuids[0])
            and panel_data.objmodel.has_uuid(src_uuids[1])
        ):
            panel.runtime.execution.cascade_warnings.append(
                _("Action %s: source object(s) were deleted — skipping.")
                % (action.func_name or action.uuid)
            )
            continue
        obj1 = panel_data.objmodel[src_uuids[0]]
        obj2 = panel_data.objmodel[src_uuids[1]]
        new_obj = panel_data.processor.recompute_2_to_1(
            action.func_name,
            obj1,
            obj2,
            param,
            plugin_origin=action.plugin_origin,
        )
        if new_obj is None:
            continue
        update_obj_in_place(output_obj, new_obj)
        insert_processing_parameters(
            output_obj,
            ProcessingParameters(
                func_name=action.func_name,
                pattern="2-to-1",
                param=param,
                source_uuids=[get_uuid(obj1), get_uuid(obj2)],
            ),
        )
        refresh_target(panel_data, out_uuid)


def recompute_1_to_0_in_place(panel: HistoryPanel, action: HistoryAction) -> None:
    """Recompute a 1-to-0 analysis on each source object in place."""
    panel_data = hchain.resolve_panel_for_action(panel, action)
    if panel_data is None:
        return
    sources = list(action.state.selection.get(panel_data.PANEL_STR_ID, []))
    if not sources:
        return
    param = action.kwargs.get("param")
    missing: list[str] = []
    for uuid in sources:
        if not panel_data.objmodel.has_uuid(uuid):
            missing.append(uuid)
            continue
        src_obj = panel_data.objmodel[uuid]
        panel_data.processor.recompute_1_to_0(
            action.func_name,
            src_obj,
            param,
            plugin_origin=action.plugin_origin,
        )
        refresh_target(panel_data, uuid)
    if missing:
        panel.runtime.execution.cascade_warnings.append(
            _("Action %s: %d analysed object(s) were deleted — skipping.")
            % (action.func_name or action.uuid, len(missing))
        )


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
    with panel.runtime.execution.recomputing_cascade() as started:
        if not started:
            flush_cascade_warnings(panel)
            return
        for action in descendants:
            action.is_stale = True
            panel.tree.refresh_action_item(action)
        QW.QApplication.processEvents()
        try:
            for action in descendants:
                try:
                    recompute_action_in_place(panel, action)
                finally:
                    if action.uuid not in panel.runtime.execution.broken_actions:
                        action.is_stale = False
                    panel.tree.refresh_action_item(action)
                    QW.QApplication.processEvents()
        finally:
            for action in descendants:
                if (
                    action.is_stale
                    and action.uuid not in panel.runtime.execution.broken_actions
                ):
                    action.is_stale = False
                    panel.tree.refresh_action_item(action)
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
