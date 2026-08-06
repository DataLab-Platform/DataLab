# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unified ROI mutation recording and replay in the History panel."""

# guitest: show

from __future__ import annotations

import sigima.proc.signal as sips
from sigima.objects import create_signal_roi
from sigima.tests.data import create_paracetamol_signal

from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import recompute as hrec
from datalab.history.action import HistoryAction
from datalab.history.core import numpy_to_json_safe
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context

SIZE = 200
SROI1 = [26, 41]
SROI2 = [125, 146]


def test_paste_roi_records_mutation_entry() -> None:
    """Pasting a ROI records a per-object mutation entry with the final ROI."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1, SROI2], indices=True)
        panel.add_object(sig1)
        sig2 = create_paracetamol_signal(SIZE)
        panel.add_object(sig2)
        sig2 = panel.objmodel[get_uuid(sig2)]
        # Seed the clipboard from the first signal, then paste on the second
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([2])
        panel.paste_roi()
        action = history.history_sessions[-1].actions[-1]
        assert action.kind == HistoryAction.KIND_MUTATION
        assert action.mutation_key == "roi"
        assert action.target_uuids == [get_uuid(sig2)]
        payload = action.kwargs.get("payload")
        assert payload is not None
        # Payload is the object's post-combination ROI (round-trip check)
        assert numpy_to_json_safe(payload.to_dict()) == numpy_to_json_safe(
            sig2.roi.to_dict()
        )


def test_delete_rois_records_mutation_and_replays() -> None:
    """Deleting ROIs records an empty-payload mutation; session replay works."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1], indices=True)
        panel.add_object(sig1)
        sig2 = create_paracetamol_signal(SIZE)
        panel.add_object(sig2)
        sig2 = panel.objmodel[get_uuid(sig2)]
        # Apply ROI to sig2 (records mutation #1 with payload)
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([2])
        panel.paste_roi()
        assert sig2.roi is not None
        # Remove all ROIs from sig2 (records mutation #2 without payload)
        panel.processor.delete_regions_of_interest()
        assert sig2.roi is None
        actions = history.history_sessions[-1].actions
        apply_action, delete_action = actions[-2], actions[-1]
        assert delete_action.kind == HistoryAction.KIND_MUTATION
        assert delete_action.mutation_key == "roi"
        assert delete_action.target_uuids == [get_uuid(sig2)]
        assert delete_action.kwargs.get("payload") is None
        # Replaying the session in order restores the ROI, then removes it
        apply_action.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is not None
        delete_action.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is None


def test_mutation_replay_applies_roi_to_targets() -> None:
    """Direct mutation replay re-applies the ROI payload to target objects."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1, SROI2], indices=True)
        panel.add_object(sig1)
        sig2 = create_paracetamol_signal(SIZE)
        panel.add_object(sig2)
        sig2 = panel.objmodel[get_uuid(sig2)]
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([2])
        panel.paste_roi()
        action = history.history_sessions[-1].actions[-1]
        assert action.kind == HistoryAction.KIND_MUTATION
        # Clear the ROI, then replay the mutation directly
        sig2.roi = None
        action.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is not None
        assert numpy_to_json_safe(sig2.roi.to_dict()) == numpy_to_json_safe(
            action.kwargs["payload"].to_dict()
        )


def test_cascade_reapplies_roi_mutation() -> None:
    """Cascade recompute re-applies a downstream ROI mutation on the output."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1, SROI2], indices=True)
        panel.add_object(sig1)
        src = create_paracetamol_signal(SIZE)
        panel.add_object(src)
        # Record a compute action producing the output object
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        compute_action = history[len(history)]
        output = panel.objmodel[compute_action.output_uuids[0]]
        # Paste a ROI onto the compute output (records a mutation action)
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([get_uuid(output)])
        panel.paste_roi()
        mutation_action = history.history_sessions[-1].actions[-1]
        assert mutation_action.kind == HistoryAction.KIND_MUTATION
        # The mutation belongs to the compute action's downstream closure
        downstream = hchain.get_downstream_actions(history, compute_action)
        assert mutation_action in downstream
        # Wipe the ROI, then recompute the cascade from the compute action
        output.roi = None
        history.recompute_cascade(compute_action)
        assert output.roi is not None
        assert numpy_to_json_safe(output.roi.to_dict()) == numpy_to_json_safe(
            mutation_action.kwargs["payload"].to_dict()
        )
        assert mutation_action.is_stale is False


def test_update_obj_in_place_preserves_roi() -> None:
    """In-place recompute keeps the target's ROI when the new object has none."""
    with datalab_test_app_context(history=True) as win:
        panel = win.signalpanel
        target = create_paracetamol_signal(SIZE)
        target.roi = create_signal_roi([SROI1], indices=True)
        panel.add_object(target)
        target = panel.objmodel[get_uuid(target)]
        saved_roi_dict = numpy_to_json_safe(target.roi.to_dict())
        new_obj = create_paracetamol_signal(SIZE)
        assert new_obj.roi is None
        hrec.update_obj_in_place(target, new_obj)
        assert target.roi is not None
        assert numpy_to_json_safe(target.roi.to_dict()) == saved_roi_dict


def test_mutation_root_has_downstream_computes() -> None:
    """A mutation root seeds its targets: consuming computes are downstream."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1], indices=True)
        panel.add_object(sig1)
        sig2 = create_paracetamol_signal(SIZE)
        panel.add_object(sig2)
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([2])
        panel.paste_roi()
        mutation_action = history.history_sessions[-1].actions[-1]
        assert mutation_action.kind == HistoryAction.KIND_MUTATION
        # Compute consuming the mutated object is downstream of the mutation
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        compute_action = history.history_sessions[-1].actions[-1]
        assert compute_action.kind == HistoryAction.KIND_COMPUTE
        downstream = hchain.get_downstream_actions(history, mutation_action)
        assert compute_action in downstream


if __name__ == "__main__":
    test_paste_roi_records_mutation_entry()
    test_delete_rois_records_mutation_and_replays()
    test_mutation_replay_applies_roi_to_targets()
    test_cascade_reapplies_roi_mutation()
    test_update_obj_in_place_preserves_roi()
    test_mutation_root_has_downstream_computes()
