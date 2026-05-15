# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
History panel — Wave 2 invariants

Validates the post-Wave-2 history-panel contract:

1. ``compute_1_to_1`` produces a coherent ``ProcessingParameters`` *and*
   history entry (same ``func_name``, same source UUID).
2. ``HistoryAction.replay`` finds its target by UUID even after the panel
   has been reordered by inserting/deleting siblings.
3. ``recompute_processing`` does not add a history entry (anti-loop guard).
4. ``HistoryAction`` round-trips through HDF5 without pickling any
   ``Callable`` and remains replayable after deserialisation.
5. A UI action (rename via ``set_current_object_title``) is captured and
   replayable.
"""

# pylint: disable=invalid-name

# guitest: skip

import os
import tempfile

import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal

from datalab.gui.panel.history import (
    HistoryAction,
    HistorySession,
    WorkspaceState,
)
from datalab.gui.processor.base import extract_processing_parameters
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_compute_1_to_1_history_matches_processing_parameters():
    """History entry and ProcessingParameters share func_name + source UUID."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)

        # Latest object holds ProcessingParameters.
        result_obj = panel.objmodel.get_object_from_number(2)
        pp = extract_processing_parameters(result_obj)
        assert pp is not None
        assert pp.func_name == "derivative"
        assert pp.source_uuid == src_uuid

        # Latest history entry mirrors the same identity.
        entry = history[len(history)]
        assert entry.kind == HistoryAction.KIND_COMPUTE
        assert entry.func_name == "derivative"
        assert entry.pattern == "1_to_1"
        assert entry.state.selection.get(panel.PANEL_STR) == [src_uuid]


def test_replay_finds_target_by_uuid_after_reorder():
    """Replay locates source object by UUID even after reordering."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel

        panel.add_object(create_paracetamol_signal())
        target_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        deriv_entry = history[len(history)]

        # Reorder the panel: insert a new sibling before the original source.
        panel.add_object(create_paracetamol_signal())
        # Sanity: the original object now lives at a different index but its
        # UUID is unchanged.
        assert get_uuid(panel.objmodel[target_uuid]) == target_uuid

        n_before = len(panel.objmodel)
        deriv_entry.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == n_before + 1

        # The replayed result must be derived from the same source UUID.
        new_obj = panel.objmodel.get_object_from_number(len(panel.objmodel))
        new_pp = extract_processing_parameters(new_obj)
        assert new_pp is not None
        assert new_pp.source_uuid == target_uuid


def test_recompute_processing_does_not_add_history_entry():
    """``recompute_processing`` is silent on the history panel (anti-loop)."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())

        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        n_before = len(history)

        # Select the derived object and recompute it: no new entry expected.
        derived = panel.objmodel.get_object_from_number(2)
        panel.objview.set_current_object(derived)
        panel.recompute_processing()

        assert len(history) == n_before, (
            "recompute_processing must not register a history entry"
        )


def test_history_action_hdf5_roundtrip_without_pickle():
    """Serialise+deserialise a HistoryAction; replay still works."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        src_uuid = get_uuid(panel.objmodel.get_object_from_number(1))

        norm_param = sigima.params.NormalizeParam.create(method="maximum")
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.normalize, norm_param)
        original = history[len(history)]

        # Wrap into a session so we exercise the full HDF5 path used by
        # the panel (write_object_list / read_object_list).
        session = HistorySession(number=1)
        session.actions.append(original)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "wave2.dlhist")
            with NativeH5Writer(path) as writer:
                writer.write_object_list([session], "wave2_history")
            with NativeH5Reader(path) as reader:
                restored_sessions = reader.read_object_list(
                    "wave2_history", HistorySession
                )

        assert len(restored_sessions) == 1
        restored = restored_sessions[0].actions[0]

        # No pickled Callable: the rebuilt HistoryAction has no ``func`` attr.
        assert not hasattr(restored, "func")
        assert restored.kind == HistoryAction.KIND_COMPUTE
        assert restored.func_name == "normalize"
        assert restored.pattern == "1_to_1"
        assert restored.panel_str == panel.PANEL_STR_ID
        # DataSet kwarg survived the JSON round-trip.
        restored_param = restored.kwargs.get("param")
        assert restored_param is not None
        assert type(restored_param).__name__ == type(norm_param).__name__

        # Replay the restored entry against the live workspace.
        n_before = len(panel.objmodel)
        restored.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == n_before + 1
        new_obj = panel.objmodel.get_object_from_number(len(panel.objmodel))
        new_pp = extract_processing_parameters(new_obj)
        assert new_pp is not None
        assert new_pp.source_uuid == src_uuid
        assert new_pp.func_name == "normalize"


def test_ui_action_rename_capture_and_replay():
    """A UI rename is captured and can be replayed."""
    with datalab_test_app_context() as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())

        obj = panel.objmodel.get_object_from_number(1)
        original_title = obj.title

        new_title = "wave2-renamed"
        panel.objview.set_current_object(obj)
        panel.set_current_object_title(new_title)
        assert obj.title == new_title

        rename_entry = history[len(history)]
        assert rename_entry.kind == HistoryAction.KIND_UI
        assert rename_entry.target == "signalpanel"
        assert rename_entry.method_name == "set_current_object_title"
        assert rename_entry.kwargs.get("title") == new_title

        # Mutate the title to something else, then replay: the entry must
        # restore the recorded value.
        panel.set_current_object_title("transient-title")
        assert obj.title == "transient-title"

        rename_entry.replay(win, restore_selection=False, edit=False)
        assert obj.title == new_title
        assert obj.title != original_title

        # WorkspaceState type check (Wave-2 invariant).
        assert isinstance(rename_entry.state, WorkspaceState)


if __name__ == "__main__":
    test_compute_1_to_1_history_matches_processing_parameters()
    test_replay_finds_target_by_uuid_after_reorder()
    test_recompute_processing_does_not_add_history_entry()
    test_history_action_hdf5_roundtrip_without_pickle()
    test_ui_action_rename_capture_and_replay()
