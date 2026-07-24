# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application workflow contracts for the History panel."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal

from datalab.gui import historytools_ops as htools
from datalab.gui.panel.history import HistoryAction
from datalab.gui.panel.history.chainmodel import build_session_chains
from datalab.gui.processor.base import extract_processing_parameters
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    add_paracetamol_signals,
    build_signal_chain,
    read_history_sessions,
    select_tree_entry,
    select_tree_session,
)


def assert_compute_action(
    action: HistoryAction, pattern: str, selection: list[str]
) -> None:
    """Check the reusable recording invariant for a compute action."""
    assert action.kind == HistoryAction.KIND_COMPUTE
    assert action.pattern == pattern
    assert action.state.selection["signal"] == selection
    assert action.output_uuids


def assert_duplicate_head(history, panel, session) -> None:
    """Check the synthetic head of an operation-rooted duplicate."""
    head = session.actions[0]
    assert head.kind == HistoryAction.KIND_UI
    assert head.method_name == "new_object"
    assert not head.kwargs and not head.state.selection
    assert len(head.output_uuids) == 1
    clone_uuid = head.output_uuids[0]
    assert history.runtime.objects.action_output_uuids[head.uuid] == [clone_uuid]
    assert history.runtime.objects.output_to_action[clone_uuid] == head.uuid
    assert clone_uuid in panel.objmodel.get_object_ids()


def test_history_recording_contract_and_output_index() -> None:
    """Record producing patterns and index every output without replay entries."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        source_uuids = add_paracetamol_signals(panel, 2)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        derivative = history[len(history)]
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(sips.average)
        average = history[len(history)]
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.difference, panel.objmodel.get_object_from_number(2)
        )
        difference = history[len(history)]
        assert_compute_action(derivative, "1_to_1", [source_uuids[0]])
        assert_compute_action(average, "n_to_1", source_uuids)
        assert_compute_action(difference, "2_to_1", [source_uuids[0]])
        assert difference.kwargs["obj2_uuids"] == [source_uuids[1]]
        for action in (derivative, average, difference):
            for output_uuid in action.output_uuids:
                assert (
                    history.runtime.objects.output_to_action[output_uuid] == action.uuid
                )
        count_before = len(history)
        derivative.replay(win, restore_selection=True, edit=False)
        assert len(history) == count_before


def test_session_replay_remaps_distinct_processing_patterns() -> None:
    """Replay chained 1-to-1, n-to-1 and ordered 2-to-1 computations."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 2)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        panel.objview.select_objects([3, 4])
        panel.processor.run_feature(sips.average)
        panel.objview.select_objects([3])
        panel.processor.run_feature(
            sips.difference, panel.objmodel.get_object_from_number(4)
        )
        session = history.history_sessions[-1]
        difference = session.actions[-1]
        original_title = panel.objmodel[difference.output_uuids[0]].title
        assert original_title.index("s003") < original_title.index("s004")
        object_count = len(panel.objmodel)
        action_count = len(history)
        session.replay(win, restore_selection=False, edit=False)
        assert len(panel.objmodel) == object_count + 4
        assert len(history) == action_count
        replayed = panel.objmodel.get_object_from_number(len(panel.objmodel))
        assert "s" in replayed.title and "-" in replayed.title


def test_analysis_replay_requires_explicit_edit() -> None:
    """Skip an analysis during ordinary replay but allow explicit edit replay."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.stats)
        analysis_action = history[len(history)]
        assert analysis_action.pattern == "1_to_0"
        with patch.object(panel.processor, "run_feature") as run_feature:
            analysis_action.replay(win, restore_selection=True, edit=False)
            run_feature.assert_not_called()
            analysis_action.replay(win, restore_selection=True, edit=True)
            run_feature.assert_called_once()


def test_history_hdf5_pristine_load_and_nonempty_import() -> None:
    """Distinguish pristine loading, non-empty import and missing history."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_path = os.path.join(tmpdir, "session.dlhist")
        empty_path = os.path.join(tmpdir, "without_history.h5")
        with datalab_test_app_context(history=True) as source:
            history, panel = source.historypanel, source.signalpanel
            history.toggle_record_mode(True)
            add_paracetamol_signals(panel, 1)
            panel.objview.select_objects([1])
            panel.processor.run_feature(sips.derivative)
            titles = [action.title for action in history]
            assert history.save_to_dlhist_file(history_path)
            with NativeH5Writer(empty_path) as writer:
                panel.serialize_to_hdf5(writer)
        with datalab_test_app_context(history=True) as target:
            history, panel = target.historypanel, target.signalpanel
            with NativeH5Reader(empty_path) as reader:
                history.deserialize_from_hdf5(reader)
            assert len(history) == 0
            assert history.open_dlhist_file(history_path)
            assert [action.title for action in history] == titles
            assert history.runtime.objects.action_output_uuids
            assert history.runtime.objects.output_to_action
            with NativeH5Reader(empty_path) as reader:
                history.deserialize_from_hdf5(reader)
            assert not history.history_sessions
            assert not history.runtime.objects.action_output_uuids
            assert not history.runtime.objects.output_to_action
            pristine_counts = (len(history.history_sessions), len(panel.objmodel))
            panel.add_object(create_paracetamol_signal())
            assert history.open_dlhist_file(history_path)
            assert len(history.history_sessions) > pristine_counts[0]
            assert len(panel.objmodel) > pristine_counts[1] + 1


def test_duplicate_creation_and_operation_rooted_chains() -> None:
    """Duplicate both root kinds and synthesize a head only when required."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        win.add_object(create_paracetamol_signal())
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        first_source = history.history_sessions[-1]
        first_source.actions[-1].plugin_origin = {
            "module": "example.plugin",
            "metadata": {"entry_points": ["derivative"]},
        }
        history.create_new_session(panel_str="signal")
        win.add_object(create_paracetamol_signal())
        panel.objview.select_objects([3])
        panel.processor.run_feature(sips.derivative)
        second_source = history.history_sessions[-1]
        history.tree.clearSelection()
        for source in (first_source, second_source):
            source_item = history.tree.topLevelItem(
                history.history_sessions.index(source)
            )
            source_item.setSelected(True)
        htools.duplicate_selected_entries(history)
        first_duplicate = history.history_sessions[1]
        second_duplicate = history.history_sessions[3]
        assert history.history_sessions == [
            first_source,
            first_duplicate,
            second_source,
            second_duplicate,
        ]
        duplicate = first_duplicate
        assert len(duplicate.actions) == len(first_source.actions)
        assert duplicate.actions[0].method_name == "new_object"
        assert duplicate.actions[0].uuid != first_source.actions[0].uuid
        assert set(duplicate.actions[0].output_uuids).isdisjoint(
            first_source.actions[0].output_uuids
        )
        duplicate_compute = duplicate.actions[-1]
        assert duplicate_compute.plugin_origin == first_source.actions[-1].plugin_origin
        duplicate_compute.plugin_origin["metadata"]["entry_points"].append("average")
        assert first_source.actions[-1].plugin_origin["metadata"]["entry_points"] == [
            "derivative"
        ]
        duplicate_output = panel.objmodel[duplicate_compute.output_uuids[0]]
        processing = extract_processing_parameters(duplicate_output)
        assert processing is not None
        assert processing.source_uuid == duplicate.actions[0].output_uuids[0]
        assert history.runtime.objects.output_to_action[get_uuid(duplicate_output)] == (
            duplicate_compute.uuid
        )
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        original = history.history_sessions[-1]
        select_tree_session(history, original)
        htools.duplicate_selected_entries(history)
        duplicate = history.history_sessions[-1]
        assert len(duplicate.actions) == len(original.actions) + 1
        assert_duplicate_head(history, panel, duplicate)
        chains = build_session_chains(duplicate)
        assert len(chains) == 1 and chains[0].root is duplicate.actions[0]


def test_edit_cascade_preserves_identity_and_action_state() -> None:
    """Cascade in place while preserving identities, metadata and edit baseline."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        chain = build_signal_chain(panel, history)
        root_action, _middle_action, leaf_action = chain.actions
        root_output, _middle_output, leaf_output = chain.outputs
        panel.objview.select_objects([leaf_output])
        panel.processor.run_feature(sips.stats)
        analysis_action = history[len(history)]
        assert analysis_action.pattern == "1_to_0"
        leaf_uuid = get_uuid(leaf_output)
        leaf_number = panel.objmodel.get_number(leaf_output)
        leaf_data = leaf_output.xydata.copy()
        leaf_output.metadata["user_marker"] = 123
        panel.objview.select_objects([2])
        assert panel.objprop.setup_processing_tab(root_output, reset_params=False)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        editor.dataset.sigma = 7.0
        with patch.object(
            panel.processor,
            "recompute_1_to_0",
            wraps=panel.processor.recompute_1_to_0,
        ) as recompute_analysis:
            report = panel.objprop.apply_processing_parameters(
                root_output, interactive=False
            )
        recompute_analysis.assert_called_once()
        assert report.success and root_action.has_pending_edits
        assert root_action.kwargs["param"].sigma == 7.0
        assert get_uuid(panel.objmodel[leaf_uuid]) == leaf_uuid
        assert panel.objmodel.get_number(panel.objmodel[leaf_uuid]) == leaf_number
        assert panel.objmodel[leaf_uuid].metadata["user_marker"] == 123
        assert not np.array_equal(panel.objmodel[leaf_uuid].xydata, leaf_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "edited.dlhist")
            assert history.save_to_dlhist_file(path)
            sessions = read_history_sessions(path, history.H5_PREFIX)
            restored = next(
                action
                for session in sessions
                for action in session.actions
                if action.uuid == root_action.uuid
            )
        assert restored.has_pending_edits and restored.kwargs["param"].sigma == 7.0
        restored.restore_kwargs()
        assert restored.kwargs["param"].sigma == 1.5
        assert leaf_action.is_stale is False


def test_deletion_reconnects_and_splices_chain() -> None:
    """Reconnect after data deletion, then splice the producing action."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        win.add_object(create_paracetamol_signal())
        source_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        normalize = history[len(history)]
        intermediate_uuid = normalize.output_uuids[0]
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        derivative = history[len(history)]
        panel.objview.select_objects([2])
        panel.remove_object(force=True)
        assert intermediate_uuid not in derivative.state.selection["signal"]
        assert source_uuid in derivative.state.selection["signal"]
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(
            sips.normalize, sigima.params.NormalizeParam.create(method="maximum")
        )
        action_to_delete = history[len(history)]
        panel.objview.select_objects([action_to_delete.output_uuids[0]])
        panel.processor.run_feature(sips.derivative)
        downstream_action = history[len(history)]
        session = history.history_sessions[-1]
        object_count = len(panel.objmodel)
        select_tree_entry(history, action_to_delete.uuid)
        htools.delete_selected(history)
        assert action_to_delete not in session.actions
        assert action_to_delete.uuid not in history.runtime.objects.action_output_uuids
        assert (
            action_to_delete.output_uuids[0]
            not in history.runtime.objects.output_to_action
        )
        assert derivative in session.actions and downstream_action in session.actions
        assert len(panel.objmodel) == object_count + 1
        assert (
            action_to_delete.output_uuids[0]
            not in downstream_action.state.selection["signal"]
        )
        chains = build_session_chains(session)
        assert sum(len(chain.actions) for chain in chains) == len(session.actions)
        removed_action_uuids = [action.uuid for action in session.actions]
        removed_output_uuids = [
            output_uuid
            for action in session.actions
            for output_uuid in action.output_uuids
        ]
        select_tree_session(history, session)
        htools.delete_selected(history)
        assert session not in history.history_sessions
        assert all(
            action_uuid not in history.runtime.objects.action_output_uuids
            for action_uuid in removed_action_uuids
        )
        assert all(
            output_uuid not in history.runtime.objects.output_to_action
            for output_uuid in removed_output_uuids
        )
