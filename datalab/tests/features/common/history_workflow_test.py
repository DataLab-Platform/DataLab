# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application workflow contracts for the History panel."""

from __future__ import annotations

import copy
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import sigima.params
import sigima.proc.signal as sips
from sigima.objects import Gauss2DParam, create_signal_roi
from sigima.tests.data import (
    create_paracetamol_signal,
    create_peak_image,
    create_sincos_image,
)
from sigima.tools.signal import fitting as signal_fitting

from datalab.adapters_metadata.common import ResultData
from datalab.config import Conf
from datalab.gui import historytools_ops as htools
from datalab.gui.creation import create_image_from_param, extract_creation_parameters
from datalab.gui.panel.history import HistoryAction
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.panel.history.chainmodel import (
    build_session_chains,
    remap_processing_parameters,
)
from datalab.gui.processor.base import (
    ProcessingParameters,
    extract_analysis_parameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history.core import numpy_to_json_safe
from datalab.history.effects import AnalysisEffects
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    add_paracetamol_signals,
    build_signal_chain,
    get_tree_item,
    read_history_sessions,
    select_tree_entry,
    select_tree_session,
)

SIZE = 200
SROI1 = [26, 41]
SROI2 = [125, 146]


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


def build_independent_signal_branch(panel, history) -> tuple[HistoryAction, ...]:
    """Build a three-action branch using UUID-based selections."""
    source_uuid = add_paracetamol_signals(panel, 1)[0]
    panel.objview.select_objects([source_uuid])
    panel.processor.run_feature(
        sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
    )
    first_action = history[len(history)]
    panel.objview.select_objects(first_action.output_uuids)
    panel.processor.run_feature(sips.derivative)
    second_action = history[len(history)]
    panel.objview.select_objects(second_action.output_uuids)
    panel.processor.run_feature(
        sips.moving_average, sigima.params.MovingAverageParam.create(n=3)
    )
    return first_action, second_action, history[len(history)]


def test_remap_processing_parameters_preserves_plugin_origin() -> None:
    """Preserve plugin provenance while remapping processing source UUIDs."""
    plugin_origin = {"module": "test_plugin.operations", "directory": "test_plugin"}
    parameters = ProcessingParameters(
        func_name="difference",
        pattern="2-to-1",
        source_uuids=["source-1", "source-2"],
        plugin_origin=plugin_origin,
    )

    remapped = remap_processing_parameters(
        parameters, {"source-1": "copy-1", "source-2": "copy-2"}
    )

    assert remapped.source_uuids == ["copy-1", "copy-2"]
    assert remapped.plugin_origin == plugin_origin


def test_history_recording_contract_and_output_index() -> None:
    """Record producing patterns and index every output."""
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
            # Full history reset drops sessions, mappings, navigation and tree
            assert isinstance(history.create_object(), HistoryAction)
            history.remove_all_objects()
            assert len(history) == 0 and not history.history_sessions
            assert not history.runtime.objects.action_output_uuids
            assert not history.runtime.objects.output_to_action
            assert history.navigation.get_active_session() is None
            assert history.tree.topLevelItemCount() == 0
            with pytest.raises(IndexError):
                history[1]  # pylint: disable=pointless-statement


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
        history.create_new_session()
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


def test_duplicate_clones_only_chain_objects() -> None:
    """Duplicate clones only chain inputs/outputs, not unrelated objects."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        # Unrelated objects alive at record time (captured in workspace state)
        unrelated = create_paracetamol_signal()
        unrelated.title = "Unrelated signal"
        panel.add_object(unrelated)
        win.imagepanel.add_object(create_sincos_image())
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(sips.derivative)
        original = history.history_sessions[-1]
        signal_count_before = len(panel.objmodel)
        image_count_before = len(win.imagepanel.objmodel)
        image_group_count_before = len(win.imagepanel.objmodel.get_groups())
        title_count_before = sum(
            panel.objmodel[uuid].title == unrelated.title
            for uuid in panel.objmodel.get_object_ids()
        )
        select_tree_session(history, original)
        htools.duplicate_selected_entries(history)
        # Only the chain source and its derivative output are cloned
        assert len(panel.objmodel) == signal_count_before + 2
        title_count_after = sum(
            panel.objmodel[uuid].title == unrelated.title
            for uuid in panel.objmodel.get_object_ids()
        )
        assert title_count_after == title_count_before
        # Image panel is untouched
        assert len(win.imagepanel.objmodel) == image_count_before
        assert len(win.imagepanel.objmodel.get_groups()) == image_group_count_before
        duplicate = history.history_sessions[-1]
        original_outputs = {
            uuid for action in original.actions for uuid in action.output_uuids
        }
        duplicate_outputs = {
            uuid for action in duplicate.actions for uuid in action.output_uuids
        }
        assert duplicate_outputs.isdisjoint(original_outputs)


def test_edit_cascade_preserves_identity_and_action_state() -> None:
    """Cascade in place while preserving identities, metadata and edit baseline."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        chain = build_signal_chain(panel, history)
        root_action, middle_action, leaf_action = chain.actions
        root_output, middle_output, leaf_output = chain.outputs
        plugin_origin = {
            "module": "test_plugin.operations",
            "directory": "test_plugin",
        }
        middle_action.plugin_origin = plugin_origin
        leaf_action.plugin_origin = plugin_origin
        panel.objview.select_objects([leaf_output])
        panel.processor.run_feature(sips.stats)
        analysis_action = history[len(history)]
        assert analysis_action.pattern == "1_to_0"
        middle_parameters = extract_processing_parameters(middle_output)
        assert middle_parameters is not None
        middle_parameters.plugin_origin = plugin_origin
        insert_processing_parameters(middle_output, middle_parameters)
        middle_action.plugin_origin = None
        analysis_action.plugin_origin = None
        analysis_parameters = extract_analysis_parameters(leaf_output)
        assert analysis_parameters is not None
        analysis_parameters.plugin_origin = plugin_origin
        leaf_output.set_metadata_option(
            "analysis_parameters", analysis_parameters.to_dict()
        )
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
        assert recompute_analysis.call_args.kwargs["plugin_origin"] == plugin_origin
        assert report.success and root_action.has_pending_edits
        assert root_action.kwargs["param"].sigma == 7.0
        assert get_uuid(panel.objmodel[leaf_uuid]) == leaf_uuid
        assert panel.objmodel.get_number(panel.objmodel[leaf_uuid]) == leaf_number
        assert panel.objmodel[leaf_uuid].metadata["user_marker"] == 123
        assert not np.array_equal(panel.objmodel[leaf_uuid].xydata, leaf_data)
        for output in (middle_output, leaf_output):
            parameters = extract_processing_parameters(output)
            assert parameters is not None
            assert parameters.plugin_origin == plugin_origin
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


def test_edit_cascade_stops_after_failed_descendant() -> None:
    """Keep a failed action and unexecuted analysis stale after cascade failure."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        panel.processor.run_feature(
            sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
        )
        first_action = history[len(history)]
        panel.objview.select_objects([2])
        panel.processor.run_feature(sips.derivative)
        failed_action = history[len(history)]
        failed_output = panel.objmodel[failed_action.output_uuids[0]]
        failed_data = failed_output.xydata.copy()
        panel.objview.select_objects([failed_action.output_uuids[0]])
        panel.processor.run_feature(sips.stats)
        analysis_action = history[len(history)]
        first_action.is_stale = True
        original_compute = panel.processor.compute_1_to_1
        call_count = 0

        def fail_second_compute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return None  # No output produced: the replay reconcile fails
            return original_compute(*args, **kwargs)

        with (
            patch.object(
                panel.processor,
                "compute_1_to_1",
                side_effect=fail_second_compute,
            ),
            patch.object(panel.processor, "recompute_1_to_0") as recompute_analysis,
        ):
            history.recompute_cascade(first_action)

        recompute_analysis.assert_not_called()
        assert first_action.is_stale is False
        assert failed_action.is_stale is True
        assert analysis_action.is_stale is True
        assert np.array_equal(failed_output.xydata, failed_data)


def test_multi_action_edit_single_session_planning() -> None:
    """Plan selected ancestors, descendants and full-session selections once.

    Selected ancestors are prompted exactly once and their analysis descendant
    is recomputed once; selecting the whole session plus one of its (stale)
    actions routes through the global replay planner without duplicates.
    """
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        chain = build_signal_chain(panel, history)
        panel.objview.select_objects([chain.outputs[-1]])
        panel.processor.run_feature(sips.stats)
        analysis_action = history[len(history)]
        selected = [chain.actions[0], chain.actions[1]]
        expected = [*chain.actions, analysis_action]

        with (
            patch.object(
                hireplay, "prompt_edit_action_params", return_value=True
            ) as prompt,
            patch.object(
                hrec, "recompute_action_in_place", return_value=True
            ) as recompute,
        ):
            hireplay.replay_actions(history, selected)

        assert [call.args[1] for call in prompt.call_args_list] == selected
        assert [call.args[1] for call in recompute.call_args_list] == expected
        assert all(action.is_stale is False for action in expected)

        # Selecting the session plus one stale action plans each action once
        session = history.history_sessions[-1]
        session_expected = list(session.actions)
        stale_action = session_expected[1]
        stale_action.is_stale = True
        select_tree_session(history, session)
        get_tree_item(history, stale_action.uuid).setSelected(True)
        selected_items = history.tree.get_selected_actions_or_sessions(
            history.history_sessions
        )
        assert selected_items == [session, stale_action]

        with (
            patch.object(hrec, "recompute_cascade") as direct_cascade,
            patch.object(
                hireplay,
                "replay_actions",
                wraps=hireplay.replay_actions,
            ) as edit_planner,
            patch.object(
                hireplay, "prompt_edit_action_params", return_value=True
            ) as prompt,
            patch.object(
                hrec, "recompute_action_in_place", return_value=True
            ) as recompute,
        ):
            hireplay.replay_restore_actions(history)

        direct_cascade.assert_not_called()
        edit_planner.assert_called_once_with(
            history, [*session_expected, stale_action], prompt=True
        )
        assert [call.args[1] for call in prompt.call_args_list] == session_expected
        assert [call.args[1] for call in recompute.call_args_list] == session_expected
        assert all(action.is_stale is False for action in session_expected)


def test_downstream_actions_follow_every_registered_output() -> None:
    """Follow second registered outputs through transitive dependencies."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        producer, consumer, descendant = build_signal_chain(panel, history).actions
        producer_second_output = "producer-second-output"
        consumer_second_output = "consumer-second-output"
        producer.output_uuids.append(producer_second_output)
        consumer.output_uuids.append(consumer_second_output)
        history.runtime.objects.action_output_uuids[producer.uuid] = list(
            producer.output_uuids
        )
        history.runtime.objects.action_output_uuids[consumer.uuid] = list(
            consumer.output_uuids
        )
        history.runtime.objects.output_to_action[producer_second_output] = producer.uuid
        history.runtime.objects.output_to_action[consumer_second_output] = consumer.uuid
        history.runtime.objects.prune_output_mapping()
        assert producer_second_output in producer.output_uuids
        assert consumer_second_output in consumer.output_uuids
        assert (
            producer_second_output
            not in (history.runtime.objects.action_output_uuids[producer.uuid])
        )
        assert (
            consumer_second_output
            not in (history.runtime.objects.action_output_uuids[consumer.uuid])
        )
        consumer.state.selection["signal"] = [producer_second_output]
        descendant.state.selection["signal"] = [consumer_second_output]

        assert hchain.get_downstream_actions(history, producer) == [
            consumer,
            descendant,
        ]


def test_multi_action_edit_cascades_across_independent_sessions() -> None:
    """Recompute edited branches from multiple sessions in global order."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        first_chain = build_independent_signal_branch(panel, history)
        history.create_new_session()
        second_chain = build_independent_signal_branch(panel, history)
        selected = [second_chain[0], first_chain[0]]
        expected = [*first_chain, *second_chain]

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(
                hrec, "recompute_action_in_place", return_value=True
            ) as recompute,
        ):
            hireplay.replay_actions(history, selected)

        assert [call.args[1] for call in recompute.call_args_list] == expected
        assert all(action.is_stale is False for action in expected)


def test_multi_action_edit_failure_skips_dependents_and_continues() -> None:
    """Leave a failed branch stale while recomputing an independent session."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        failed_chain = build_independent_signal_branch(panel, history)
        history.create_new_session()
        successful_chain = build_independent_signal_branch(panel, history)
        failed_root = failed_chain[0]
        failed_output_uuid = failed_root.output_uuids[0]
        failed_root.output_uuids.clear()
        history.runtime.objects.action_output_uuids.pop(failed_root.uuid)
        history.runtime.objects.output_to_action.pop(failed_output_uuid)
        failed_output = panel.objmodel[failed_output_uuid]
        processing_parameters = extract_processing_parameters(failed_output)
        assert not failed_root.output_uuids
        assert failed_root.uuid not in history.runtime.objects.action_output_uuids
        assert failed_output_uuid not in history.runtime.objects.output_to_action
        assert processing_parameters is not None
        assert processing_parameters.func_name == failed_root.func_name
        assert hchain.recorded_action_output_uuids(history, failed_root) == [
            failed_output_uuid
        ]
        recomputed: list[HistoryAction] = []

        def recompute_action(_panel, action):
            recomputed.append(action)
            return action is not failed_root

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(
                hrec, "recompute_action_in_place", side_effect=recompute_action
            ),
        ):
            hireplay.replay_actions(history, [failed_root, successful_chain[0]])

        assert recomputed == [failed_root, *successful_chain]
        assert all(action.is_stale is True for action in failed_chain)
        assert all(action.is_stale is False for action in successful_chain)


def test_multi_action_edit_cancel_restores_entry_pending_edit() -> None:
    """Restore current kwargs and their saved baseline after a later cancel."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        first_action, second_action = build_signal_chain(panel, history).actions[:2]
        first_action.snapshot_kwargs()
        first_action.kwargs["param"].sigma = 2.5

        def prompt(_panel, action):
            if action is first_action:
                action.kwargs["param"].sigma = 3.5
                return True
            return False

        with patch.object(hireplay, "prompt_edit_action_params", side_effect=prompt):
            hireplay.replay_actions(history, [first_action, second_action])

        assert first_action.kwargs["param"].sigma == 2.5
        assert first_action.saved_kwargs["param"].sigma == 1.5


def test_multi_action_edit_cancel_skips_deferred_ui_replay() -> None:
    """Do not replay noncompute UI actions when a later dialog is cancelled."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        compute_action = build_signal_chain(panel, history).actions[0]
        ui_action = HistoryAction(
            title="Select next",
            kind=HistoryAction.KIND_UI,
            target="signalpanel",
            method_name="select_next",
        )

        with (
            patch.object(ui_action, "replay") as replay,
            patch.object(hireplay, "prompt_edit_action_params", return_value=False),
        ):
            hireplay.replay_actions(history, [ui_action, compute_action])

        replay.assert_not_called()


def test_multi_action_edit_preserves_mixed_ui_compute_order() -> None:
    """Execute deferred UI and planned compute actions in global session order."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(sips.derivative)
        first_compute = history[len(history)]
        ui_action = history.add_ui_entry("Select next", "signalpanel", "select_next")
        assert ui_action is not None
        panel.objview.select_objects(first_compute.output_uuids)
        panel.processor.run_feature(sips.derivative)
        second_compute = history[len(history)]
        execution_order = []

        def recompute(_panel, action):
            execution_order.append(action)
            return True

        def replay_ui(*_args, **_kwargs):
            execution_order.append(ui_action)

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(hrec, "recompute_action_in_place", side_effect=recompute),
            patch.object(ui_action, "replay", side_effect=replay_ui),
        ):
            hireplay.replay_actions(history, [first_compute, ui_action])

        assert execution_order == [first_compute, ui_action, second_compute]


def test_multi_action_edit_flushes_cascade_warnings_once() -> None:
    """Flush warnings exactly once after executing a custom replay plan."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        action = build_signal_chain(panel, history).actions[0]

        def recompute(_panel, _action):
            history.runtime.execution.cascade_warnings.append("expected warning")
            return True

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(hrec, "recompute_action_in_place", side_effect=recompute),
            patch.object(
                hrec,
                "flush_cascade_warnings",
                wraps=hrec.flush_cascade_warnings,
            ) as flush,
        ):
            hireplay.replay_actions(history, [action])

        flush.assert_called_once_with(history)
        assert not history.runtime.execution.cascade_warnings


def test_restore_failure_marks_action_stale_without_cascade() -> None:
    """Stop restore recomputation when its root action fails."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        action = build_signal_chain(panel, history).actions[0]
        action.snapshot_kwargs()
        action.kwargs["param"].sigma = 7.0

        with (
            patch.object(hrec, "recompute_action_in_place", return_value=False),
            patch.object(hrec, "recompute_cascade") as recompute_cascade,
        ):
            hireplay.restore_action_params(history, action)

        recompute_cascade.assert_not_called()
        assert action.is_stale is True


def test_restore_recomputes_stale_action_without_pending_edits() -> None:
    """Recompute a stale action on restore even without pending edits."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        action = build_signal_chain(panel, history).actions[0]
        action.is_stale = True
        assert not action.has_pending_edits

        with (
            patch.object(
                hrec, "recompute_action_in_place", return_value=True
            ) as recompute,
            patch.object(hrec, "recompute_cascade") as recompute_cascade,
        ):
            hireplay.restore_action_params(history, action)

        recompute.assert_called_once_with(history, action)
        recompute_cascade.assert_called_once_with(history, action)
        assert action.is_stale is False


def test_empty_analysis_result_is_successful() -> None:
    """Treat an executed analysis with no detections as successful."""
    with datalab_test_app_context(history=True) as win:
        panel = win.signalpanel
        panel.new_object(edit=False)
        signal = panel.objview.get_current_object()
        assert signal is not None

        with patch.object(panel.processor, "compute_1_to_0", return_value=ResultData()):
            success = panel.processor.recompute_1_to_0("stats", signal)

        assert success is True


def test_legacy_resultdata_defaults_execution_success() -> None:
    """Use the dataclass default when legacy state lacks execution_success."""
    result = ResultData()
    del result.__dict__["execution_success"]

    assert result.execution_success is True


def test_2_to_1_failure_does_not_partially_mutate_outputs() -> None:
    """Discard fresh outputs and warn on a replay cardinality mismatch."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 4)
        actions = []
        for first, second in ((1, 2), (3, 4)):
            panel.objview.select_objects([first])
            panel.processor.run_feature(
                sips.difference, panel.objmodel.get_object_from_number(second)
            )
            actions.append(history[len(history)])
        action = actions[0]
        action.output_uuids.extend(actions[1].output_uuids)
        history.runtime.objects.action_output_uuids[action.uuid] = list(
            action.output_uuids
        )
        outputs = [panel.objmodel[uuid] for uuid in action.output_uuids]
        original_data = [obj.xydata.copy() for obj in outputs]
        object_count = len(panel.objmodel)

        # The synthetic action records two outputs but its captured selection
        # only produces one: the replay must discard the fresh output and
        # leave the recorded outputs untouched
        success = hrec.recompute_action_in_place(history, action)

        assert success is False
        assert action.is_stale is True
        assert len(panel.objmodel) == object_count
        assert any(
            "expected 2" in warning
            for warning in history.runtime.execution.cascade_warnings
        )
        history.runtime.execution.cascade_warnings.clear()
        for output, data in zip(outputs, original_data):
            assert np.array_equal(output.xydata, data)


def test_2_to_1_refresh_failure_rolls_back_and_resyncs_outputs() -> None:
    """Commit all 2-to-1 outputs before refresh and fully roll back on failure."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 3)
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(
            sips.difference, panel.objmodel.get_object_from_number(3)
        )
        action = history[len(history)]
        assert len(action.output_uuids) == 2
        outputs = [panel.objmodel[uuid] for uuid in action.output_uuids]
        identities = [id(obj) for obj in outputs]
        # Mutate the outputs so a successful commit is distinguishable from a
        # rollback restoring the pre-replay state
        original_titles = []
        original_data = []
        for index, output in enumerate(outputs):
            output.title = f"mutated-{index}"
            output.xydata = output.xydata * 0.0
            original_titles.append(output.title)
            original_data.append(output.xydata.copy())
        refresh_effects = []

        def refresh_with_failure(_panel, output_uuid):
            refresh_effects.append((output_uuid, [obj.title for obj in outputs]))
            if len(refresh_effects) == 2:
                raise RuntimeError(f"refresh failed #{len(refresh_effects)}")

        with patch.object(hrec, "refresh_target", side_effect=refresh_with_failure):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        assert [effect[0] for effect in refresh_effects] == [
            action.output_uuids[0],
            action.output_uuids[1],
            action.output_uuids[0],
            action.output_uuids[1],
        ]
        # Both outputs were committed before the first refresh...
        assert all(
            title != original
            for title, original in zip(refresh_effects[0][1], original_titles)
        )
        # ...and both were restored before the rollback refreshes
        assert refresh_effects[2][1] == original_titles
        for index, output in enumerate(outputs):
            assert id(output) == identities[index]
            assert output.title == original_titles[index]
            assert np.array_equal(output.xydata, original_data[index])
        history.runtime.execution.cascade_warnings.clear()


def test_1_to_n_refresh_failure_rolls_back_and_resyncs_outputs() -> None:
    """Commit all 1-to-n outputs before refresh and fully roll back on failure."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        panel.objview.select_objects([source_uuid])
        feature = panel.processor.get_feature("gaussian_filter")
        params = [
            sigima.params.GaussianParam.create(sigma=1.5),
            sigima.params.GaussianParam.create(sigma=2.5),
        ]
        panel.processor.compute_1_to_n(feature.function, params=params, edit=False)
        action = history[len(history)]
        assert action.pattern == "1_to_n"
        assert len(action.output_uuids) == 2
        outputs = [panel.objmodel[uuid] for uuid in action.output_uuids]
        identities = [id(obj) for obj in outputs]
        # Mutate the outputs so a successful commit is distinguishable from a
        # rollback restoring the pre-replay state
        original_titles = []
        original_data = []
        for index, output in enumerate(outputs):
            output.title = f"mutated-{index}"
            output.xydata = output.xydata * 0.0
            original_titles.append(output.title)
            original_data.append(output.xydata.copy())
        refresh_effects = []

        def refresh_with_failure(_panel, output_uuid):
            refresh_effects.append((output_uuid, [obj.title for obj in outputs]))
            if len(refresh_effects) == 2:
                raise RuntimeError(f"refresh failed #{len(refresh_effects)}")

        with patch.object(hrec, "refresh_target", side_effect=refresh_with_failure):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        assert [effect[0] for effect in refresh_effects] == [
            action.output_uuids[0],
            action.output_uuids[1],
            action.output_uuids[0],
            action.output_uuids[1],
        ]
        assert all(
            title != original
            for title, original in zip(refresh_effects[0][1], original_titles)
        )
        assert refresh_effects[2][1] == original_titles
        for index, output in enumerate(outputs):
            assert id(output) == identities[index]
            assert output.title == original_titles[index]
            assert np.array_equal(output.xydata, original_data[index])
        history.runtime.execution.cascade_warnings.clear()

        # Misaligned recording: with a params list longer than the recorded
        # outputs, the replay produces more objects than expected and the
        # cardinality guard rejects the recompute
        action.kwargs["params"].append(sigima.params.GaussianParam.create(sigma=3.5))
        object_count = len(panel.objmodel)
        assert hrec.recompute_action_in_place(history, action) is False
        assert len(panel.objmodel) == object_count
        assert any(
            "expected 2" in warning
            for warning in history.runtime.execution.cascade_warnings
        )
        history.runtime.execution.cascade_warnings.clear()


def test_1_to_0_failure_rolls_back_all_source_metadata() -> None:
    """Roll back analysis sources on failure, full-snapshot and targeted alike."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        source_uuids = add_paracetamol_signals(panel, 2)
        panel.objview.select_objects([1, 2])
        panel.processor.run_feature(sips.stats)
        action = history[len(history)]
        sources = [panel.objmodel[uuid] for uuid in source_uuids]
        for index, source in enumerate(sources):
            source.metadata["user_marker"] = index

        call_count = 0

        def fail_second_analysis(_func_name, source, _param, **_kwargs):
            nonlocal call_count
            call_count += 1
            source.metadata["temporary_analysis"] = call_count
            return call_count == 1

        with patch.object(
            panel.processor,
            "recompute_1_to_0",
            side_effect=fail_second_analysis,
        ):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        for index, source in enumerate(sources):
            assert source.metadata["user_marker"] == index
            assert "temporary_analysis" not in source.metadata

        # Manifest-driven analysis: a failed recompute rolls back only the
        # manifest keys and leaves unrelated user metadata untouched
        image_panel = win.imagepanel
        img = create_peak_image()
        image_panel.add_object(img)
        det_param = sigima.params.Peak2DDetectionParam.create(
            create_rois=False, threshold=0.5
        )
        with Conf.show_result_dialog.context(False):
            image_panel.processor.run_feature("peak_detection", det_param)
        img_action = history[len(history)]
        img_uuid = get_uuid(img)
        manifest = AnalysisEffects.from_dict(img_action.effects[img_uuid])
        manifest_keys = manifest.metadata_added + manifest.metadata_replaced
        geometry_key = next(key for key in manifest_keys if key.startswith("Geometry_"))
        # Simulate a user having deleted one analysis result key beforehand
        del img.metadata[geometry_key]
        present_key = next(key for key in manifest_keys if key in img.metadata)
        value_before = copy.deepcopy(img.metadata[present_key])
        img.metadata["user_marker"] = 123
        effects_before = copy.deepcopy(img_action.effects)

        def failing_recompute(_func_name, obj, _param, **_kwargs):
            obj.metadata[geometry_key] = "recreated-by-failed-attempt"
            obj.metadata[present_key] = "corrupted"
            raise RuntimeError("forced recompute failure")

        with patch.object(
            image_panel.processor, "recompute_1_to_0", side_effect=failing_recompute
        ):
            try:
                hrec.recompute_1_to_0_in_place(history, img_action)
            except RuntimeError:
                pass
            else:
                raise AssertionError("RuntimeError should have propagated")

        assert img.metadata["user_marker"] == 123, "Unrelated key must be untouched"
        assert geometry_key not in img.metadata, (
            "Manifest key absent before the recompute must be deleted on rollback"
        )
        assert img.metadata[present_key] == value_before, (
            "Manifest key must be restored to its pre-recompute value"
        )
        assert img_action.effects == effects_before, "Manifest must be unchanged"


def test_1_to_0_cascade_uses_roi_safe_parameter_copy() -> None:
    """Disable ROI creation on a copy during analysis cascade recomputation."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        param = SimpleNamespace(create_rois=True)
        action = HistoryAction()
        action.kind = HistoryAction.KIND_COMPUTE
        action.pattern = "1_to_0"
        action.target = "signalpanel"
        action.panel_str = "signal"
        action.func_name = "stats"
        action.kwargs = {"param": param}
        action.state.selection = {panel.PANEL_STR_ID: [source_uuid]}

        # The guard now lives inside recompute_1_to_0: spy on compute_1_to_0
        # to observe the parameter actually passed to the executed analysis
        with patch.object(
            panel.processor,
            "compute_1_to_0",
            return_value=SimpleNamespace(execution_success=True),
        ) as compute:
            success = hrec.recompute_action_in_place(history, action)

        assert success is True
        passed_param = compute.call_args.args[1]
        assert passed_param is not param
        assert passed_param.create_rois is False
        assert action.kwargs["param"].create_rois is True


def test_replay_recreates_deleted_output_under_recorded_uuid() -> None:
    """Re-create a deleted compute output under its recorded UUID on replay."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(sips.derivative)
        action = history[len(history)]
        output_uuid = action.output_uuids[0]
        expected_data = panel.objmodel[output_uuid].xydata.copy()
        object_count = len(panel.objmodel)
        panel.objview.select_objects([output_uuid])
        panel.remove_object(force=True)
        assert not panel.objmodel.has_uuid(output_uuid)
        assert action.output_uuids == [output_uuid]

        with patch.object(hrec, "flush_cascade_warnings"):
            hireplay.replay_actions(history, [action], prompt=False)

        assert not history.runtime.execution.cascade_warnings
        assert action.is_stale is False
        assert len(panel.objmodel) == object_count
        assert panel.objmodel.has_uuid(output_uuid)
        recreated = panel.objmodel[output_uuid]
        assert get_uuid(recreated) == output_uuid
        assert np.array_equal(recreated.xydata, expected_data)
        assert history.runtime.objects.output_to_action[output_uuid] == action.uuid
        assert history.runtime.objects.action_output_uuids[action.uuid] == [output_uuid]
        parameters = extract_processing_parameters(recreated)
        assert parameters is not None
        assert parameters.source_uuid == source_uuid


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
        # Orphan cleanup: unattended runs never auto-remove orphans; explicit
        # removal purges the surviving output object
        orphan_uuid = action_to_delete.output_uuids[0]
        assert panel.objmodel.has_uuid(orphan_uuid)
        assert not htools.confirm_orphan_removal(history, [(panel, orphan_uuid)])
        htools.remove_orphan_objects(history, [(panel, orphan_uuid)])
        assert not panel.objmodel.has_uuid(orphan_uuid)
        assert len(panel.objmodel) == object_count
        # Leaf deletion: no downstream chain to split, output object survives
        leaf_output_uuid = downstream_action.output_uuids[0]
        select_tree_entry(history, downstream_action.uuid)
        htools.delete_selected(history)
        assert downstream_action not in session.actions
        assert panel.objmodel.has_uuid(leaf_output_uuid)
        # Deleting a producer whose output is already gone yields no orphan
        derivative_output_uuid = derivative.output_uuids[0]
        panel.objview.select_objects([derivative_output_uuid])
        panel.remove_object(force=True)
        assert derivative in session.actions
        # Keep the session alive with a fresh action before splicing the last
        # original one out
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(sips.derivative)
        assert history[len(history)] in session.actions
        object_count_before = len(panel.objmodel)
        select_tree_entry(history, derivative.uuid)
        htools.delete_selected(history)
        assert derivative not in session.actions
        assert len(panel.objmodel) == object_count_before
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


def test_replay_survives_unexpected_recompute_exception() -> None:
    """Contain unexpected exceptions and warn about pattern-less computes."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        acts = list(build_signal_chain(panel, history).actions)
        patternless = HistoryAction(
            title="Legacy compute",
            kind=HistoryAction.KIND_COMPUTE,
            panel_str="signal",
            func_name="mystery",
            pattern=None,
        )
        history.history_sessions[-1].add_action(patternless)
        original = hrec.recompute_compute_in_place

        def flaky(panel_, action_):
            if action_ is acts[1]:
                raise IndexError("boom")
            return original(panel_, action_)

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(hrec, "recompute_compute_in_place", flaky),
            patch.object(hrec, "flush_cascade_warnings") as flush,
        ):
            hireplay.replay_actions(history, [*acts, patternless], prompt=True)

        assert acts[0].is_stale is False
        assert acts[1].is_stale is True  # failed action stays flagged
        assert acts[2].is_stale is True  # downstream blocked by the failure
        flush.assert_called()
        warnings = history.runtime.execution.cascade_warnings
        assert any("boom" in w for w in warnings)
        assert any("mystery" in w for w in warnings)


def test_analysis_effects_manifest_populated_and_recomputed() -> None:
    """Populate the effects manifest by a 1-to-0 analysis and keep it stable."""
    with datalab_test_app_context(console=False, history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.imagepanel
        img = create_peak_image()
        panel.add_object(img)
        det_param = sigima.params.Peak2DDetectionParam.create(
            create_rois=True, threshold=0.5
        )
        with Conf.show_result_dialog.context(False):
            panel.processor.run_feature("peak_detection", det_param)
        action = history[len(history)]
        assert action.effects is not None, "1-to-0 action must carry effects"
        src_uuid = get_uuid(img)
        assert src_uuid in action.effects
        manifest = AnalysisEffects.from_dict(action.effects[src_uuid])
        assert any(
            key.startswith("Geometry_") and key.endswith("_dict")
            for key in manifest.metadata_added
        ), f"Expected a Geometry_*_dict key, got {manifest.metadata_added}"
        assert manifest.roi_modified is True, "Detection ROIs must flag roi_modified"
        # The image had no ROI before the detection: recorded as "" sentinel
        assert manifest.roi_before == ""
        assert img.roi is not None, "Detection must have created ROIs"
        roi_dict_first = numpy_to_json_safe(img.roi.to_dict())
        geometry_key = next(
            key for key in manifest.metadata_added if key.startswith("Geometry_")
        )
        geometry_first = copy.deepcopy(img.metadata[geometry_key])
        added_before = manifest.metadata_added
        # A history recompute keeps first-run keys under metadata_added
        assert hrec.recompute_1_to_0_in_place(history, action) is True
        manifest = AnalysisEffects.from_dict(action.effects[src_uuid])
        assert set(added_before) <= set(manifest.metadata_added), (
            "First-run keys must stay under metadata_added after recompute"
        )
        assert not set(added_before) & set(manifest.metadata_replaced)
        # The recompute restored the pre-analysis ROI (none) and re-ran the
        # detection with first-run semantics: ROIs and results are identical
        # to the first run instead of being detected inside their own ROIs
        assert manifest.roi_before == ""
        assert img.roi is not None, "Recompute must regenerate detection ROIs"
        assert numpy_to_json_safe(img.roi.to_dict()) == roi_dict_first, (
            "Regenerated ROIs must match the first run"
        )
        assert numpy_to_json_safe(img.metadata[geometry_key]) == numpy_to_json_safe(
            geometry_first
        ), "Recomputed detection results must match the first run"


def test_replay_reapplies_user_roi_edits_after_detection_recompute() -> None:
    """Regenerate detection ROIs on recompute, then re-apply user ROI edits."""
    with datalab_test_app_context(console=False, history=True) as win:
        history = win.historypanel
        history.toggle_record_mode(True)
        panel = win.imagepanel
        img = create_peak_image()
        panel.add_object(img)
        det_param = sigima.params.Peak2DDetectionParam.create(
            create_rois=True, threshold=0.5
        )
        with Conf.show_result_dialog.context(False):
            panel.processor.run_feature("peak_detection", det_param)
        detection = history[len(history)]
        assert img.roi is not None
        # User deletes the detection ROIs: recorded as a mutation entry
        panel.objview.select_objects([get_uuid(img)])
        panel.processor.delete_regions_of_interest()
        assert img.roi is None
        mutation = history[len(history)]
        assert mutation.kind == HistoryAction.KIND_MUTATION
        # Recomputing the detection regenerates the ROIs from the recorded
        # pre-analysis state (instead of re-detecting inside the ROIs)...
        assert hrec.recompute_1_to_0_in_place(history, detection) is True
        assert img.roi is not None, "Detection recompute must regenerate ROIs"
        # ...and replaying the recorded sequence re-applies the user's edit
        mutation.replay(win, restore_selection=True, edit=False)
        assert img.roi is None, "Replay must re-apply the user's ROI edit"


def test_roi_mutation_recording_replay_and_partial_targets() -> None:
    """Record paste/delete ROI mutations, replay them, tolerate deleted targets."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        sig1 = create_paracetamol_signal(SIZE)
        sig1.roi = create_signal_roi([SROI1, SROI2], indices=True)
        panel.add_object(sig1)
        sig2 = create_paracetamol_signal(SIZE)
        panel.add_object(sig2)
        sig2 = panel.objmodel[get_uuid(sig2)]
        sig3 = create_paracetamol_signal(SIZE)
        panel.add_object(sig3)
        sig3 = panel.objmodel[get_uuid(sig3)]
        # Paste onto two targets: one mutation entry per object carrying the
        # post-combination ROI payload
        panel.objview.select_objects([1])
        panel.copy_roi()
        panel.objview.select_objects([2, 3])
        panel.paste_roi()
        actions = history.history_sessions[-1].actions
        paste2, paste3 = actions[-2], actions[-1]
        for action, sig in ((paste2, sig2), (paste3, sig3)):
            assert action.kind == HistoryAction.KIND_MUTATION
            assert action.mutation_key == "roi"
            assert action.target_uuids == [get_uuid(sig)]
            payload = action.kwargs.get("payload")
            assert payload is not None
            assert numpy_to_json_safe(payload.to_dict()) == numpy_to_json_safe(
                sig.roi.to_dict()
            )
        # Direct replay re-applies the payload after the ROI was cleared
        sig2.roi = None
        paste2.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is not None
        assert numpy_to_json_safe(sig2.roi.to_dict()) == numpy_to_json_safe(
            paste2.kwargs["payload"].to_dict()
        )
        # Deleting ROIs records one empty-payload mutation for both targets
        panel.objview.select_objects([2, 3])
        panel.processor.delete_regions_of_interest()
        assert sig2.roi is None and sig3.roi is None
        delete_action = history.history_sessions[-1].actions[-1]
        assert delete_action.kind == HistoryAction.KIND_MUTATION
        assert delete_action.mutation_key == "roi"
        assert delete_action.kwargs.get("payload") is None
        assert set(delete_action.target_uuids) == {get_uuid(sig2), get_uuid(sig3)}
        # Replaying the recorded sequence restores then removes the ROI
        paste2.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is not None
        delete_action.replay(win, restore_selection=True, edit=False)
        assert sig2.roi is None
        # Cascade recompute tolerates a deleted target: warn and apply to the rest
        sig2.roi = create_signal_roi([SROI1], indices=True)
        panel.objview.select_objects([get_uuid(sig3)])
        panel.remove_object(force=True)
        assert hrec.recompute_mutation_in_place(history, delete_action) is True
        assert sig2.roi is None
        assert any(
            "deleted" in warning
            for warning in history.runtime.execution.cascade_warnings
        )
        history.runtime.execution.cascade_warnings.clear()


def test_cascade_reapplies_roi_mutation() -> None:
    """Cascade recompute re-applies or blocks a downstream ROI mutation."""
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
        # A failed upstream recompute blocks the deferred mutation replay
        output.roi = None
        with (
            patch.object(hrec, "recompute_action_in_place", return_value=False),
            patch.object(hrec, "flush_cascade_warnings"),
        ):
            hireplay.replay_actions(
                history, [compute_action, mutation_action], prompt=False
            )
        assert output.roi is None
        assert compute_action.is_stale is True
        compute_action.is_stale = False
        history.runtime.execution.cascade_warnings.clear()

        # Edit mode is refused for ROI mutations: the payload is kept as is
        # and no downstream cascade is triggered
        output.roi = None
        payload_before = mutation_action.kwargs["payload"]
        with patch.object(hrec, "recompute_cascade") as cascade:
            hireplay.replay_actions(history, [mutation_action], prompt=True)
        cascade.assert_not_called()
        assert mutation_action.kwargs["payload"] is payload_before
        assert output.roi is not None
        assert numpy_to_json_safe(output.roi.to_dict()) == numpy_to_json_safe(
            payload_before.to_dict()
        )


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


def test_edited_image_creation_recomputes_downstream_in_place() -> None:
    """Regenerate an edited image before recomputing its existing descendant."""
    initial_param = Gauss2DParam.create(
        title="Initial Gaussian",
        height=24,
        width=28,
        x0=-4.0,
        y0=2.0,
        sigma=1.2,
        a=25.0,
    )
    edited_param = Gauss2DParam.create(
        title="Edited Gaussian",
        height=24,
        width=28,
        x0=3.0,
        y0=-2.0,
        sigma=3.5,
        a=80.0,
    )
    expected_source = create_image_from_param(edited_param)
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.imagepanel
        history.toggle_record_mode(True)
        source = create_image_from_param(initial_param)
        source_uuid = get_uuid(source)
        panel.add_object(source)
        source = panel.objmodel[source_uuid]
        source_identity = id(source)
        initial_source_data = source.data.copy()
        # Synthetic creation action carrying the edited parameters (the
        # recorded output is the pre-existing source object)
        creation_action = HistoryAction(
            title="Create edited Gaussian",
            kind=HistoryAction.KIND_UI,
            target="imagepanel",
            method_name="new_object",
            kwargs={"param": edited_param},
        )
        creation_action.output_uuids = [source_uuid]
        # Record the real downstream compute through the UI path
        panel.objview.select_objects([source_uuid])
        panel.processor.run_feature(
            "gaussian_filter", sigima.params.GaussianParam.create(sigma=2.0)
        )
        downstream_action = history[len(history)]
        downstream_uuid = downstream_action.output_uuids[0]
        downstream = panel.objmodel[downstream_uuid]
        downstream_identity = id(downstream)
        initial_downstream_data = downstream.data.copy()

        with patch.object(
            hrec, "create_image_from_param", wraps=create_image_from_param
        ) as create_image_mock:
            assert hrec.recompute_creation_in_place(history, creation_action)
            assert hrec.recompute_compute_in_place(history, downstream_action)

        create_image_mock.assert_called_once()
        assert create_image_mock.call_args.args[0] is edited_param
        assert not history.runtime.execution.cascade_warnings

        assert panel.objmodel[source_uuid] is source
        assert id(source) == source_identity
        np.testing.assert_allclose(source.data, expected_source.data)
        assert not np.array_equal(source.data, initial_source_data)
        creation_param = extract_creation_parameters(source)
        assert isinstance(creation_param, Gauss2DParam)
        for name in ("height", "width", "x0", "y0", "sigma", "a"):
            assert getattr(creation_param, name) == getattr(edited_param, name)

        assert panel.objmodel[downstream_uuid] is downstream
        assert id(downstream) == downstream_identity
        assert not np.array_equal(downstream.data, initial_downstream_data)
        downstream_params = extract_processing_parameters(downstream)
        assert downstream_params is not None
        assert downstream_params.source_uuid == source_uuid


def test_interactive_fit_records_replays_in_place_and_refuses_edit() -> None:
    """Record an interactive fit, replay it in place, refuse edit mode."""

    def fake_gaussian_dialog(x, y, parent=None):
        """Deterministic stand-in for the interactive Gaussian fit dialog."""
        del parent, y
        fit_params = signal_fitting.create_fit_params(
            "gaussian",
            {"amplitude": 1200.0, "sigma": 20.0, "x0": 100.0, "y0": 5.0},
            interactive=True,
        )
        return signal_fitting.evaluate_fit(x, **fit_params), [], fit_params

    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        panel.processor.compute_fit("Gaussian fit", fake_gaussian_dialog)
        # The fit is recorded as a replayable UI action with its output UUID
        action = history[len(history)]
        assert action.kind == HistoryAction.KIND_UI
        assert action.target == "signalprocessor"
        assert action.method_name == "recompute_fit"
        fit_params = action.kwargs["fit_params"]
        assert fit_params["fit_type"] == "gaussian"
        assert len(action.output_uuids) == 1
        fitted_uuid = action.output_uuids[0]
        assert action.kwargs["output_uuid"] == fitted_uuid
        fitted = panel.objmodel[fitted_uuid]
        expected_y = fitted.y.copy()
        object_count = len(panel.objmodel)
        # Headless replay updates the fitted curve in place (no duplicate)
        fitted.set_xydata(fitted.x, np.zeros_like(fitted.y))
        action.replay(win, restore_selection=True, edit=False)
        assert len(panel.objmodel) == object_count
        assert panel.objmodel[fitted_uuid] is fitted
        np.testing.assert_allclose(fitted.y, expected_y)
        assert fitted.metadata["fit_params"] == fit_params
        # Edit mode is refused: the recorded fit is preserved untouched
        fitted.set_xydata(fitted.x, np.full_like(fitted.y, 3.0))
        action.replay(win, restore_selection=True, edit=True)
        assert len(panel.objmodel) == object_count
        np.testing.assert_allclose(fitted.y, 3.0)
        assert action.kwargs["fit_params"] == fit_params
        # A deleted output is re-created under its recorded UUID at replay
        panel.objview.select_objects([fitted_uuid])
        panel.remove_object(force=True)
        assert not panel.objmodel.has_uuid(fitted_uuid)
        action.replay(win, restore_selection=True, edit=False)
        assert panel.objmodel.has_uuid(fitted_uuid)
        np.testing.assert_allclose(panel.objmodel[fitted_uuid].y, expected_y)
        # The nested fit parameters survive a .dlhist round-trip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fit.dlhist")
            assert history.save_to_dlhist_file(path)
            sessions = read_history_sessions(path, history.H5_PREFIX)
        restored = next(
            restored_action
            for session in sessions
            for restored_action in session.actions
            if restored_action.uuid == action.uuid
        )
        assert restored.method_name == "recompute_fit"
        assert restored.kwargs["fit_params"] == fit_params
        assert restored.kwargs["output_uuid"] == fitted_uuid
