# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Application workflow contracts for the History panel."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sigima.params
import sigima.proc.signal as sips
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata.common import ResultData
from datalab.gui import historytools_ops as htools
from datalab.gui.panel.history import HistoryAction
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
from datalab.gui.processor.catcher import CompOut
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


def test_replay_resolves_feature_with_plugin_origin_and_paramclass() -> None:
    """Resolve replayed features with their plugin and parameter identities."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        param = sigima.params.GaussianParam.create(sigma=1.5)
        panel.processor.run_feature(sips.gaussian_filter, param)
        action = history[len(history)]
        plugin_origin = {
            "module": "test_plugin.operations",
            "directory": "test_plugin",
        }
        action.plugin_origin = plugin_origin

        with patch.object(
            panel.processor, "get_feature", wraps=panel.processor.get_feature
        ) as get_feature:
            action.replay(win, restore_selection=True, edit=False)

        get_feature.assert_called_once_with(
            action.func_name,
            plugin_origin=plugin_origin,
            paramclass_name=type(param).__name__,
        )


def test_replay_1_to_n_resolves_feature_with_first_paramclass() -> None:
    """Resolve a 1-to-n feature with the first stored parameter identity."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        add_paracetamol_signals(panel, 1)
        panel.objview.select_objects([1])
        params = [sigima.params.GaussianParam.create(sigma=1.5)]
        action = HistoryAction()
        action.kind = HistoryAction.KIND_COMPUTE
        action.pattern = "1_to_n"
        action.target = "signalpanel"
        action.panel_str = "signal"
        action.func_name = sips.gaussian_filter.__name__
        action.kwargs = {"params": params}

        with (
            patch.object(
                panel.processor, "get_feature", return_value=sips.gaussian_filter
            ) as get_feature,
            patch.object(panel.processor, "run_feature"),
        ):
            action.replay_compute(win, edit=False)

        get_feature.assert_called_once_with(
            action.func_name,
            plugin_origin=None,
            paramclass_name=type(params[0]).__name__,
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
        original_recompute = panel.processor.recompute_1_to_1
        call_count = 0

        def fail_second_recompute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return CompOut(error_msg="expected cascade failure")
            return original_recompute(*args, **kwargs)

        with (
            patch.object(
                panel.processor,
                "recompute_1_to_1",
                side_effect=fail_second_recompute,
            ),
            patch.object(panel.processor, "recompute_1_to_0") as recompute_analysis,
        ):
            history.recompute_cascade(first_action)

        recompute_analysis.assert_not_called()
        assert first_action.is_stale is False
        assert failed_action.is_stale is True
        assert analysis_action.is_stale is True
        assert np.array_equal(failed_output.xydata, failed_data)


def test_multi_action_edit_recomputes_selected_descendants_once() -> None:
    """Recompute selected ancestors and their analysis descendant exactly once."""
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
            hireplay.edit_mode_replay_actions(history, selected)

        assert [call.args[1] for call in prompt.call_args_list] == selected
        assert [call.args[1] for call in recompute.call_args_list] == expected
        assert all(action.is_stale is False for action in expected)


def test_multi_action_edit_cascades_across_independent_sessions() -> None:
    """Recompute edited branches from multiple sessions in global order."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        first_chain = build_independent_signal_branch(panel, history)
        history.create_new_session(panel_str="signal")
        second_chain = build_independent_signal_branch(panel, history)
        selected = [second_chain[0], first_chain[0]]
        expected = [*first_chain, *second_chain]

        with (
            patch.object(hireplay, "prompt_edit_action_params", return_value=True),
            patch.object(
                hrec, "recompute_action_in_place", return_value=True
            ) as recompute,
        ):
            hireplay.edit_mode_replay_actions(history, selected)

        assert [call.args[1] for call in recompute.call_args_list] == expected
        assert all(action.is_stale is False for action in expected)


def test_multi_action_edit_failure_skips_dependents_and_continues() -> None:
    """Leave a failed branch stale while recomputing an independent session."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        history.toggle_edit_mode(True)
        failed_chain = build_independent_signal_branch(panel, history)
        history.create_new_session(panel_str="signal")
        successful_chain = build_independent_signal_branch(panel, history)
        failed_root = failed_chain[0]
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
            hireplay.edit_mode_replay_actions(
                history, [failed_root, successful_chain[0]]
            )

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
            hireplay.edit_mode_replay_actions(history, [first_action, second_action])

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
            hireplay.edit_mode_replay_actions(history, [ui_action, compute_action])

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
            hireplay.edit_mode_replay_actions(history, [first_compute, ui_action])

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
            hireplay.edit_mode_replay_actions(history, [action])

        flush.assert_called_once_with(history)
        assert history.runtime.execution.cascade_warnings == []


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
    """Stage every pairwise result before mutating existing outputs."""
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
        staged_result = outputs[0].copy()
        staged_result.xydata = staged_result.xydata * 0.0

        with patch.object(
            panel.processor,
            "recompute_2_to_1",
            side_effect=[staged_result, None],
        ):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        for output, data in zip(outputs, original_data):
            assert np.array_equal(output.xydata, data)


def test_2_to_1_refresh_failure_rolls_back_and_resyncs_outputs() -> None:
    """Refresh only after commit and resync every target after rollback."""
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
        identities = [id(obj) for obj in outputs]
        original_titles = [obj.title for obj in outputs]
        original_data = [obj.xydata.copy() for obj in outputs]
        original_metadata = [obj.metadata.copy() for obj in outputs]
        original_sources = [
            extract_processing_parameters(obj).source_uuids for obj in outputs
        ]
        staged_results = [obj.copy() for obj in outputs]
        staged_titles = []
        for index, result in enumerate(staged_results):
            result.title = f"staged-{index}"
            staged_titles.append(result.title)
            result.xydata = result.xydata * 0.0
        refresh_effects = []

        def refresh_with_failure(_panel, output_uuid):
            refresh_effects.append(
                (
                    output_uuid,
                    [obj.title for obj in outputs],
                    [
                        extract_processing_parameters(obj).source_uuids
                        for obj in outputs
                    ],
                )
            )
            if len(refresh_effects) in (2, 3):
                raise RuntimeError(f"refresh failed #{len(refresh_effects)}")

        with (
            patch.object(
                panel.processor,
                "recompute_2_to_1",
                side_effect=staged_results,
            ),
            patch.object(
                hrec,
                "refresh_target",
                side_effect=refresh_with_failure,
            ),
        ):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        assert [effect[0] for effect in refresh_effects] == [
            action.output_uuids[0],
            action.output_uuids[1],
            action.output_uuids[0],
            action.output_uuids[1],
        ]
        assert refresh_effects[0][1] == staged_titles
        assert refresh_effects[0][2] == original_sources
        assert refresh_effects[2][1] == original_titles
        for index, output in enumerate(outputs):
            assert id(output) == identities[index]
            assert output.title == original_titles[index]
            assert np.array_equal(output.xydata, original_data[index])
            assert output.metadata == original_metadata[index]


def test_1_to_n_refresh_failure_rolls_back_and_resyncs_outputs() -> None:
    """Commit all 1-to-n outputs before refresh and fully roll back on failure."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        source_uuid = add_paracetamol_signals(panel, 1)[0]
        actions = []
        for _index in range(2):
            panel.objview.select_objects([source_uuid])
            panel.processor.run_feature(sips.derivative)
            actions.append(history[len(history)])
        action = actions[0]
        action.pattern = "1_to_n"
        action.kwargs = {
            "params": [
                sigima.params.GaussianParam.create(sigma=1.5),
                sigima.params.GaussianParam.create(sigma=2.5),
            ]
        }
        action.output_uuids.extend(actions[1].output_uuids)
        history.runtime.objects.action_output_uuids[action.uuid] = list(
            action.output_uuids
        )
        outputs = [panel.objmodel[uuid] for uuid in action.output_uuids]
        identities = [id(obj) for obj in outputs]
        original_titles = [obj.title for obj in outputs]
        original_data = [obj.xydata.copy() for obj in outputs]
        original_metadata = [obj.metadata.copy() for obj in outputs]
        staged_results = [obj.copy() for obj in outputs]
        staged_titles = []
        for index, result in enumerate(staged_results):
            result.title = f"staged-{index}"
            staged_titles.append(result.title)
            result.xydata = result.xydata * 0.0
        refresh_effects = []

        def refresh_with_failure(_panel, output_uuid):
            refresh_effects.append((output_uuid, [obj.title for obj in outputs]))
            if len(refresh_effects) in (2, 3):
                raise RuntimeError(f"refresh failed #{len(refresh_effects)}")

        with (
            patch.object(
                panel.processor,
                "recompute_1_to_n",
                return_value=staged_results,
            ),
            patch.object(hrec, "refresh_target", side_effect=refresh_with_failure),
        ):
            success = hrec.recompute_action_in_place(history, action)

        assert success is False
        assert [effect[0] for effect in refresh_effects] == [
            action.output_uuids[0],
            action.output_uuids[1],
            action.output_uuids[0],
            action.output_uuids[1],
        ]
        assert refresh_effects[0][1] == staged_titles
        assert refresh_effects[2][1] == original_titles
        for index, output in enumerate(outputs):
            assert id(output) == identities[index]
            assert output.title == original_titles[index]
            assert np.array_equal(output.xydata, original_data[index])
            assert output.metadata == original_metadata[index]


def test_1_to_0_failure_rolls_back_all_source_metadata() -> None:
    """Restore every analysis source when a later recomputation fails."""
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

        def fail_second_analysis(_func_name, source, _param, plugin_origin=None):
            del plugin_origin
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
        action.func_name = "analysis"
        action.kwargs = {"param": param}
        action.state.selection = {panel.PANEL_STR_ID: [source_uuid]}

        with patch.object(
            panel.processor, "recompute_1_to_0", return_value=True
        ) as recompute:
            success = hrec.recompute_action_in_place(history, action)

        assert success is True
        passed_param = recompute.call_args.args[2]
        assert passed_param is not param
        assert passed_param.create_rois is False
        assert action.kwargs["param"].create_rois is True


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
