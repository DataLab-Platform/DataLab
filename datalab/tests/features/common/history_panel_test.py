# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""History panel replay and cross-panel navigation contracts."""

from __future__ import annotations

import sigima.proc.image as sipi
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.tests.data import create_paracetamol_signal, create_sincos_image

from datalab.gui.panel.history import HistoryTree
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import (
    build_signal_chain,
    get_tree_item,
    is_session_bold,
    select_tree_entry,
)


def test_panel_replay_restores_selection_without_outputs() -> None:
    """Use panel replay to restore selection without recording or new output."""
    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.signalpanel
        history.toggle_record_mode(True)
        panel.add_object(create_paracetamol_signal())
        source_uuid = get_uuid(panel.objmodel.get_object_from_number(1))
        panel.objview.select_objects([1])
        panel.processor.run_feature(sips.derivative)
        action = history[len(history)]
        output_uuid = action.output_uuids[0]
        select_tree_entry(history, action.uuid)
        assert panel.objview.get_sel_object_uuids() == [output_uuid]
        object_count, action_count = len(panel.objmodel), len(history)
        history.replay_restore_actions(replay=True, restore_selection=True)
        assert (
            len(panel.objmodel),
            len(history),
            panel.objview.get_sel_object_uuids(),
        ) == (object_count, action_count, [source_uuid])
        assert action in history.history_sessions[-1].actions and (
            history.runtime.objects.action_output_uuids[action.uuid] == [output_uuid]
        )
        assert (
            output_uuid in panel.objmodel.get_object_ids()
            and history.runtime.objects.output_to_action[output_uuid] == action.uuid
        )
        panel.objview.select_objects([output_uuid])
        panel.remove_object(force=True)
        assert (
            action in history.history_sessions[-1].actions
            and output_uuid not in panel.objmodel.get_object_ids()
            and action.uuid not in history.runtime.objects.action_output_uuids
            and output_uuid not in history.runtime.objects.output_to_action
        )
        select_tree_entry(history, action.uuid)
        history.replay_restore_actions(replay=False, restore_selection=True)
        assert panel.objview.get_sel_object_uuids() == [source_uuid]


def test_cross_panel_sessions_navigation_and_tree_state() -> None:
    """Coordinate active sessions, navigation, tree state and selection fallback."""
    with datalab_test_app_context(history=True) as win:
        history = win.historypanel
        signal_panel, image_panel = win.signalpanel, win.imagepanel
        history.toggle_record_mode(True)
        signal_chain = build_signal_chain(signal_panel, history)
        first_signal_action, middle_signal_action, last_signal_action = (
            signal_chain.actions
        )
        signal_uuid = first_signal_action.state.selection["signal"][0]
        signal_session = next(
            session
            for session in history.history_sessions
            if first_signal_action in session.actions
        )
        assert all(action in signal_session.actions for action in signal_chain.actions)
        navigation_states = []
        select_tree_entry(history, first_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        select_tree_entry(history, middle_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        select_tree_entry(history, last_signal_action.uuid)
        navigation_states.append(
            (
                history.ui.actions["step_prev"].isEnabled(),
                history.ui.actions["step_next"].isEnabled(),
            )
        )
        assert navigation_states == [(False, True), (True, True), (True, False)]
        image_panel.add_object(create_sincos_image())
        image_panel.objview.select_objects([1])
        image_panel.processor.run_feature(sipi.inverse)
        image_action = history[len(history)]
        image_session = history.navigation.get_active_session("image")
        assert (
            history.navigation.get_active_session("signal") is signal_session
            and image_session is not None
            and signal_session is not image_session
        )
        bold_before = (
            is_session_bold(history, signal_session),
            is_session_bold(history, image_session),
        )
        history.tree.populate_tree(history.history_sessions)
        assert bold_before == (True, True) and (
            is_session_bold(history, signal_session),
            is_session_bold(history, image_session),
        ) == (True, True)
        output_uuid = first_signal_action.output_uuids[0]
        select_tree_entry(history, first_signal_action.uuid)
        assert signal_panel.objview.get_sel_object_uuids() == [output_uuid]
        signal_panel.objview.select_objects([output_uuid])
        signal_panel.remove_object(force=True)
        history.toggle_record_mode(False)
        image_panel.objview.select_objects([1])
        image_panel.remove_object(force=True)
        history.refresh_compatibility_items()
        tree_action_uuids = set()
        iterator = QW.QTreeWidgetItemIterator(history.tree)
        while iterator.value():
            uuid = iterator.value().data(0, QC.Qt.UserRole)
            if uuid is not None:
                tree_action_uuids.add(uuid)
            iterator += 1
        image_item = get_tree_item(history, image_action.uuid)
        assert (
            all(
                first_signal_action not in session.actions
                for session in history.history_sessions
            )
            and middle_signal_action in signal_session.actions
            and last_signal_action in signal_session.actions
            and middle_signal_action.state.selection["signal"] == [signal_uuid]
            and first_signal_action.uuid
            not in history.runtime.objects.action_output_uuids
            and output_uuid not in history.runtime.objects.output_to_action
            and first_signal_action.uuid not in tree_action_uuids
            and {middle_signal_action.uuid, last_signal_action.uuid}.issubset(
                tree_action_uuids
            )
            and image_item.data(0, HistoryTree.COMPATIBILITY_ROLE) is False
            and image_item.foreground(0).color().isValid()
            and image_item.data(0, QC.Qt.UserRole) == image_action.uuid
        )
