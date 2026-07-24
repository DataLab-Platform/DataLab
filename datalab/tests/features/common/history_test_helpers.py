# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Typed builders and selectors for History panel tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import sigima.params
import sigima.proc.signal as sips
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.tests.data import create_paracetamol_signal

from datalab.h5.native import NativeH5Reader
from datalab.history.action import HistoryAction
from datalab.history.replaymap import ReplayUuidMap
from datalab.history.session import HistorySession
from datalab.history.workspace_state import WorkspaceState

if TYPE_CHECKING:
    from datalab.gui.panel.history import HistoryPanel
    from datalab.gui.panel.signal import SignalPanel


@dataclass(frozen=True)
class SignalChain:
    """Objects and actions produced by a three-step signal chain."""

    actions: tuple[HistoryAction, HistoryAction, HistoryAction]
    outputs: tuple[Any, Any, Any]


class ReplayObjectModel:
    """Minimal ordered object model used by replay-map tests."""

    def __init__(self, objects: list[tuple[str, str]]) -> None:
        self.objects = {uuid: SimpleNamespace(title=title) for uuid, title in objects}

    def __getitem__(self, uuid: str) -> SimpleNamespace:
        """Return an object by UUID."""
        return self.objects[uuid]

    def get_object_ids(self) -> list[str]:
        """Return object UUIDs in panel order."""
        return list(self.objects)

    def add(self, uuid: str, title: str) -> None:
        """Add an object to the model."""
        self.objects[uuid] = SimpleNamespace(title=title)

    def remove(self, uuid: str) -> None:
        """Remove an object from the model."""
        self.objects.pop(uuid)


def build_replay_map(
    signal_objects: list[tuple[str, str]],
    image_objects: list[tuple[str, str]] | None = None,
) -> tuple[ReplayUuidMap, ReplayObjectModel, ReplayObjectModel]:
    """Build a replay map backed by minimal signal and image panels."""
    signal_model = ReplayObjectModel(signal_objects)
    image_model = ReplayObjectModel(image_objects or [])
    signal_panel = SimpleNamespace(PANEL_STR_ID="signal", objmodel=signal_model)
    image_panel = SimpleNamespace(PANEL_STR_ID="image", objmodel=image_model)
    panels = cast(Any, (signal_panel, image_panel))
    return ReplayUuidMap(panels), signal_model, image_model


def build_workspace_state(
    selection: list[str], titles: list[str] | None = None
) -> WorkspaceState:
    """Build a signal workspace state with stable object metadata."""
    state = WorkspaceState()
    state.selection = {"signal": selection}
    state.states = {"signal": ["(10,)"] * len(selection)}
    state.titles = {"signal": titles or [f"Object {index}" for index in selection]}
    state.object_metadata = {
        "signal": {
            uuid: {"shape": [10], "ndim": 1, "title": title}
            for uuid, title in zip(selection, state.titles["signal"])
        }
    }
    return state


def build_history_action() -> HistoryAction:
    """Build a serializable compute action containing every UUID reference."""
    action = HistoryAction(
        title="Difference",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        func_name="difference",
        pattern="2_to_1",
        kwargs={"obj2_uuids": ["second-uuid"], "pairwise": False},
        state=build_workspace_state(["source-uuid"], ["Source"]),
    )
    action.output_uuids = ["output-uuid"]
    return action


def add_paracetamol_signals(panel: SignalPanel, count: int) -> list[str]:
    """Add paracetamol signals and return their UUIDs in panel order."""
    for _index in range(count):
        panel.add_object(create_paracetamol_signal())
    return panel.objmodel.get_object_ids()[-count:]


def build_signal_chain(panel: SignalPanel, history: HistoryPanel) -> SignalChain:
    """Build Gaussian, derivative and moving-average processing outputs."""
    add_paracetamol_signals(panel, 1)
    panel.objview.select_objects([1])
    panel.processor.run_feature(
        sips.gaussian_filter, sigima.params.GaussianParam.create(sigma=1.5)
    )
    first_action = history[len(history)]
    first_output = panel.objmodel.get_object_from_number(2)
    panel.objview.select_objects([2])
    panel.processor.run_feature(sips.derivative)
    second_action = history[len(history)]
    second_output = panel.objmodel.get_object_from_number(3)
    panel.objview.select_objects([3])
    parameter = sigima.params.MovingAverageParam.create(n=3)
    panel.processor.run_feature(sips.moving_average, parameter)
    third_action = history[len(history)]
    third_output = panel.objmodel.get_object_from_number(4)
    return SignalChain(
        (first_action, second_action, third_action),
        (first_output, second_output, third_output),
    )


def read_history_sessions(
    path: str, section: str = "history_session"
) -> list[HistorySession]:
    """Read serialized history sessions from a file."""
    with NativeH5Reader(path) as reader:
        return reader.read_object_list(section, HistorySession)


def delete_hdf5_items_by_name(group: Any, item_name: str) -> None:
    """Delete HDF5 attributes and groups with a name recursively."""
    if item_name in group.attrs:
        del group.attrs[item_name]
    if not hasattr(group, "keys"):
        return
    for key in list(group.keys()):
        if key == item_name:
            del group[key]
        else:
            delete_hdf5_items_by_name(group[key], item_name)


def get_tree_item(history: HistoryPanel, uuid: str) -> QW.QTreeWidgetItem:
    """Return the tree item identified by an entry UUID."""
    iterator = QW.QTreeWidgetItemIterator(history.tree)
    while iterator.value():
        item = iterator.value()
        if item.data(0, QC.Qt.UserRole) == uuid:
            return item
        iterator += 1
    raise LookupError(uuid)


def select_tree_entry(history: HistoryPanel, uuid: str) -> None:
    """Select the tree item identified by an entry UUID."""
    item = get_tree_item(history, uuid)
    history.tree.clearSelection()
    history.tree.setCurrentItem(item)
    item.setSelected(True)


def select_tree_session(history: HistoryPanel, session: HistorySession) -> None:
    """Select a session's top-level tree item."""
    item = history.tree.topLevelItem(history.history_sessions.index(session))
    history.tree.clearSelection()
    history.tree.setCurrentItem(item)
    item.setSelected(True)


def is_session_bold(history: HistoryPanel, session: HistorySession) -> bool:
    """Return whether a session's tree item uses a bold font."""
    item = history.tree.topLevelItem(history.history_sessions.index(session))
    return item is not None and item.font(0).bold()
