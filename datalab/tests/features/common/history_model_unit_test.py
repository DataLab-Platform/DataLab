# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for history persistence, copying and recompute."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sigima.objects import Gauss2DParam, ImageObj

from datalab.gui import historysession_ops as hsess
from datalab.gui import historytools_ops as hops
from datalab.gui.creation import (
    create_image_from_param,
    extract_creation_parameters,
)
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.panel.history import runtime as hruntime
from datalab.gui.panel.history.ui import HistoryPanelUI
from datalab.gui.processor.base import (
    BaseProcessor,
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import HISTORY_ACTION_SCHEMA_VERSION, HISTORY_SCHEMA_VERSION
from datalab.history.session import HistorySession
from datalab.history.workspace_state import WorkspaceState
from datalab.objectmodel import get_uuid, set_uuid
from datalab.tests.features.common.history_test_helpers import (
    build_history_action,
    delete_hdf5_items_by_name,
    read_history_sessions,
)


class CascadeObjectModel:
    """Minimal object model for pure cascade recomputation tests."""

    def __init__(self, objects: list[ImageObj]) -> None:
        self.objects = {get_uuid(obj): obj for obj in objects}

    def __getitem__(self, uuid: str) -> ImageObj:
        """Return the image identified by ``uuid``."""
        return self.objects[uuid]

    def has_uuid(self, uuid: str) -> bool:
        """Return whether ``uuid`` exists in the model."""
        return uuid in self.objects

    def get_object_ids(self) -> list[str]:
        """Return all object UUIDs in insertion order."""
        return list(self.objects)


class PromptExecution:
    """Minimal execution state for input-session prompt tests."""

    def __init__(self, prompt_allowed: bool = True) -> None:
        self.suppress_session_prompt = False
        self.prompt_allowed = prompt_allowed
        self.prompt_count = 0

    def start_session_input_prompt(self) -> bool:
        """Record that the debounce guard was reached."""
        self.prompt_count += 1
        return self.prompt_allowed

    @contextmanager
    def session_prompt_suppressed(self):
        """Suppress session prompts while preserving the previous state."""
        previous = self.suppress_session_prompt
        self.suppress_session_prompt = True
        try:
            yield
        finally:
            self.suppress_session_prompt = previous


class PromptNavigation:
    """Minimal single active-session registry for prompt tests."""

    def __init__(self, active_session: HistorySession | None) -> None:
        self.active_session = active_session

    def current_panel_str(self) -> str:
        """Return the fallback panel used for per-action panel resolution."""
        return "signal"

    def get_active_session(self) -> HistorySession | None:
        """Return the single active recording session."""
        return self.active_session


class PromptTree:
    """Minimal tree recorder for production history routing."""

    def __init__(self) -> None:
        self.rebuilt_session_indices: list[int] = []
        self.rearrange_count = 0

    def rebuild_session(self, session_index: int) -> None:
        """Record the rebuilt session index."""
        self.rebuilt_session_indices.append(session_index)

    def rearrange_tree(self) -> None:
        """Record that tree layout was refreshed."""
        self.rearrange_count += 1


class PromptUI:
    """Minimal UI state recorder for production history routing."""

    def __init__(self) -> None:
        self.update_count = 0

    def update_actions_state(self) -> None:
        """Record that action states were refreshed."""
        self.update_count += 1


class PromptPanel:
    """Minimal history panel for pure input-session prompt tests."""

    def __init__(
        self,
        sessions: list[HistorySession] | None = None,
        prompt_allowed: bool = True,
    ) -> None:
        self.record_mode_enabled = True
        self.history_sessions = list(sessions or [])
        active = self.history_sessions[-1] if self.history_sessions else None
        self.navigation = PromptNavigation(active)
        self.runtime = SimpleNamespace(execution=PromptExecution(prompt_allowed))
        self.tree = PromptTree()
        self.ui = PromptUI()
        self.created_sessions: list[HistorySession] = []
        self.prompt_behaviors: list[hsess.SessionBehavior | None] = []
        self.prompt_suppressed_states: list[bool] = []
        self.added_actions: list[HistoryAction] = []
        self.registered_outputs: list[tuple[HistoryAction, list[str]]] = []
        self.compatibility_refresh_count = 0
        self.events: list[str] = []

    def is_replaying(self) -> bool:
        """Return whether history replay is active."""
        return False

    def create_new_session(self) -> HistorySession:
        """Create and activate a session without constructing GUI objects."""
        session = HistorySession(number=len(self.history_sessions) + 1)
        self.history_sessions.append(session)
        self.navigation.active_session = session
        self.created_sessions.append(session)
        return session

    def maybe_start_session_for_input(
        self,
        *,
        load: bool = False,
        behavior: hsess.SessionBehavior | None = None,
    ) -> bool:
        """Forward to the session operation while recording the evaluation."""
        self.events.append("session_decision")
        self.prompt_behaviors.append(behavior)
        self.prompt_suppressed_states.append(
            self.runtime.execution.suppress_session_prompt
        )
        return hsess.maybe_start_session_for_input(self, load=load, behavior=behavior)

    @contextmanager
    def session_prompt_suppressed(self):
        """Suppress input-session prompts for the context scope."""
        with self.runtime.execution.session_prompt_suppressed():
            yield

    def add_ui_entry(
        self,
        action_title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
        **kwargs,
    ) -> HistoryAction | None:
        """Forward a UI entry through the production history operation."""
        return hsess.add_ui_entry(
            self, action_title, target, method_name, save_state, **kwargs
        )

    def register_action_outputs(
        self, action: HistoryAction, output_uuids: list[str]
    ) -> None:
        """Record output registration performed by the main window."""
        self.registered_outputs.append((action, output_uuids))

    def add_object(self, action: HistoryAction) -> None:
        """Route an action through the production session operation."""
        self.added_actions.append(action)
        hsess.add_object(self, action)

    def refresh_compatibility_items(self) -> None:
        """Record that compatibility state was refreshed."""
        self.compatibility_refresh_count += 1


def make_prompt_session(panel_str: str, *, populated: bool) -> HistorySession:
    """Return a session optionally populated with one panel action."""
    session = HistorySession(number=1)
    if populated:
        session.add_action(HistoryAction(panel_str=panel_str))
    return session


def test_history_session_default_and_explicit_titles() -> None:
    """Translate only the default title and preserve explicit titles."""
    with patch("datalab.history.session._", return_value="Traitement") as translate:
        default_session = HistorySession(number=7)
        explicit_session = HistorySession(title="Acquisition", number=8)
    assert default_session.title == "Traitement"
    assert default_session.number == 7
    assert explicit_session.title == "Acquisition"
    assert explicit_session.number == 8
    translate.assert_called_once_with("Processing")


def test_ui_action_description_is_empty_when_callable_is_unresolved() -> None:
    """Return no fallback description when UI callable resolution returns None."""
    action = HistoryAction(kind=HistoryAction.KIND_UI)

    with patch.object(action, "resolve_callable", return_value=None):
        assert action.description == ""


def test_compute_n_to_1_uses_provided_history_title() -> None:
    """Pass the localized operation title to the history panel."""
    history_panel = SimpleNamespace(
        add_compute_entry_from_pp=Mock(return_value=object()),
        capture_outputs=lambda _action: nullcontext(),
    )
    processor = SimpleNamespace(
        panel=SimpleNamespace(
            PANEL_STR_ID="signal",
            objview=SimpleNamespace(
                get_sel_objects=Mock(return_value=[]),
                get_sel_groups=Mock(return_value=[]),
            ),
            objmodel=object(),
        ),
        mainwindow=SimpleNamespace(historypanel=history_panel),
        _get_plugin_origin_for=Mock(return_value=None),
    )

    def average(_objects: list[object]) -> None:
        return None

    with patch(
        "datalab.gui.processor.base.create_progress_bar",
        return_value=nullcontext(Mock()),
    ):
        BaseProcessor.compute_n_to_1(
            processor, average, title="Moyenne", pairwise=False
        )

    history_panel.add_compute_entry_from_pp.assert_called_once()
    assert history_panel.add_compute_entry_from_pp.call_args.args[0] == "Moyenne"


@pytest.mark.parametrize("column", (0, 2))
@pytest.mark.parametrize("selected_kind", ("action", "session"))
def test_history_tree_double_click_replays_current_selection_without_restoring(
    selected_kind: str, column: int
) -> None:
    """Replay the current action or session selection from either tree column."""
    if selected_kind == "action":
        selected_row: HistoryAction | HistorySession = HistoryAction()
        expected_actions = [selected_row]
    else:
        selected_row = HistorySession()
        selected_row.add_action(HistoryAction())
        selected_row.add_action(HistoryAction())
        expected_actions = list(selected_row.actions)
    selected_row.is_current_state_compatible = Mock(return_value=True)
    clicked_row = HistoryAction()
    tree = SimpleNamespace(
        customContextMenuRequested=Mock(),
        itemDoubleClicked=Mock(),
        itemSelectionChanged=Mock(),
        get_selected_actions_or_sessions=Mock(return_value=[selected_row]),
    )
    mainwindow = object()
    panel = SimpleNamespace(
        tree=tree,
        history_sessions=[],
        mainwindow=mainwindow,
        refresh_compatibility_items=Mock(),
        replaying=nullcontext,
        output_suppressed=nullcontext,
        runtime=SimpleNamespace(execution=SimpleNamespace(edit_mode=False)),
        navigation=SimpleNamespace(
            sync_panel_selection=Mock(),
            update_state_widget=Mock(),
            set_active_session_from_selection=Mock(),
        ),
    )
    panel.replay_restore_actions = lambda **kwargs: hireplay.replay_restore_actions(
        panel, **kwargs
    )
    ui = HistoryPanelUI.__new__(HistoryPanelUI)
    ui.panel = panel

    ui.setup_connections()
    double_click_slot = tree.itemDoubleClicked.connect.call_args.args[0]
    with patch.object(hireplay, "replay_actions") as replay_actions_mock:
        double_click_slot(clicked_row, column)

    selected_row.is_current_state_compatible.assert_called_once_with(
        mainwindow, restore_selection=False
    )
    replay_actions_mock.assert_called_once_with(panel, expected_actions, prompt=False)
    assert clicked_row not in replay_actions_mock.call_args.args[1]


def test_image_creation_extends_active_signal_session_when_rejected() -> None:
    """Chain an image creation into the single active recording session."""
    signal_session = make_prompt_session("signal", populated=True)
    signal_actions = list(signal_session.actions)
    panel = PromptPanel([signal_session])
    unattended = SimpleNamespace(unattended=True, accept_dialogs=False)
    with (
        patch.object(hsess, "execenv", unattended),
        patch.object(
            hsess.Conf.proc.history_new_session_behavior, "get", return_value="ask"
        ),
    ):
        action = hsess.add_ui_entry(
            panel,
            "New image",
            target="imagepanel",
            method_name="new_object",
            save_state=False,
        )
    assert action is panel.added_actions[0]
    assert action.panel_str == "image"
    assert panel.prompt_behaviors == [None]
    assert panel.runtime.execution.prompt_count == 1
    assert panel.created_sessions == []
    assert panel.navigation.get_active_session() is signal_session
    assert signal_session.actions == signal_actions + [action]


def test_empty_active_session_skips_prompt() -> None:
    """Reuse an empty active session without reaching the debounce."""
    image_session = make_prompt_session("image", populated=False)
    panel = PromptPanel([image_session])
    created = hsess.maybe_start_session_for_input(panel)
    assert created is False
    assert panel.runtime.execution.prompt_count == 0
    assert panel.created_sessions == []


def test_omitted_session_behavior_reads_live_general_policy() -> None:
    """Resolve the general policy on every omitted-behavior call."""
    image_session = make_prompt_session("image", populated=True)
    panel = PromptPanel([image_session])
    option = hsess.Conf.proc.history_new_session_behavior

    with patch.object(option, "get", side_effect=["no", "yes"]) as get_behavior:
        first_created = panel.maybe_start_session_for_input()
        second_created = panel.maybe_start_session_for_input()

    assert first_created is False
    assert second_created is True
    assert len(panel.created_sessions) == 1
    assert panel.prompt_behaviors == [None, None]
    assert get_behavior.call_count == 2


def test_explicit_session_behaviors_bypass_prompt() -> None:
    """Apply explicit yes/no policies without debounce or a dialog."""
    cases: tuple[tuple[hsess.SessionBehavior, bool], ...] = (
        ("yes", True),
        ("no", False),
    )
    attended = SimpleNamespace(unattended=False, accept_dialogs=False)
    for behavior, expected_created in cases:
        image_session = make_prompt_session("image", populated=True)
        panel = PromptPanel([image_session])
        with (
            patch.object(hsess, "execenv", attended),
            patch.object(hsess.QW, "QMessageBox") as message_box,
        ):
            created = panel.maybe_start_session_for_input(behavior=behavior)
        assert created is expected_created
        assert panel.runtime.execution.prompt_count == 0
        assert len(panel.created_sessions) == (1 if expected_created else 0)
        message_box.question.assert_not_called()


def test_invalid_session_behavior_has_no_side_effects() -> None:
    """Reject an invalid policy before debounce or session creation."""
    image_session = make_prompt_session("image", populated=True)
    panel = PromptPanel([image_session])

    with pytest.raises(ValueError, match="Invalid session behavior"):
        hsess.maybe_start_session_for_input(
            panel,
            behavior=cast(hsess.SessionBehavior, "invalid"),
        )

    assert panel.runtime.execution.prompt_count == 0
    assert panel.created_sessions == []
    assert panel.history_sessions == [image_session]


def test_accepted_prompt_routes_action_to_new_session() -> None:
    """Record an image creation in the newly accepted session."""
    image_session = make_prompt_session("image", populated=True)
    previous_actions = list(image_session.actions)
    panel = PromptPanel([image_session])
    unattended = SimpleNamespace(unattended=True, accept_dialogs=True)
    with (
        patch.object(hsess, "execenv", unattended),
        patch.object(
            hsess.Conf.proc.history_new_session_behavior, "get", return_value="ask"
        ),
    ):
        action = hsess.add_ui_entry(
            panel,
            "New image",
            target="imagepanel",
            method_name="new_object",
            save_state=False,
        )
    new_session = panel.navigation.get_active_session()
    assert panel.runtime.execution.prompt_count == 1
    assert panel.created_sessions == [new_session]
    assert new_session is panel.history_sessions[-1]
    assert new_session.actions == [action]
    assert image_session.actions == previous_actions


def test_accepted_prompt_replaces_active_session() -> None:
    """Make the freshly created session the single active recording session."""
    signal_session = make_prompt_session("signal", populated=True)
    panel = PromptPanel([signal_session])
    unattended = SimpleNamespace(unattended=True, accept_dialogs=True)
    with patch.object(hsess, "execenv", unattended):
        created = hsess.maybe_start_session_for_input(panel, behavior="ask")
    assert created is True
    assert panel.runtime.execution.prompt_count == 1
    assert panel.created_sessions == [panel.navigation.get_active_session()]
    assert panel.navigation.get_active_session() is not signal_session


def test_input_prompt_debounce_is_global() -> None:
    """Debounce a synchronous burst of prompts on a single timer window."""
    signal_session = make_prompt_session("signal", populated=True)
    panel = PromptPanel([signal_session])
    execution = hruntime.HistoryExecutionState()
    panel.runtime = SimpleNamespace(execution=execution)
    callbacks = []
    unattended = SimpleNamespace(unattended=True, accept_dialogs=False)

    with (
        patch.object(
            hruntime.QC.QTimer,
            "singleShot",
            side_effect=lambda _delay, callback: callbacks.append(callback),
        ),
        patch.object(hsess, "execenv", unattended),
    ):
        assert not hsess.maybe_start_session_for_input(panel, behavior="ask")
        assert not hsess.maybe_start_session_for_input(panel, behavior="ask")

    assert execution.session_input_pending is True
    assert len(callbacks) == 1
    callbacks[0]()
    assert execution.session_input_pending is False


def test_ui_entries_serialize_data_target_ownership() -> None:
    """Round-trip data-target ownership and preserve legacy target fallback."""
    expected_ownership = {
        "signalpanel": "signal",
        "imagepanel": "image",
        "signalprocessor": "signal",
        "imageprocessor": "image",
        "mainwindow": None,
        "historypanel": None,
    }
    panel = PromptPanel([])
    actions = []
    for target, panel_str in expected_ownership.items():
        action = hsess.add_ui_entry(
            panel,
            target,
            target=target,
            method_name="refresh",
            save_state=False,
        )
        assert action is not None
        assert action.panel_str == panel_str
        actions.append(action)

    legacy_ownership = {
        "signalprocessor": "signal",
        "imageprocessor": "image",
    }
    legacy_actions = [
        HistoryAction(
            title=target,
            kind=HistoryAction.KIND_UI,
            panel_str="",
            target=target,
            method_name="refresh",
        )
        for target in legacy_ownership
    ]
    session = HistorySession(number=1)
    for action in actions + legacy_actions:
        session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        with NativeH5Reader(path) as reader:
            loaded_actions = reader.read_object_list("history_session", HistorySession)[
                0
            ].actions

    for action, (target, panel_str) in zip(
        loaded_actions[: len(actions)], expected_ownership.items()
    ):
        assert action.target == target
        assert action.panel_str == panel_str
        assert action.effective_panel_str() == (panel_str or "")
    for action, (target, panel_str) in zip(
        loaded_actions[len(actions) :], legacy_ownership.items()
    ):
        assert action.target == target
        assert action.panel_str == ""
        assert action.effective_panel_str() == panel_str


def test_resolve_panel_for_data_owned_ui_actions() -> None:
    """Resolve legacy processor and data-panel load actions to image data."""
    signal_panel = SimpleNamespace(name="signal")
    image_panel = SimpleNamespace(name="image")
    history_panel = SimpleNamespace(
        mainwindow=SimpleNamespace(
            signalpanel=signal_panel,
            imagepanel=image_panel,
        )
    )
    image_actions = (
        HistoryAction(
            kind=HistoryAction.KIND_UI,
            panel_str="",
            target="imageprocessor",
            method_name="run_feature",
        ),
        HistoryAction(
            kind=HistoryAction.KIND_UI,
            panel_str="",
            target="imagepanel",
            method_name="load_from_files",
        ),
    )

    for action in image_actions:
        assert hchain.resolve_panel_for_action(history_panel, action) is image_panel
    for target in ("mainwindow", "historypanel"):
        action = HistoryAction(
            kind=HistoryAction.KIND_UI,
            target=target,
            method_name="refresh",
        )
        assert hchain.resolve_panel_for_action(history_panel, action) is None


def test_history_tools_action_panel_str_uses_effective_ownership() -> None:
    """Recognize a legacy image processor while preserving ownerless fallback."""
    action = HistoryAction(
        kind=HistoryAction.KIND_UI,
        panel_str="",
        target="imageprocessor",
    )
    assert hops.action_panel_str(action) == "image"
    assert hops.action_panel_str(HistoryAction(target="mainwindow")) == "signal"


def test_image_histogram_action_keeps_image_ownership_in_active_session() -> None:
    """Keep a signal-valued histogram owned by its image source panel."""
    signal_session = make_prompt_session("signal", populated=True)
    image_session = make_prompt_session("image", populated=True)
    signal_actions = list(signal_session.actions)
    image_actions = list(image_session.actions)
    signal_output = SimpleNamespace(uuid="histogram-signal", panel_str="signal")
    image_source_panel = SimpleNamespace(name="image")
    panel = PromptPanel([signal_session, image_session])
    panel.mainwindow = SimpleNamespace(
        signalpanel=SimpleNamespace(name="signal", objects=[signal_output]),
        imagepanel=image_source_panel,
    )
    action = HistoryAction(
        title="Histogram",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="image",
        func_name="histogram",
        pattern="1_to_1",
    )
    action.output_uuids = [signal_output.uuid]

    panel.add_object(action)

    assert signal_output.panel_str == "signal"
    assert action.effective_panel_str() == "image"
    assert hops.action_panel_str(action) == "image"
    assert hchain.resolve_panel_for_action(panel, action) is image_source_panel
    assert signal_session.actions == signal_actions
    assert image_session.actions == image_actions + [action]


def test_debounce_rejection_does_not_start_or_prompt() -> None:
    """Keep routing in the active session when debounce rejects."""
    image_session = make_prompt_session("image", populated=True)
    panel = PromptPanel([image_session], prompt_allowed=False)
    attended = SimpleNamespace(unattended=False, accept_dialogs=True)
    with (
        patch.object(hsess, "execenv", attended),
        patch.object(hsess.QW, "QMessageBox") as message_box,
        patch.object(
            hsess.Conf.proc.history_new_session_behavior, "get", return_value="ask"
        ),
    ):
        action = hsess.add_ui_entry(
            panel,
            "New image",
            target="imagepanel",
            method_name="new_object",
            save_state=False,
        )
    assert panel.runtime.execution.prompt_count == 1
    assert panel.created_sessions == []
    assert panel.navigation.get_active_session() is image_session
    assert image_session.actions[-1] is action
    message_box.question.assert_not_called()


def test_unattended_reject_keeps_populated_active_session() -> None:
    """Leave the active session untouched after unattended rejection."""
    image_session = make_prompt_session("image", populated=True)
    previous_actions = list(image_session.actions)
    panel = PromptPanel([image_session])
    unattended = SimpleNamespace(unattended=True, accept_dialogs=False)
    with patch.object(hsess, "execenv", unattended):
        created = hsess.maybe_start_session_for_input(panel, behavior="ask")
    assert created is False
    assert panel.runtime.execution.prompt_count == 1
    assert panel.created_sessions == []
    assert panel.navigation.get_active_session() is image_session
    assert image_session.actions == previous_actions


def test_mainwindow_add_object_decides_before_mutation_and_suppresses_entry() -> None:
    """Decide before adding and suppress the creation entry's second prompt."""
    image_session = make_prompt_session("image", populated=True)
    historypanel = PromptPanel([image_session])
    added_objects = []

    def add_to_image_panel(obj, group_id, set_current):
        historypanel.events.append("panel_mutation")
        added_objects.append((obj, group_id, set_current))

    mainwindow = SimpleNamespace(
        confirm_memory_state=lambda: True,
        signalpanel=SimpleNamespace(),
        imagepanel=SimpleNamespace(add_object=add_to_image_panel),
        historypanel=historypanel,
    )
    unattended = SimpleNamespace(unattended=True, accept_dialogs=False)
    image = ImageObj()
    with (
        patch.object(hsess, "execenv", unattended),
        patch.object(
            hsess.Conf.proc.history_new_session_behavior, "get", return_value="ask"
        ),
    ):
        added = DLMainWindow.add_object(mainwindow, image, new_session_behavior="ask")

    assert added is True
    assert added_objects == [(image, "", True)]
    assert historypanel.events[:3] == [
        "session_decision",
        "panel_mutation",
        "session_decision",
    ]
    assert historypanel.prompt_behaviors == ["ask", None]
    assert historypanel.prompt_suppressed_states == [False, True]
    assert historypanel.runtime.execution.prompt_count == 1
    assert len(historypanel.registered_outputs) == 1


def test_mainwindow_add_object_preserves_no_record_and_memory_rejection() -> None:
    """Keep data addition without recording and stop entirely on memory refusal."""
    image_session = make_prompt_session("image", populated=True)
    historypanel = PromptPanel([image_session])
    historypanel.record_mode_enabled = False
    added_objects = []

    def add_to_image_panel(obj, group_id, set_current):
        added_objects.append((obj, group_id, set_current))

    mainwindow = SimpleNamespace(
        confirm_memory_state=lambda: True,
        signalpanel=SimpleNamespace(),
        imagepanel=SimpleNamespace(add_object=add_to_image_panel),
        historypanel=historypanel,
    )
    image = ImageObj()
    assert DLMainWindow.add_object(mainwindow, image, new_session_behavior="no") is True
    assert added_objects == [(image, "", True)]
    assert historypanel.added_actions == []
    assert historypanel.registered_outputs == []

    mainwindow.confirm_memory_state = lambda: False
    assert (
        DLMainWindow.add_object(mainwindow, ImageObj(), new_session_behavior="yes")
        is False
    )
    assert added_objects == [(image, "", True)]
    assert historypanel.prompt_behaviors == ["no"]


def test_action_hdf5_current_and_legacy_contract() -> None:
    """Round-trip current fields and apply all legacy defaults."""
    action = build_history_action()
    action.plugin_origin = {
        "module": "example.plugin",
        "metadata": {"entry_points": ["difference"]},
    }
    action.snapshot_kwargs()
    action.kwargs["pairwise"] = True
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        current = read_history_sessions(path)[0].actions[0]
        for attribute in ("selection", "states", "titles"):
            values = getattr(action.state, attribute)
            setattr(action.state, attribute, {"Signal Panel": values["signal"]})
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
            for field in ("schema_version", "uuid", "saved_kwargs", "output_uuids"):
                delete_hdf5_items_by_name(writer.h5, field)
            delete_hdf5_items_by_name(writer.h5, "object_metadata")
        legacy = read_history_sessions(path)[0].actions[0]
    assert current.uuid == action.uuid
    assert current.schema_version == HISTORY_ACTION_SCHEMA_VERSION
    assert current.output_uuids == ["output-uuid"]
    assert current.plugin_origin == action.plugin_origin
    assert current.has_pending_edits and bool(current.kwargs["pairwise"])
    assert legacy.schema_version == HISTORY_SCHEMA_VERSION
    assert legacy.uuid != action.uuid and legacy.output_uuids == []
    assert not legacy.has_pending_edits and legacy.state.object_metadata == {}
    assert legacy.state.selection == {"signal": ["source-uuid"]}
    assert legacy.state.states == {"signal": ["(10,)"]}
    assert legacy.state.titles == {"signal": ["Source"]}


def test_action_copy_remaps_all_uuid_references() -> None:
    """Copy an action independently and rewrite every captured UUID field."""
    action = build_history_action()
    action.plugin_origin = {
        "module": "example.plugin",
        "metadata": {"entry_points": ["difference"]},
    }
    copied = action.copy_with_uuid_remap(
        {
            "signal": {
                "source-uuid": "new-source",
                "second-uuid": "new-second",
                "output-uuid": "new-output",
            }
        }
    )
    assert copied is not action and copied.uuid != action.uuid
    assert copied.state.selection == {"signal": ["new-source"]}
    assert copied.state.object_metadata == {
        "signal": {"new-source": {"shape": [10], "ndim": 1, "title": "Source"}}
    }
    assert copied.kwargs["obj2_uuids"] == "new-second"
    assert copied.output_uuids == ["new-output"]
    assert copied.plugin_origin == action.plugin_origin
    copied.state.object_metadata["signal"]["new-source"]["shape"] = [20]
    copied.plugin_origin["metadata"]["entry_points"].append("average")
    assert action.state.object_metadata["signal"]["source-uuid"]["shape"] == [10]
    assert action.plugin_origin["metadata"]["entry_points"] == ["difference"]


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
    source = create_image_from_param(initial_param)
    source_uuid = get_uuid(source)
    source_identity = id(source)
    initial_source_data = source.data.copy()
    expected_source = create_image_from_param(edited_param)

    downstream = source.copy()
    set_uuid(downstream)
    downstream.title = "Initial downstream"
    downstream.data = np.full(source.data.shape, -1.0)
    downstream_uuid = get_uuid(downstream)
    downstream_identity = id(downstream)
    initial_downstream_data = downstream.data.copy()
    insert_processing_parameters(
        downstream,
        ProcessingParameters(
            func_name="unit_transform",
            pattern="1-to-1",
            source_uuid=source_uuid,
        ),
    )

    creation_action = HistoryAction(
        title="Create edited Gaussian",
        kind=HistoryAction.KIND_UI,
        target="imagepanel",
        method_name="new_object",
        kwargs={"param": edited_param},
    )
    creation_action.output_uuids = [source_uuid]
    downstream_state = WorkspaceState()
    downstream_state.selection = {"image": [source_uuid]}
    downstream_action = HistoryAction(
        title="Transform edited Gaussian",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="image",
        func_name="unit_transform",
        pattern="1_to_1",
        state=downstream_state,
    )
    downstream_action.output_uuids = [downstream_uuid]
    session = HistorySession(number=1)
    session.add_action(creation_action)
    session.add_action(downstream_action)

    processor_source_objects: list[ImageObj] = []
    processor_source_data: list[np.ndarray] = []

    def recompute_1_to_1(
        func_name: str | None,
        source_obj: ImageObj,
        param: object,
        *,
        plugin_origin: dict[str, object] | None,
    ) -> SimpleNamespace:
        assert func_name == "unit_transform"
        assert param is None
        assert plugin_origin is None
        processor_source_objects.append(source_obj)
        processor_source_data.append(source_obj.data.copy())
        new_obj = source_obj.copy()
        new_obj.title = f"unit_transform({source_obj.title})"
        new_obj.data = source_obj.data.astype(float) * 2.0 + 3.0
        return SimpleNamespace(cancelled=False, error_msg=None, result=new_obj)

    def apply_recomputed_object_in_place(
        obj: ImageObj,
        new_obj: ImageObj,
        proc_params: ProcessingParameters,
    ) -> None:
        hrec.update_obj_in_place(obj, new_obj)
        insert_processing_parameters(obj, proc_params)

    object_model = CascadeObjectModel([source, downstream])
    data_panel = SimpleNamespace(
        PANEL_STR_ID="image",
        objmodel=object_model,
        processor=SimpleNamespace(recompute_1_to_1=recompute_1_to_1),
        objprop=SimpleNamespace(
            apply_recomputed_object_in_place=apply_recomputed_object_in_place
        ),
    )
    runtime = SimpleNamespace(
        objects=SimpleNamespace(
            action_output_uuids={
                creation_action.uuid: [source_uuid],
                downstream_action.uuid: [downstream_uuid],
            }
        ),
        execution=SimpleNamespace(cascade_warnings=[], broken_actions=set()),
    )
    history_panel = SimpleNamespace(
        runtime=runtime,
        history_sessions=[session],
    )
    refreshed_uuids: list[str] = []

    def record_refresh(panel_data: object, output_uuid: str) -> None:
        assert panel_data is data_panel
        refreshed_uuids.append(output_uuid)

    descendants = hrec.hchain.get_downstream_actions(history_panel, creation_action)
    assert descendants == [downstream_action]

    with (
        patch.object(hrec.hchain, "resolve_panel_for_action", return_value=data_panel),
        patch.object(
            hrec, "create_image_from_param", wraps=create_image_from_param
        ) as create_image_mock,
        patch.object(hrec, "refresh_target", side_effect=record_refresh),
    ):
        assert hrec.recompute_creation_in_place(history_panel, creation_action)
        assert hrec.recompute_1_to_1_in_place(history_panel, downstream_action)

    create_image_mock.assert_called_once()
    assert create_image_mock.call_args.args[0] is edited_param

    assert id(source) == source_identity
    assert object_model[source_uuid] is source
    assert get_uuid(source) == source_uuid
    np.testing.assert_allclose(source.data, expected_source.data)
    assert not np.array_equal(source.data, initial_source_data)
    creation_param = extract_creation_parameters(source)
    assert isinstance(creation_param, Gauss2DParam)
    for name in ("height", "width", "x0", "y0", "sigma", "a"):
        assert getattr(creation_param, name) == getattr(edited_param, name)

    assert processor_source_objects == [source]
    np.testing.assert_allclose(processor_source_data[0], expected_source.data)
    assert id(downstream) == downstream_identity
    assert object_model[downstream_uuid] is downstream
    assert get_uuid(downstream) == downstream_uuid
    np.testing.assert_allclose(
        downstream.data, expected_source.data.astype(float) * 2.0 + 3.0
    )
    assert not np.array_equal(downstream.data, initial_downstream_data)
    downstream_params = extract_processing_parameters(downstream)
    assert downstream_params is not None
    assert downstream_params.source_uuid == source_uuid
    assert refreshed_uuids == [source_uuid, downstream_uuid]
    assert runtime.execution.cascade_warnings == []
