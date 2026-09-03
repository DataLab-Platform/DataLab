# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for history persistence, copying and recompute."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import pytest
from sigima.objects import (
    Gauss2DParam,
    ImageObj,
    SignalObj,
    SignalROI,
    create_image_from_param,
    create_image_roi,
    create_signal_roi,
)
from sigima.tests.data import create_paracetamol_signal

from datalab.gui import historysession_ops as hsess
from datalab.gui import historytools_ops as hops
from datalab.gui.main import DLMainWindow
from datalab.gui.panel.history import chain as hchain
from datalab.gui.panel.history import interactive_replay as hireplay
from datalab.gui.panel.history import recompute as hrec
from datalab.gui.panel.history import runtime as hruntime
from datalab.gui.panel.history.chainmodel import ProcessingChain, UuidCloneRegistry
from datalab.gui.processor.base import (
    BaseProcessor,
    FeatureNotFoundError,
    ProcessingParameters,
    extract_processing_parameters,
    insert_processing_parameters,
)
from datalab.h5.native import NativeH5Reader, NativeH5Writer
from datalab.history.action import HistoryAction
from datalab.history.core import (
    HISTORY_ACTION_SCHEMA_VERSION,
    HISTORY_SCHEMA_VERSION,
    decode_roi,
    numpy_to_json_safe,
)
from datalab.history.effects import AnalysisEffects, capture_effects, merge_effects
from datalab.history.session import HistorySession
from datalab.history.workspace_state import WorkspaceState
from datalab.objectmodel import get_uuid
from datalab.tests.features.common.history_test_helpers import (
    CascadeObjectModel,
    build_history_action,
    build_workspace_state,
    delete_hdf5_items_by_name,
    read_history_sessions,
)


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


def test_image_creation_extends_active_signal_session_when_rejected() -> None:
    """Chain an image creation into the single active recording session."""
    signal_session = make_prompt_session("signal", populated=True)
    signal_actions = list(signal_session.actions)
    panel = PromptPanel([signal_session])
    unattended = SimpleNamespace(unattended=True, accept_dialogs=False)
    with (
        patch.object(hsess, "execenv", unattended),
        patch.object(
            hsess.Conf.history_new_session_behavior, "get", return_value="ask"
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
    assert not panel.created_sessions
    assert panel.navigation.get_active_session() is signal_session
    assert signal_session.actions == signal_actions + [action]


def test_empty_active_session_skips_prompt() -> None:
    """Reuse an empty active session without reaching the debounce."""
    image_session = make_prompt_session("image", populated=False)
    panel = PromptPanel([image_session])
    created = hsess.maybe_start_session_for_input(panel)
    assert created is False
    assert panel.runtime.execution.prompt_count == 0
    assert not panel.created_sessions


@pytest.mark.parametrize(
    ("behavior", "policy_values", "expected_created"),
    (
        (None, ("no", "yes"), (False, True)),
        ("yes", ("ask",), (True,)),
        ("no", ("ask",), (False,)),
        ("invalid", (), None),
    ),
)
def test_session_behavior_policy_matrix(
    behavior: str | None,
    policy_values: tuple[str, ...],
    expected_created: tuple[bool, ...] | None,
) -> None:
    """Resolve omitted, explicit and invalid session policies without dialogs.

    Omitted behaviors re-read the live general policy on every call, explicit
    yes/no policies bypass both debounce and dialog, and invalid policies are
    rejected before any side effect.
    """
    image_session = make_prompt_session("image", populated=True)
    panel = PromptPanel([image_session])
    attended = SimpleNamespace(unattended=False, accept_dialogs=False)
    option = hsess.Conf.history_new_session_behavior
    with (
        patch.object(hsess, "execenv", attended),
        patch.object(hsess.QW, "QMessageBox") as message_box,
        patch.object(option, "get", side_effect=list(policy_values)) as get_policy,
    ):
        if expected_created is None:
            with pytest.raises(ValueError, match="Invalid session behavior"):
                panel.maybe_start_session_for_input(
                    behavior=cast(hsess.SessionBehavior, behavior)
                )
            created = []
        else:
            created = [
                panel.maybe_start_session_for_input(
                    behavior=cast(hsess.SessionBehavior, behavior)
                )
                for _call in policy_values
            ]
    assert created == list(expected_created or ())
    assert len(panel.created_sessions) == sum(created)
    assert panel.runtime.execution.prompt_count == 0
    message_box.question.assert_not_called()
    assert get_policy.call_count == (len(policy_values) if behavior is None else 0)
    if expected_created is None:
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
            hsess.Conf.history_new_session_behavior, "get", return_value="ask"
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
    """Debounce a synchronous prompt burst and keep routing in place.

    Only the first prompt of a synchronous burst opens a dialog; while the
    debounce window is pending, further creations are routed to the active
    session without starting a new one or prompting again.
    """
    signal_session = make_prompt_session("signal", populated=True)
    panel = PromptPanel([signal_session])
    # pylint: disable=attribute-defined-outside-init
    panel.mainwindow = None  # dialog parent for the patched QMessageBox
    execution = hruntime.HistoryExecutionState()
    panel.runtime = SimpleNamespace(execution=execution)
    callbacks = []
    attended = SimpleNamespace(unattended=False, accept_dialogs=False)

    with (
        patch.object(
            hruntime.QC.QTimer,
            "singleShot",
            side_effect=lambda _delay, callback: callbacks.append(callback),
        ),
        patch.object(hsess, "execenv", attended),
        patch.object(hsess.QW, "QMessageBox") as message_box,
        patch.object(
            hsess.Conf.history_new_session_behavior, "get", return_value="ask"
        ),
    ):
        # First call passes the debounce and opens the (rejected) dialog
        assert not hsess.maybe_start_session_for_input(panel, behavior="ask")
        # Second call is debounced: no second dialog
        assert not hsess.maybe_start_session_for_input(panel, behavior="ask")
        # Creation entries routed during the window stay in the active session
        action = hsess.add_ui_entry(
            panel,
            "New image",
            target="imagepanel",
            method_name="new_object",
            save_state=False,
        )

    assert message_box.question.call_count == 1
    assert execution.session_input_pending is True
    assert len(callbacks) == 1
    assert not panel.created_sessions
    assert panel.navigation.get_active_session() is signal_session
    assert signal_session.actions[-1] is action
    callbacks[0]()
    assert execution.session_input_pending is False
    # Re-entrance guards yield False when already active
    with execution.recomputing_cascade() as started:
        assert started is True
        with execution.recomputing_cascade() as nested:
            assert nested is False
    with execution.replaying_edits() as started:
        assert started is True
        with execution.replaying_edits() as nested:
            assert nested is False


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
    # pylint: disable=attribute-defined-outside-init
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
    assert not panel.created_sessions
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
            hsess.Conf.history_new_session_behavior, "get", return_value="ask"
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
    assert not historypanel.added_actions
    assert not historypanel.registered_outputs

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


def test_action_hdf5_roi_decode_failure_degrades_locally() -> None:
    """Degrade a corrupt ROI kwarg locally instead of aborting the file load."""
    action = build_history_action()
    # Raw marker dict passes through encode_kwargs untouched, simulating a
    # persisted ROI payload referencing an untrusted (non-sigima) module.
    corrupt_payload = {
        "__roi_json__": json.dumps(
            {"module": "evil.module", "class": "FakeROI", "data": {}}
        )
    }
    action.kwargs["payload"] = corrupt_payload
    # A corrupted edit snapshot must be dropped (no usable rollback value).
    action.saved_kwargs = {"payload": dict(corrupt_payload)}
    valid_action = build_history_action()
    session = HistorySession(number=1)
    session.add_action(action)
    session.add_action(valid_action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        with pytest.warns(UserWarning, match="ROI kwarg"):
            loaded = read_history_sessions(path)[0]
    broken, intact = loaded.actions
    # The broken action loads with degraded kwargs and is permanently
    # incompatible, so it can never be replayed with altered semantics.
    assert broken.decode_failed is True
    assert broken.kwargs == {}
    assert broken.saved_kwargs is None
    assert broken.has_pending_edits is False
    assert broken.is_current_state_compatible(Mock()) is False
    assert broken.copy().decode_failed is True
    # The rest of the file loads normally.
    assert intact.decode_failed is False
    assert intact.kwargs == {"obj2_uuids": ["second-uuid"], "pairwise": False}


def test_recompute_in_place_skips_decode_failed_action() -> None:
    """Never execute an action whose persisted parameters failed to decode."""
    warnings: list[str] = []
    panel = SimpleNamespace(
        runtime=SimpleNamespace(execution=SimpleNamespace(cascade_warnings=warnings)),
        tree=SimpleNamespace(refresh_action_item=lambda action: None),
    )
    action = HistoryAction(
        title="Edit ROI", kind=HistoryAction.KIND_MUTATION, mutation_key="roi"
    )
    action.decode_failed = True
    assert hrec.recompute_action_in_place(panel, action) is False
    # Not marked stale: the action is permanently non-recomputable.
    assert action.is_stale is False
    assert any("Edit ROI" in warning for warning in warnings)


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


def _make_analysis_action(obj_uuid: str) -> HistoryAction:
    """Build a 1-to-0 compute action analysing ``obj_uuid``."""
    return HistoryAction(
        title="FWHM",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        func_name="fwhm",
        pattern="1_to_0",
        state=build_workspace_state([obj_uuid]),
    )


def test_find_analysis_action_two_pass_matching() -> None:
    """Prefer the effects manifest, then fall back to the input-uuid heuristic."""
    obj_uuid = "analysed-uuid"
    older = _make_analysis_action(obj_uuid)
    newer = _make_analysis_action(obj_uuid)
    session = HistorySession(number=1)
    session.add_action(older)
    session.add_action(newer)
    panel = SimpleNamespace(history_sessions=[session])
    manifest = {obj_uuid: AnalysisEffects(metadata_added=["fwhm"]).to_dict()}
    # Manifest pass: only the older action recorded effects for the object,
    # so it wins over the more recent heuristic-only match
    older.effects = manifest
    assert hchain.find_analysis_action(panel, obj_uuid, "fwhm") is older
    # Both actions carry a manifest: the most recent one wins
    newer.effects = dict(manifest)
    assert hchain.find_analysis_action(panel, obj_uuid, "fwhm") is newer
    # Legacy pass: without any manifest, the input-uuid heuristic matches the
    # most recent action
    older.effects = None
    newer.effects = None
    assert hchain.find_analysis_action(panel, obj_uuid, "fwhm") is newer
    # No match for another function name or another object
    assert hchain.find_analysis_action(panel, obj_uuid, "fw1e2") is None
    assert hchain.find_analysis_action(panel, "other-uuid", "fwhm") is None


def test_object_metadata_roi_signature_presence_and_stability() -> None:
    """Expose a stable ROI signature and omit it for ROI-less objects."""
    obj = create_paracetamol_signal()
    assert "roi" not in WorkspaceState.get_object_metadata(obj)
    obj.roi = create_signal_roi([[10, 20]], indices=True)
    signature = WorkspaceState.get_object_metadata(obj)["roi"]
    obj.roi = create_signal_roi([[10, 20]], indices=True)
    assert WorkspaceState.get_object_metadata(obj)["roi"] == signature
    obj.roi = create_signal_roi([[15, 30]], indices=True)
    assert WorkspaceState.get_object_metadata(obj)["roi"] != signature


def test_state_compatibility_handles_roi_signature_and_legacy_metadata() -> None:
    """Tolerate legacy metadata without ROI key and flag ROI drift otherwise."""
    obj = create_paracetamol_signal()
    obj.roi = create_signal_roi([[10, 20]], indices=True)
    uuid = get_uuid(obj)
    mainwindow = cast(
        DLMainWindow,
        SimpleNamespace(
            signalpanel=SimpleNamespace(
                PANEL_STR_ID="signal", objmodel=CascadeObjectModel([obj])
            ),
            imagepanel=SimpleNamespace(
                PANEL_STR_ID="image", objmodel=CascadeObjectModel([])
            ),
        ),
    )
    state = WorkspaceState()
    state.selection = {"signal": [uuid]}
    recorded = WorkspaceState.get_object_metadata(obj)
    state.object_metadata = {"signal": {uuid: recorded}}
    assert state.is_current_state_compatible(mainwindow)
    # Legacy tolerance: metadata recorded without "roi" vs current object with ROI
    legacy = dict(recorded)
    del legacy["roi"]
    state.object_metadata = {"signal": {uuid: legacy}}
    assert state.is_current_state_compatible(mainwindow)
    # ROI drift against a recorded signature flags incompatibility
    state.object_metadata = {"signal": {uuid: dict(recorded, roi="0" * 16)}}
    assert not state.is_current_state_compatible(mainwindow)


def test_workspace_state_save_tolerates_object_without_data() -> None:
    """Capture states without crashing when a selected object has no data."""
    obj = create_paracetamol_signal()
    empty_obj = SignalObj()
    empty_obj.title = "Empty"

    def make_panel(panel_id: str, objs: list) -> SimpleNamespace:
        return SimpleNamespace(
            PANEL_STR_ID=panel_id,
            objmodel=CascadeObjectModel(objs),
            objview=SimpleNamespace(get_sel_objects=lambda include_groups=True: objs),
        )

    mainwindow = cast(
        DLMainWindow,
        SimpleNamespace(
            signalpanel=make_panel("signal", [obj, empty_obj]),
            imagepanel=make_panel("image", []),
        ),
    )
    state = WorkspaceState()
    state.save(mainwindow)
    assert state.states["signal"] == [str(obj.data.shape), ""]
    assert state.object_metadata["signal"][get_uuid(empty_obj)] == {}


def test_mutation_action_model_contract() -> None:
    """Round-trip, copy and remap mutation actions with and without payload."""

    def roi_as_dict(roi: SignalROI) -> dict:
        return numpy_to_json_safe(roi.to_dict())

    def build_mutation_action(payload: SignalROI | None) -> HistoryAction:
        return HistoryAction(
            title="Edit regions of interest",
            kind=HistoryAction.KIND_MUTATION,
            panel_str="signal",
            mutation_key="roi",
            target_uuids=["first-uuid", "second-uuid"],
            kwargs={"payload": payload},
        )

    def roundtrip(action: HistoryAction) -> HistoryAction:
        session = HistorySession(number=1)
        session.add_action(action)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.dlhist")
            with NativeH5Writer(path) as writer:
                writer.write_object_list([session], "history_session")
            return read_history_sessions(path)[0].actions[0]

    payload = create_signal_roi([[26, 41], [125, 146]], indices=True)
    loaded = roundtrip(build_mutation_action(payload))
    assert loaded.kind == HistoryAction.KIND_MUTATION
    assert loaded.mutation_key == "roi"
    assert loaded.target_uuids == ["first-uuid", "second-uuid"]
    assert loaded.panel_str == "signal"
    loaded_payload = loaded.kwargs.get("payload")
    assert isinstance(loaded_payload, SignalROI)
    assert type(loaded_payload) is type(payload)
    assert roi_as_dict(loaded_payload) == roi_as_dict(payload)
    # A None payload (ROI deletion) is dropped at construction time and
    # decoded back as a missing kwarg
    deletion = build_mutation_action(None)
    assert "payload" not in deletion.kwargs
    loaded = roundtrip(deletion)
    assert loaded.kind == HistoryAction.KIND_MUTATION
    assert loaded.target_uuids == ["first-uuid", "second-uuid"]
    assert loaded.kwargs.get("payload") is None
    # Copies are independent; UUID remapping rewrites the mutation targets
    action = build_mutation_action(create_signal_roi([[26, 41]], indices=True))
    copied = action.copy()
    assert copied is not action and copied.uuid != action.uuid
    assert copied.mutation_key == "roi"
    assert copied.target_uuids == action.target_uuids
    assert copied.target_uuids is not action.target_uuids
    assert roi_as_dict(copied.kwargs["payload"]) == roi_as_dict(
        action.kwargs["payload"]
    )
    remapped = action.copy_with_uuid_remap(
        {"signal": {"first-uuid": "new-first", "second-uuid": "new-second"}}
    )
    assert remapped.target_uuids == ["new-first", "new-second"]
    assert action.target_uuids == ["first-uuid", "second-uuid"]


def test_capture_effects_metadata_and_roi_diff() -> None:
    """Diff metadata keys and flag only genuine ROI changes."""
    obj = create_image_from_param(Gauss2DParam.create(height=16, width=16))
    obj.metadata["untouched"] = 1
    obj.metadata["changed_scalar"] = 5
    obj.metadata["changed_array"] = np.arange(3)
    with capture_effects(obj) as effects:
        obj.metadata["new_key"] = "hello"
        obj.metadata["changed_scalar"] = 6
        obj.metadata["changed_array"] = np.arange(4)
        obj.metadata["__uuid"] = "synthetic-uuid"
        obj.metadata["__number"] = 42
    assert effects.metadata_added == ["new_key"]
    assert effects.metadata_replaced == ["changed_array", "changed_scalar"]
    assert "untouched" not in effects.metadata_added + effects.metadata_replaced
    assert "__uuid" not in effects.metadata_added
    assert "__number" not in effects.metadata_added
    assert effects.roi_modified is False
    # No ROI before/after: unmodified
    with capture_effects(obj) as effects:
        pass
    assert effects.roi_modified is False
    # ROI creation flags the capture
    with capture_effects(obj) as effects:
        obj.roi = create_image_roi("rectangle", [2, 2, 5, 5])
    assert effects.roi_modified is True
    # No ROI existed before: recorded as the restorable "no ROI" sentinel
    assert effects.roi_before == ""
    # An existing ROI left untouched by the analysis is not modified
    with capture_effects(obj) as effects:
        obj.metadata["another_key"] = 0
    assert effects.roi_modified is False
    assert effects.roi_before is None
    # Re-assigning an equal ROI is not a modification (relies on ROI equality)
    with capture_effects(obj) as effects:
        obj.roi = create_image_roi("rectangle", [2, 2, 5, 5])
    assert effects.roi_modified is False
    # Changing the ROI geometry is a modification
    with capture_effects(obj) as effects:
        obj.roi = create_image_roi("rectangle", [3, 3, 6, 6])
    assert effects.roi_modified is True
    # The pre-execution ROI payload round-trips through encode/decode
    restored = decode_roi(effects.roi_before)
    assert numpy_to_json_safe(restored.to_dict()) == numpy_to_json_safe(
        create_image_roi("rectangle", [2, 2, 5, 5]).to_dict()
    )


def test_analysis_effects_round_trip_merge_and_persistence() -> None:
    """Round-trip manifests through dict/HDF5 and merge recompute captures."""
    effects = AnalysisEffects(
        metadata_added=["Geometry_peak_detection_dict"],
        metadata_replaced=["analysis_parameters"],
        roi_modified=True,
    )
    payload = effects.to_dict()
    assert payload == {
        "metadata_added": ["Geometry_peak_detection_dict"],
        "metadata_replaced": ["analysis_parameters"],
        "roi_modified": True,
    }
    assert AnalysisEffects.from_dict(payload) == effects
    assert AnalysisEffects.from_dict({}) == AnalysisEffects()
    # roi_before round-trips only when recorded (legacy payloads unchanged)
    recorded = AnalysisEffects(roi_modified=True, roi_before="")
    recorded_payload = recorded.to_dict()
    assert "roi_before" in recorded_payload
    assert AnalysisEffects.from_dict(recorded_payload) == recorded
    # Merge semantics: added-stays-added, sticky roi_modified, sorted output
    new = AnalysisEffects(
        metadata_added=["b", "a"], metadata_replaced=["c"], roi_modified=False
    )
    merged = merge_effects(None, new)
    assert merged == AnalysisEffects(["a", "b"], ["c"], False)
    previous = AnalysisEffects(metadata_added=["result"], roi_modified=True)
    recomputed = AnalysisEffects(metadata_replaced=["result", "params"])
    merged = merge_effects(previous, recomputed)
    assert merged.metadata_added == ["result"]
    assert merged.metadata_replaced == ["params"]
    assert merged.roi_modified is True
    # roi_before is first-run sticky: recompute captures never overwrite it
    previous = AnalysisEffects(roi_modified=True, roi_before="first-run-payload")
    recomputed = AnalysisEffects(roi_modified=True, roi_before="recompute-payload")
    merged = merge_effects(previous, recomputed)
    assert merged.roi_before == "first-run-payload"
    # A legacy previous manifest adopts the freshly recorded roi_before
    legacy_previous = AnalysisEffects(roi_modified=True)
    merged = merge_effects(legacy_previous, recomputed)
    assert merged.roi_before == "recompute-payload"
    # HDF5 round-trip on an action, with legacy tolerance (no effects group)
    action = build_history_action()
    action.effects = {"source-uuid": payload}
    session = HistorySession(number=1)
    session.add_action(action)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "history.dlhist")
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
        loaded = read_history_sessions(path)[0].actions[0]
        assert loaded.effects == action.effects
        with NativeH5Writer(path) as writer:
            writer.write_object_list([session], "history_session")
            delete_hdf5_items_by_name(writer.h5, "effects")
        legacy = read_history_sessions(path)[0].actions[0]
        assert legacy.effects is None


def test_update_obj_in_place_preserves_roi() -> None:
    """In-place recompute keeps the target's ROI when the new object has none."""
    target = create_paracetamol_signal()
    target.roi = create_signal_roi([[10, 20]], indices=True)
    saved_roi_dict = numpy_to_json_safe(target.roi.to_dict())
    new_obj = create_paracetamol_signal()
    assert new_obj.roi is None
    hrec.update_obj_in_place(target, new_obj)
    assert target.roi is not None
    assert numpy_to_json_safe(target.roi.to_dict()) == saved_roi_dict


def test_recompute_dispatch_guards_and_missing_feature() -> None:
    """Reject non-recomputable actions and diagnose missing plugin features."""
    warnings: list[str] = []
    panel = SimpleNamespace(
        runtime=SimpleNamespace(execution=SimpleNamespace(cascade_warnings=warnings)),
        mainwindow=SimpleNamespace(
            signalpanel=SimpleNamespace(objmodel=CascadeObjectModel([])),
            imagepanel=SimpleNamespace(objmodel=CascadeObjectModel([])),
        ),
    )
    # Non-creation UI actions are silently not recomputable
    noncompute = HistoryAction(kind=HistoryAction.KIND_UI, method_name="select_next")
    assert hrec.recompute_action_in_place(panel, noncompute) is False
    assert not warnings
    # Unsupported compute patterns queue a warning
    unsupported = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE, func_name="mystery", pattern="3_to_2"
    )
    assert hrec.recompute_action_in_place(panel, unsupported) is False
    assert any("mystery" in warning for warning in warnings)
    # A missing plugin feature flags the action and queues a diagnostic
    action = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE, func_name="plugin_func", pattern="1_to_1"
    )
    error = FeatureNotFoundError(
        "plugin_func",
        plugin_origin={"directory": "myplugin"},
        paramclass_name="MyParam",
    )
    with patch.object(hrec, "recompute_compute_in_place", side_effect=error):
        assert hrec.recompute_action_in_place(panel, action) is False
    assert action.is_stale is True
    assert any(
        "myplugin/plugins:plugin_func" in warning and "MyParam" in warning
        for warning in warnings
    )
    # Mutation guards: unresolved data panel, then no recorded targets
    orphan_mutation = HistoryAction(
        title="Edit ROI", kind=HistoryAction.KIND_MUTATION, mutation_key="roi"
    )
    assert hrec.recompute_action_in_place(panel, orphan_mutation) is False
    targetless = HistoryAction(
        title="Edit ROI",
        kind=HistoryAction.KIND_MUTATION,
        panel_str="signal",
        mutation_key="roi",
    )
    assert hrec.recompute_action_in_place(panel, targetless) is False
    assert sum("Edit ROI" in warning for warning in warnings) == 2


def test_find_creation_action_for_output_fallback_scan() -> None:
    """Fall back to scanning creation outputs when the mapping is stale."""
    creation = HistoryAction(
        title="New signal",
        kind=HistoryAction.KIND_UI,
        target="signalpanel",
        method_name="new_object",
    )
    compute = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE, func_name="derivative", pattern="1_to_1"
    )
    session = HistorySession(number=1)
    session.add_action(creation)
    session.add_action(compute)
    runtime = SimpleNamespace(
        objects=SimpleNamespace(
            output_to_action={"created-uuid": compute.uuid},
            action_output_uuids={creation.uuid: ["created-uuid"]},
        )
    )
    panel = SimpleNamespace(history_sessions=[session], runtime=runtime)
    # The mapped action is not a creation: the fallback scan finds the head
    assert hchain.find_creation_action_for_output(panel, "created-uuid") is creation
    assert hchain.find_creation_action_for_output(panel, "unknown-uuid") is None
    empty = SimpleNamespace(history_sessions=[])
    assert hchain.find_creation_action_for_output(empty, "created-uuid") is None


def test_plan_reconnection_dead_source_warning_and_producer_removal() -> None:
    """Warn on dead sources, then reconnect and remove the dead producer."""
    source = create_paracetamol_signal()
    source_uuid = get_uuid(source)
    removed_uuid = "removed-output"
    consumer_obj = create_paracetamol_signal()
    consumer_uuid = get_uuid(consumer_obj)
    insert_processing_parameters(
        consumer_obj,
        ProcessingParameters(
            func_name="derivative", pattern="1-to-1", source_uuid=removed_uuid
        ),
    )
    producer = HistoryAction(
        title="Normalize",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        func_name="normalize",
        pattern="1_to_1",
        state=build_workspace_state([source_uuid]),
    )
    producer.output_uuids = [removed_uuid]
    consumer_action = HistoryAction(
        title="Derivative",
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        func_name="derivative",
        pattern="1_to_1",
        state=build_workspace_state([removed_uuid]),
    )
    session = HistorySession(number=1)
    session.add_action(producer)
    session.add_action(consumer_action)

    def make_panel(objects: list) -> tuple[SimpleNamespace, SimpleNamespace]:
        signal_panel = SimpleNamespace(
            PANEL_STR_ID="signal", objmodel=CascadeObjectModel(objects)
        )
        panel = SimpleNamespace(
            history_sessions=[session],
            runtime=SimpleNamespace(
                objects=SimpleNamespace(
                    output_to_action={removed_uuid: producer.uuid},
                    action_output_uuids={producer.uuid: [removed_uuid]},
                    remove_action_outputs=Mock(),
                )
            ),
            mainwindow=SimpleNamespace(
                signalpanel=signal_panel,
                imagepanel=SimpleNamespace(
                    PANEL_STR_ID="image", objmodel=CascadeObjectModel([])
                ),
            ),
        )
        return panel, signal_panel

    # Dead source: the plan carries a warning and applying it is a no-op
    panel, signal_panel = make_panel([consumer_obj])
    plan = hchain.plan_reconnection(panel, signal_panel, removed_uuid)
    assert plan.warning is not None and "Normalize" in plan.warning
    assert [target.object_uuid for target in plan.targets] == [consumer_uuid]
    assert plan.targets[0].action is consumer_action
    roots: list[HistoryAction] = []
    hchain.apply_reconnection_plan(panel, signal_panel, plan, roots)
    assert not roots
    assert extract_processing_parameters(consumer_obj).source_uuid == removed_uuid
    # Reconnection warnings are silenced in unattended mode
    with (
        patch.object(hchain, "execenv", SimpleNamespace(unattended=True)),
        patch.object(hchain.QW, "QMessageBox") as message_box,
    ):
        hchain.show_reconnection_warnings(panel, [plan.warning])
    message_box.warning.assert_not_called()

    # Alive source: reconnect the consumer and remove the outputless producer
    panel, signal_panel = make_panel([source, consumer_obj])
    plan = hchain.plan_reconnection(panel, signal_panel, removed_uuid)
    assert plan.warning is None
    assert plan.source_uuid == source_uuid
    assert plan.remove_producer is True
    roots = []
    hchain.apply_reconnection_plan(panel, signal_panel, plan, roots)
    assert roots == [consumer_action]
    assert extract_processing_parameters(consumer_obj).source_uuid == source_uuid
    assert consumer_action.state.selection["signal"] == [source_uuid]
    assert producer not in session.actions
    panel.runtime.objects.remove_action_outputs.assert_called_once_with(producer)


def test_prepare_action_param_edit_skips_paramless_actions() -> None:
    """Return no edit target (and skip the dialog) for param-less actions."""
    paramless = (
        HistoryAction(kind=HistoryAction.KIND_UI, method_name="new_object"),
        HistoryAction(kind=HistoryAction.KIND_COMPUTE, pattern="1_to_1"),
        HistoryAction(kind=HistoryAction.KIND_COMPUTE, pattern="1_to_n"),
        HistoryAction(kind=HistoryAction.KIND_MUTATION, mutation_key="roi"),
    )
    panel = SimpleNamespace(mainwindow=None)
    for action in paramless:
        assert hireplay.prepare_action_param_edit(action) is None
        assert hireplay.prompt_edit_action_params(panel, action) is None


def test_make_synthetic_heads_falls_back_to_default_title() -> None:
    """Use the default head title when the cloned object cannot be resolved."""
    root = HistoryAction(
        kind=HistoryAction.KIND_COMPUTE,
        panel_str="signal",
        func_name="derivative",
        pattern="1_to_1",
        state=build_workspace_state(["external-uuid"]),
    )
    chain = ProcessingChain(root=root, session=HistorySession(number=1), actions=[root])
    registry = UuidCloneRegistry()
    registry.register("signal", "external-uuid", "clone-uuid", object())

    class RaisingModel:
        """Object model whose lookups always fail."""

        def __getitem__(self, uuid: str) -> None:
            raise KeyError(uuid)

    panel = SimpleNamespace(
        mainwindow=SimpleNamespace(
            signalpanel=SimpleNamespace(objmodel=RaisingModel()), imagepanel=None
        )
    )
    heads = hops.make_synthetic_heads(panel, chain, registry)
    assert len(heads) == 1
    head = heads[0]
    assert head.method_name == "new_object"
    assert head.output_uuids == ["clone-uuid"]
    assert head.title == hops._("Initial state")
