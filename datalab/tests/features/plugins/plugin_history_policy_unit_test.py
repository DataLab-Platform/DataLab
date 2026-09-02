# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Pure unit contracts for plugin input history policies."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast
from unittest.mock import patch

import numpy as np
import pytest
from sigima import ImageObj, SignalObj
from sigima.io.image import ImageIORegistry
from sigima.io.signal import SignalIORegistry

from datalab.config import Conf
from datalab.control.proxy import LocalProxy
from datalab.gui import historysession_ops as hsess
from datalab.gui.main import DLMainWindow
from datalab.plugins import PluginRegistry

testdata_path = Path(__file__).parents[3] / "plugins" / "datalab_testdata.py"
testdata_spec = importlib.util.spec_from_file_location(
    "datalab_testdata", testdata_path
)
if testdata_spec is None or testdata_spec.loader is None:
    raise ImportError(f"Unable to load test data plugin from {testdata_path}")
testdata_plugin = importlib.util.module_from_spec(testdata_spec)
testdata_spec.loader.exec_module(testdata_plugin)
PluginTestData = testdata_plugin.PluginTestData


class AddCallRecorder:
    """Record object-add calls received from a local proxy."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, hsess.SessionBehavior | None]] = []

    def add_object(
        self,
        obj: SignalObj | ImageObj,
        group_id: str = "",
        set_current: bool = True,
        new_session_behavior: hsess.SessionBehavior | None = None,
    ) -> bool:
        """Record an object add."""
        del obj, group_id, set_current
        self.calls.append(("object", new_session_behavior))
        return True

    def add_signal(
        self, *args: Any, new_session_behavior: hsess.SessionBehavior | None = None
    ) -> bool:
        """Record a signal add."""
        del args
        self.calls.append(("signal", new_session_behavior))
        return True

    def add_image(
        self, *args: Any, new_session_behavior: hsess.SessionBehavior | None = None
    ) -> bool:
        """Record an image add."""
        del args
        self.calls.append(("image", new_session_behavior))
        return True


class HistoryPromptRecorder:
    """Record prompt evaluations and suppression state for multi-load tests."""

    def __init__(self) -> None:
        self.suppressed = False
        self.calls: list[tuple[bool, hsess.SessionBehavior | None, bool]] = []
        self.decision_count = 0

    def maybe_start_session_for_input(
        self,
        *,
        load: bool = False,
        behavior: hsess.SessionBehavior | None = None,
    ) -> bool:
        """Record a validated session-policy evaluation."""
        if behavior is not None and behavior not in hsess.SESSION_BEHAVIORS:
            raise ValueError(f"Invalid session behavior: {behavior!r}")
        self.calls.append((load, behavior, self.suppressed))
        if not self.suppressed and behavior != "no":
            self.decision_count += 1
        return False

    def add_ui_entry(
        self,
        action_title: str,
        target: str,
        method_name: str,
        save_state: bool = True,
    ) -> None:
        """Record the nested creation evaluation from the main window."""
        del action_title, target, method_name, save_state
        self.maybe_start_session_for_input()

    @contextmanager
    def session_prompt_suppressed(self) -> Generator[None, None, None]:
        """Suppress decisions while preserving nested context state."""
        previous = self.suppressed
        self.suppressed = True
        try:
            yield
        finally:
            self.suppressed = previous


class MultiLoadMainRecorder:
    """Exercise the production main-window boundary for local proxy adds."""

    def __init__(self) -> None:
        self.historypanel = HistoryPromptRecorder()
        self.memory_allowed = True
        self.memory_confirmation_count = 0
        self.added_objects: list[SignalObj | ImageObj] = []
        self.signalpanel = SimpleNamespace(add_object=self.record_signal_object)
        self.imagepanel = SimpleNamespace(add_object=self.record_image_object)

    def confirm_memory_state(self) -> bool:
        """Return the controlled memory confirmation result."""
        self.memory_confirmation_count += 1
        return self.memory_allowed

    def record_signal_object(
        self, obj: SignalObj, group_id: str, set_current: bool
    ) -> None:
        """Record a signal panel mutation."""
        del group_id, set_current
        assert isinstance(obj, SignalObj)
        self.added_objects.append(obj)

    def record_image_object(
        self, obj: ImageObj, group_id: str, set_current: bool
    ) -> None:
        """Record an image panel mutation."""
        del group_id, set_current
        assert isinstance(obj, ImageObj)
        self.added_objects.append(obj)

    def add_object(
        self,
        obj: SignalObj | ImageObj,
        group_id: str = "",
        set_current: bool = True,
        new_session_behavior: hsess.SessionBehavior | None = None,
    ) -> bool:
        """Add an object through the production main-window method."""
        return DLMainWindow.add_object(
            as_mainwindow(self),
            obj,
            group_id,
            set_current,
            new_session_behavior,
        )

    def add_signal(
        self, *args: Any, new_session_behavior: hsess.SessionBehavior | None = None
    ) -> bool:
        """Add a signal through the production main-window method."""
        return DLMainWindow.add_signal(
            as_mainwindow(self), *args, new_session_behavior=new_session_behavior
        )

    def add_image(
        self, *args: Any, new_session_behavior: hsess.SessionBehavior | None = None
    ) -> bool:
        """Add an image through the production main-window method."""
        return DLMainWindow.add_image(
            as_mainwindow(self), *args, new_session_behavior=new_session_behavior
        )


class PluginProxyRecorder:
    """Record the Test Data plugin's batch scope and object ordering."""

    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    @contextmanager
    def multiload_session(
        self, panel: str, new_session_behavior: hsess.SessionBehavior | None = None
    ) -> Generator[None, None, None]:
        """Record entry and exit of a plugin multi-load context."""
        del new_session_behavior
        self.events.append(("enter", panel))
        try:
            yield
        finally:
            self.events.append(("exit", panel))

    def add_object(self, obj: object) -> None:
        """Record an object in insertion order."""
        self.events.append(("add", obj))


class ProgressRecorder:
    """Controllable progress-bar context for plugin load tests."""

    def __init__(self, cancel_states: list[bool]) -> None:
        self.cancel_states = iter(cancel_states)
        self.values: list[int] = []

    def __enter__(self) -> ProgressRecorder:
        return self

    def __exit__(self, *args: Any) -> None:
        del args

    def setValue(self, value: int) -> None:  # pylint: disable=invalid-name
        """Record a progress value."""
        self.values.append(value)

    def wasCanceled(self) -> bool:  # pylint: disable=invalid-name
        """Return the next controlled cancellation state."""
        return next(self.cancel_states, False)


def as_mainwindow(mainwindow: object) -> DLMainWindow:
    """Cast a pure test double to the local proxy's window interface."""
    return cast("DLMainWindow", mainwindow)


def test_local_proxy_resolves_plugin_policy_live_and_explicit_wins() -> None:
    """Read plugin policy per add while preserving explicit priority."""
    mainwindow = AddCallRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")
    option = Conf.history_plugin_new_session_behavior
    xdata = np.array([0.0, 1.0])
    ydata = np.array([1.0, 2.0])

    with patch.object(option, "get", side_effect=["yes", "no"]) as get_behavior:
        proxy.add_object(SignalObj())
        proxy.add_signal("signal", xdata, ydata)
        proxy.add_image("image", np.ones((2, 2)), new_session_behavior="ask")

    assert mainwindow.calls == [
        ("object", "yes"),
        ("signal", "no"),
        ("image", "ask"),
    ]
    assert get_behavior.call_count == 2

    local_mainwindow = AddCallRecorder()
    LocalProxy(as_mainwindow(local_mainwindow)).add_object(ImageObj())
    assert local_mainwindow.calls == [("object", None)]


def test_plugin_registration_marks_proxy_for_live_plugin_policy() -> None:
    """Create plugin proxies with the plugin input source marker."""
    plugin = PluginTestData()
    mainwindow = AddCallRecorder()
    option = Conf.history_plugin_new_session_behavior

    with patch.object(PluginRegistry, "register_plugin"):
        plugin.register(as_mainwindow(mainwindow))

    assert plugin.proxy.input_source == "plugin"
    with patch.object(option, "get", return_value="no") as get_behavior:
        plugin.proxy.add_object(ImageObj())
    assert mainwindow.calls == [("object", "no")]
    get_behavior.assert_called_once_with()


def test_plugin_multiload_decides_once_and_suppresses_internal_adds() -> None:
    """Apply the batch policy once and use no for later additions."""
    mainwindow = MultiLoadMainRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")
    multiload_option = Conf.history_plugin_multiload_behavior
    add_option = Conf.history_plugin_new_session_behavior
    first = SignalObj()
    second = SignalObj()

    with (
        patch.object(multiload_option, "get", return_value="ask") as get_multiload,
        patch.object(add_option, "get", return_value="no") as get_add,
    ):
        with proxy.multiload_session("signal"):
            assert proxy.add_object(first) is True
            assert proxy.add_object(second) is True

    assert mainwindow.added_objects == [first, second]
    assert mainwindow.historypanel.calls == [
        (False, "ask", False),
        (False, None, True),
        (False, "no", False),
        (False, None, True),
    ]
    assert mainwindow.historypanel.decision_count == 1
    assert mainwindow.historypanel.suppressed is False
    assert proxy.multiload_state is None
    get_multiload.assert_called_once_with()
    get_add.assert_not_called()


def test_multiload_validates_inputs_and_explicit_policy_wins() -> None:
    """Validate batch inputs before yielding and prioritize explicit policy."""
    mainwindow = MultiLoadMainRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")
    option = Conf.history_plugin_multiload_behavior

    with patch.object(option, "get") as get_behavior:
        with proxy.multiload_session("image", new_session_behavior="no"):
            assert proxy.multiload_state is not None
            assert proxy.multiload_state.panel == "image"
            assert proxy.multiload_state.behavior == "no"
            assert proxy.multiload_state.decision_applied is False
    get_behavior.assert_not_called()
    assert proxy.multiload_state is None
    assert not mainwindow.historypanel.calls

    with pytest.raises(ValueError, match="Invalid data panel"):
        with proxy.multiload_session(cast(Any, "macro")):
            pass
    with pytest.raises(ValueError, match="Invalid session behavior"):
        with proxy.multiload_session("signal", cast(hsess.SessionBehavior, "invalid")):
            pass

    with proxy.multiload_session("signal", "ask"):
        outer_state = proxy.multiload_state
        with pytest.raises(RuntimeError, match="Nested multiload sessions"):
            with proxy.multiload_session("image", "no"):
                pass
        assert proxy.multiload_state is outer_state
    assert proxy.multiload_state is None


def test_multiload_defers_empty_and_pre_add_exception_decisions() -> None:
    """Leave history untouched when a batch never attempts an insertion."""
    mainwindow = MultiLoadMainRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")
    option = Conf.history_plugin_multiload_behavior

    with patch.object(option, "get", return_value="ask") as get_behavior:
        with proxy.multiload_session("signal"):
            pass
        with pytest.raises(ValueError, match="before first add"):
            with proxy.multiload_session("image"):
                raise ValueError("before first add")

    assert get_behavior.call_count == 2
    assert proxy.multiload_state is None
    assert mainwindow.memory_confirmation_count == 0
    assert not mainwindow.historypanel.calls
    assert not mainwindow.added_objects


def test_multiload_memory_rejection_does_not_consume_first_decision() -> None:
    """Defer the batch decision until an object passes memory confirmation."""
    mainwindow = MultiLoadMainRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")
    rejected = SignalObj()
    accepted = SignalObj()

    with proxy.multiload_session("signal", "ask"):
        mainwindow.memory_allowed = False
        assert proxy.add_object(rejected) is False
        assert proxy.multiload_state is not None
        assert proxy.multiload_state.decision_applied is False
        mainwindow.memory_allowed = True
        assert proxy.add_object(accepted) is True
        assert proxy.multiload_state.decision_applied is True

    assert mainwindow.memory_confirmation_count == 2
    assert mainwindow.added_objects == [accepted]
    assert mainwindow.historypanel.calls == [
        (False, "ask", False),
        (False, None, True),
    ]
    assert mainwindow.historypanel.decision_count == 1


def test_multiload_rejects_panel_mismatch_before_mainwindow_mutation() -> None:
    """Reject an object from another panel before memory or history changes."""
    mainwindow = MultiLoadMainRecorder()
    proxy = LocalProxy(as_mainwindow(mainwindow), input_source="plugin")

    with proxy.multiload_session("signal", "ask"):
        with pytest.raises(ValueError, match="during a signal multiload session"):
            proxy.add_object(ImageObj())

    assert mainwindow.memory_confirmation_count == 0
    assert not mainwindow.historypanel.calls
    assert not mainwindow.added_objects


def test_typed_adds_report_memory_rejection() -> None:
    """Keep add_signal and add_image return values truthful on rejection."""
    mainwindow = MultiLoadMainRecorder()
    mainwindow.memory_allowed = False
    proxy = LocalProxy(as_mainwindow(mainwindow))
    xdata = np.array([0.0, 1.0])
    ydata = np.array([1.0, 2.0])

    assert proxy.add_signal("signal", xdata, ydata) is False
    assert proxy.add_image("image", np.ones((2, 2))) is False
    assert mainwindow.memory_confirmation_count == 2
    assert not mainwindow.historypanel.calls
    assert not mainwindow.added_objects


@pytest.mark.parametrize(
    ("registry_class", "panel_str", "panel_attribute"),
    (
        (SignalIORegistry, "signal", "signalpanel"),
        (ImageIORegistry, "image", "imagepanel"),
    ),
)
def test_testdata_multiload_selects_registry_panel_and_preserves_order(
    registry_class, panel_str: str, panel_attribute: str
) -> None:
    """Select the registry's panel and preserve progress and object order."""
    plugin = PluginTestData()
    signalpanel = object()
    imagepanel = object()
    plugin.main = cast(
        Any, SimpleNamespace(signalpanel=signalpanel, imagepanel=imagepanel)
    )
    proxy = PluginProxyRecorder()
    plugin.proxy = cast(LocalProxy, proxy)
    first = object()
    second = object()
    progress = ProgressRecorder([False, False])
    progress_calls = []

    def create_progress(parent, title, max_):
        progress_calls.append((parent, title, max_))
        return progress

    with (
        patch.object(
            testdata_plugin.helpers,
            "read_test_objects",
            return_value=[("first", first), ("second", second)],
        ),
        patch.object(testdata_plugin, "create_progress_bar", create_progress),
    ):
        plugin.load_test_objs(registry_class, "Load objects")

    expected_panel = getattr(plugin.main, panel_attribute)
    assert progress_calls == [(expected_panel, "Load objects", 2)]
    assert progress.values == [1, 2]
    assert proxy.events == [
        ("enter", panel_str),
        ("add", first),
        ("add", second),
        ("exit", panel_str),
    ]


def test_testdata_multiload_preserves_immediate_cancellation() -> None:
    """Leave the lazy batch empty when progress is cancelled immediately."""
    plugin = PluginTestData()
    signalpanel = object()
    plugin.main = cast(
        Any, SimpleNamespace(signalpanel=signalpanel, imagepanel=object())
    )
    proxy = PluginProxyRecorder()
    plugin.proxy = cast(LocalProxy, proxy)
    first = object()
    second = object()
    progress = ProgressRecorder([True])

    with (
        patch.object(
            testdata_plugin.helpers,
            "read_test_objects",
            return_value=[
                ("first", first),
                ("second", second),
            ],
        ),
        patch.object(testdata_plugin, "create_progress_bar", return_value=progress),
    ):
        plugin.load_test_objs(SignalIORegistry, "Load signals")

    assert progress.values == [1]
    assert proxy.events == [
        ("enter", "signal"),
        ("exit", "signal"),
    ]
