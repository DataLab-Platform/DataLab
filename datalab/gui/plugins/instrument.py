# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Window showing a plugin instrument: live view and settings."""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures import wait as wait_futures
from typing import TYPE_CHECKING, Callable, Iterable

from guidata.configtools import get_icon
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.qthelpers import win32_fix_title_bar_background
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj
from sigimax.adapters_plotpy.objects.signal import CURVESTYLES

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import Conf, _
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.plugins.instruments import (
        InstrumentAcquisition,
        InstrumentFrame,
        PluginInstrument,
    )

__all__ = [
    "AcquisitionTarget",
    "InstrumentWindow",
    "SerialWorker",
    "wait_instruments_closed",
]


def wait_instruments_closed(
    closings: Iterable[tuple[str, Future]], timeout: float
) -> list[str]:
    """Wait for instruments being closed, within a common time limit.

    Args:
        closings: title of each instrument, with the call closing it
        timeout: maximum waiting time for all instruments (s)

    Returns:
        Titles of the instruments that were not closed in time
    """
    closings = list(closings)
    _done, not_done = wait_futures([future for _title, future in closings], timeout)
    return [title for title, future in closings if future in not_done]


class SerialWorker:
    """Run calls one at a time in a daemon thread.

    Unlike a thread pool, a call stuck in third-party code does not prevent
    DataLab from quitting.
    """

    def __init__(self) -> None:
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        thread = threading.Thread(target=self._run, name="instrument", daemon=True)
        thread.start()

    def submit(self, func: Callable, *args) -> Future:
        """Schedule a call and return its future."""
        future: Future = Future()
        self._queue.put((future, func, args))
        return future

    def shutdown(self) -> None:
        """Stop the thread once the scheduled calls are done."""
        self._queue.put(None)

    def _run(self) -> None:
        """Run scheduled calls until shutdown."""
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, func, args = item
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(func(*args))
                except BaseException as exc:  # pylint: disable=broad-except
                    future.set_exception(exc)


@dataclasses.dataclass(frozen=True)
class AcquisitionTarget:
    """Destination of an acquisition, taken when it is requested.

    Args:
        new_group: True to add the acquired objects to a new group
        group_ids: current group of each panel (``"signal"``, ``"image"``), if
         any: the panel of the objects is only known once they are acquired
    """

    new_group: bool
    group_ids: dict[str, str | None]


class InstrumentWindow(QW.QDialog):
    """Live view of a plugin instrument, next to its settings.

    The view is refreshed when settings change, and periodically in live mode.
    Each acquisition is added to the workspace in a new group or, when the
    **New group** option is off, in the current group of the panel, as they are
    when the acquisition is requested. The
    instrument is opened with the window and closed with it; if opening it
    fails, the reason is shown, and it is opened again before the next call. Its
    methods are called one at a time in a worker thread: a request made while a
    call is running is done next (an acquisition before a refresh, repeated
    refreshes only once).

    Args:
        main: DataLab main window
        instrument: instrument opened by a plugin tool
        title: window title
        executor: worker running the calls of the instrument, shared by the
         successive windows of the instrument so that a window opened again runs
         its calls after those of the closed one (default: a worker of its own,
         stopped when the window is closed)
        option_key: key under which the state of the **New group** option is
         remembered in the configuration (default: not remembered, the option
         starts from :attr:`PluginInstrument.new_group`)
    """

    #: Delay between a settings change and the refresh of the view (ms)
    REFRESH_DELAY = 150

    #: Maximum time DataLab waits for an instrument to close (s)
    CLOSE_TIMEOUT = 5.0

    SIG_CALL_DONE = QC.Signal(str, object, object, object)

    def __init__(
        self,
        main: DLMainWindow,
        instrument: PluginInstrument,
        title: str,
        executor: SerialWorker | None = None,
        option_key: str | None = None,
    ) -> None:
        super().__init__(main)
        win32_fix_title_bar_background(self)
        self.main = main
        self.instrument = instrument
        self.option_key = option_key
        self.setWindowTitle(title.rstrip(".…"))
        self.setModal(False)
        self.plotwidget: PlotWidget | None = None
        self.items: list = []
        self._plot_kind: str | None = None
        self._plot_signature: tuple | None = None
        self._value_range: tuple[float, float] | None = None
        self._owns_executor = executor is None
        self._executor = SerialWorker() if executor is None else executor
        self._busy = False
        self._opened = False
        self._pending: str | None = None
        # Destination of the requested acquisition, until the instrument is called
        self._acquire_target: AcquisitionTarget | None = None
        self._close_future: Future | None = None
        self.SIG_CALL_DONE.connect(self._call_done)

        splitter = QW.QSplitter(QC.Qt.Horizontal)
        self._stage = QW.QWidget()
        self._stage.setMinimumSize(360, 280)
        self._stage_layout = QW.QVBoxLayout(self._stage)
        self._stage_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QW.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(QC.Qt.PlainText)
        self._stage_layout.addWidget(self.summary_label)
        splitter.addWidget(self._stage)

        settings_widget = QW.QWidget()
        settings_layout = QW.QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        form = QW.QWidget()
        form_layout = QW.QVBoxLayout(form)
        grid = QW.QGridLayout()
        grid.setAlignment(QC.Qt.AlignTop)
        form_layout.addLayout(grid)
        form_layout.addStretch()
        scroll.setWidget(form)
        settings_layout.addWidget(scroll, 1)
        self.status_label = QW.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QC.Qt.PlainText)
        settings_layout.addWidget(self.status_label)
        commands = QW.QHBoxLayout()
        self.live_button = QW.QPushButton(get_icon("replay.svg"), _("Live"))
        self.live_button.setCheckable(True)
        self.live_button.setToolTip(_("Refresh the view continuously"))
        self.live_button.toggled.connect(self.set_live)
        commands.addWidget(self.live_button)
        commands.addStretch()
        self.new_group_box = QW.QCheckBox(_("New group"))
        self.new_group_box.setToolTip(
            _(
                "Add each acquisition to a new group, "
                "otherwise to the current group of the panel"
            )
        )
        self.new_group_box.setChecked(self._stored_new_group())
        self.new_group_box.toggled.connect(self._store_new_group)
        commands.addWidget(self.new_group_box)
        self.acquire_button = QW.QPushButton(get_icon("record.svg"), _("Acquire"))
        self.acquire_button.setToolTip(_("Add an acquisition to the workspace"))
        self.acquire_button.clicked.connect(self.acquire)
        commands.addWidget(self.acquire_button)
        settings_layout.addLayout(commands)
        splitter.addWidget(settings_widget)

        self._refresh_timer = QC.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._live_timer = QC.QTimer(self)
        self._live_timer.setInterval(max(50, int(instrument.live_interval_ms)))
        self._live_timer.timeout.connect(self.refresh)
        self.finished.connect(self.stop)

        self.edit_layout = DataSetEditLayout(
            self,
            instrument.settings,
            grid,
            change_callback=self.settings_changed,
            auto_sliders=True,
        )

        buttons = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QW.QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(buttons)

        self.resize(1100, 650)
        screen = self.screen().availableGeometry()
        self.resize(
            min(self.width(), screen.width()), min(self.height(), screen.height())
        )
        splitter.setSizes([650, 450])
        # The first view is requested once the instrument is opened (filling the
        # settings editor reported changes, which do not need another view)
        self._refresh_timer.stop()
        self._busy = True
        self._pending = "preview"
        self._executor.submit(self._call, "open")

    def _call(self, method: str, target: AcquisitionTarget | None = None) -> None:
        """Call an instrument method (worker thread).

        Args:
            method: name of the method
            target: destination of an acquisition, returned with the result
        """
        result, error = None, None
        try:
            result = getattr(self.instrument, method)()
        except Exception as exc:  # pylint: disable=broad-except
            # Instruments are third-party code: keep the window usable
            error = exc
            if method == "close":
                logging.getLogger(__name__).exception("Closing instrument failed")
        try:
            self.SIG_CALL_DONE.emit(method, result, error, target)
        except RuntimeError:
            pass  # Window deleted after being closed

    def _request(self, method: str) -> None:
        """Call ``preview`` or ``acquire`` now, or after the running call.

        If opening the instrument failed, it is opened again first."""
        if self._close_future is not None:
            return
        if self._busy:
            if self._pending != "acquire":
                self._pending = method
            return
        if not self._apply_settings():
            return
        if not self._opened:
            self._busy = True
            self._pending = method
            self._executor.submit(self._call, "open")
            return
        target = None
        if method == "acquire":
            self._live_timer.stop()
            self.acquire_button.setEnabled(False)
            # Requests made during the acquisition do not change its destination
            target, self._acquire_target = self._acquire_target, None
        self._busy = True
        self._executor.submit(self._call, method, target)

    def _call_done(
        self,
        method: str,
        result,
        error: Exception | None,
        target: AcquisitionTarget | None,
    ) -> None:
        """Show the result of an instrument call, then make the pending one."""
        if method == "close":
            return
        self._busy = False
        if method == "acquire":
            self.acquire_button.setEnabled(True)
            # Live mode may have been switched off during the acquisition
            if self.live_button.isChecked() and self._close_future is None:
                self._live_timer.start()
        if self._close_future is not None:
            return
        if method == "open":
            self._opened = error is None
        if error is not None:
            self.status_label.setText(str(error))
            if method == "open":
                # Keep the reason in view: the instrument is not called until
                # the next request opens it again (e.g. after a settings change)
                self._pending = None
                return
        elif method == "preview":
            self._show_frame(result)
        elif method == "acquire":
            self._add_acquired(result, target)
        pending, self._pending = self._pending, None
        if pending is not None:
            self._request(pending)

    def is_idle(self) -> bool:
        """Return True if no instrument call is running, requested or scheduled."""
        return not (
            self._busy or self._pending is not None or self._refresh_timer.isActive()
        )

    def wait_until_idle(self, timeout: float = 10.0) -> bool:
        """Process events until no instrument call is running or requested.

        Args:
            timeout: maximum waiting time (s)

        Returns:
            True if the window is idle, False on timeout
        """
        deadline = time.monotonic() + timeout
        while not self.is_idle():
            if time.monotonic() > deadline:
                return False
            QW.QApplication.processEvents()
            time.sleep(0.005)
        return True

    def wait_closed(self, timeout: float = CLOSE_TIMEOUT) -> bool:
        """Wait until the instrument is closed, after the window was closed.

        Args:
            timeout: maximum waiting time (s)

        Returns:
            True if the instrument is closed, False on timeout
        """
        if self._close_future is None:
            return False
        done, _not_done = wait_futures([self._close_future], timeout)
        return bool(done)

    @property
    def close_future(self) -> Future | None:
        """Call closing the instrument, once the window was closed (else None)"""
        return self._close_future

    def child_title(self, item) -> str:
        """Supply the usual guidata title for nested editors."""
        return f"{self.windowTitle()} - {item.label()}"

    def settings_changed(self) -> None:
        """Refresh the view shortly after the settings changed."""
        self._refresh_timer.start(self.REFRESH_DELAY)

    def set_live(self, live: bool) -> None:
        """Start or stop refreshing the view continuously."""
        if live:
            self._live_timer.start()
        else:
            self._live_timer.stop()

    def _stored_new_group(self) -> bool:
        """Return the remembered state of the **New group** option, or the
        default state set by the instrument."""
        default = bool(self.instrument.new_group)
        if self.option_key is None:
            return default
        return bool(Conf.instrument_new_group.get().get(self.option_key, default))

    def _store_new_group(self, new_group: bool) -> None:
        """Remember the state of the **New group** option chosen by the user."""
        if self.option_key is not None:
            states = dict(Conf.instrument_new_group.get())
            states[self.option_key] = bool(new_group)
            Conf.instrument_new_group.set(states)

    def stop(self) -> None:
        """Stop refreshing the view and close the instrument."""
        self._refresh_timer.stop()
        self._live_timer.stop()
        if self._close_future is None:
            self._pending = None
            self._close_future = self._executor.submit(self._call, "close")
            if self._owns_executor:
                self._executor.shutdown()

    def closeEvent(self, event: QG.QCloseEvent) -> None:  # pylint: disable=invalid-name
        """Close the instrument, even if the window was never shown (Qt only
        emits ``finished`` when a visible dialog is closed)."""
        self.stop()
        super().closeEvent(event)

    def _apply_settings(self) -> bool:
        """Write the edited values into the instrument settings."""
        if not self.edit_layout.check_all_values():
            self.status_label.setText(_("Invalid settings"))
            return False
        self.edit_layout.accept_changes()
        return True

    def refresh(self) -> None:
        """Show a live frame for the current settings."""
        self._request("preview")

    def _show_frame(self, frame: InstrumentFrame) -> None:
        """Draw a frame returned by the instrument."""
        try:
            self.render(frame)
        except Exception as exc:  # pylint: disable=broad-except
            self.status_label.setText(str(exc))
            return
        self.status_label.clear()

    def render(self, frame: InstrumentFrame) -> None:
        """Draw a frame in the view."""
        first = frame.objects[0]
        kind = "image" if isinstance(first, ImageObj) else "curve"
        if self._plot_kind != kind:
            if self.plotwidget is not None:
                self._stage_layout.removeWidget(self.plotwidget)
                self.plotwidget.deleteLater()
            self.plotwidget = PlotWidget(self._stage, options=PlotOptions(type=kind))
            self._stage_layout.insertWidget(0, self.plotwidget, 1)
            self._plot_kind = kind
            self._plot_signature = None
            self.items = []
        plot = self.plotwidget.plot
        generator = CURVESTYLES.curve_style
        try:
            CURVESTYLES.curve_style = CURVESTYLES.style_generator()
            items = [
                create_adapter_from_object(obj).make_item() for obj in frame.objects
            ]
        finally:
            CURVESTYLES.curve_style = generator
        if self.items:
            plot.del_items(self.items)
        self.items = items
        for item in items:
            plot.add_item(item)
            item.unselect()
        plot.set_titles(
            title=first.title if kind == "image" else "",
            xlabel=first.xlabel,
            xunit=first.xunit,
            ylabel=(first.ylabel, getattr(first, "zlabel", "")),
            yunit=(first.yunit, getattr(first, "zunit", "")),
        )
        if kind == "image":
            signature = (kind, first.data.shape, first.x0, first.y0, first.dx, first.dy)
            if frame.value_range is not None:
                items[0].set_lut_range(frame.value_range)
                plot.update_colormap_axis(items[0])
        else:
            signature = (
                kind,
                tuple(
                    (obj.x.size, obj.x[0], obj.x[-1]) if obj.x.size else (0,)
                    for obj in frame.objects
                ),
            )
        if signature != self._plot_signature:
            plot.do_autoscale(replot=False)
            if kind == "curve" and frame.value_range is not None:
                plot.set_axis_limits("left", *frame.value_range)
        elif kind == "curve" and frame.value_range != self._value_range:
            # Like a new V/div on a scope: Y follows, the X zoom is kept
            if frame.value_range is None:
                plot.do_autoscale(replot=False, axis_id=plot.get_axis_id("left"))
            else:
                plot.set_axis_limits("left", *frame.value_range)
        self._plot_signature = signature
        self._value_range = frame.value_range
        self.summary_label.setText(frame.summary)
        plot.replot()

    def acquire(self) -> None:
        """Add an acquisition to the workspace, in a new group or in the current
        group of the panel, depending on the **New group** option.

        The option and the current group are taken now: changing them while the
        acquisition is running does not change its destination."""
        self._acquire_target = self.acquisition_target()
        self._request("acquire")

    def acquisition_target(self) -> AcquisitionTarget:
        """Return the destination of an acquisition requested now: state of the
        **New group** option and current group of each panel."""
        return AcquisitionTarget(
            self.new_group_box.isChecked(),
            {
                "signal": self.main.signalpanel.objview.get_current_group_id(),
                "image": self.main.imagepanel.objview.get_current_group_id(),
            },
        )

    def _add_acquired(
        self, acquisition: InstrumentAcquisition, target: AcquisitionTarget | None
    ) -> None:
        """Add the objects acquired by the instrument to the workspace."""
        try:
            group_title = self.add_acquisition(acquisition, target)
        except Exception as exc:  # pylint: disable=broad-except
            self.status_label.setText(str(exc))
            return
        self.status_label.setText(
            _("%d objects added to group '%s'")
            % (len(acquisition.objects), group_title)
        )

    def add_acquisition(
        self,
        acquisition: InstrumentAcquisition,
        target: AcquisitionTarget | None = None,
    ) -> str:
        """Add acquired objects to their panel.

        The objects are added to the current group of the panel if the **New
        group** option is off and the panel has a current group, otherwise to a
        new group. A new group is also created if the current group was removed
        since the acquisition was requested.

        Args:
            acquisition: acquired objects
            target: destination taken when the acquisition was requested
             (default: current state of the option and of the panel)

        Returns:
            Title of the group where the objects were added
        """
        if target is None:
            target = self.acquisition_target()
        objects = acquisition.objects
        panel_name = "signal" if isinstance(objects[0], SignalObj) else "image"
        panel = (
            self.main.signalpanel if panel_name == "signal" else self.main.imagepanel
        )
        with self.main.context_no_refresh():
            group_id = None if target.new_group else target.group_ids.get(panel_name)
            existing = {get_uuid(group) for group in panel.objmodel.get_groups()}
            if group_id not in existing:
                # No current group, or group removed since the request
                group_id = None
            if group_id is None:
                group = panel.add_group(acquisition.group_title)
                try:
                    # pylint: disable=protected-access
                    panel._add_objects(objects, get_uuid(group))
                except Exception:
                    panel.objview.remove_item(get_uuid(group), refresh=False)
                    panel.objmodel.remove_group(group)
                    raise
            else:
                group = panel.objmodel.get_group(group_id)
                # pylint: disable=protected-access
                panel._add_objects(objects, group_id)
        self.main.set_current_panel(panel_name)
        panel.objview.select_objects(objects)
        return group.title
