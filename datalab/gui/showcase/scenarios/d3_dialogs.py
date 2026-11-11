# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase scenario D3: dialog drivers
====================================

Helpers driving the modal dialogs of DataLab while they are executed, as a user
would do: ticking a check box, choosing a value in a list, typing a value,
dragging a slider, clicking on a plot, then on *OK*.

A modal dialog blocks the scenario until it is closed: the drivers are called by a
timer started before the action showing the dialog is triggered (see
:func:`drive_dialogs`).
"""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import sigima.objects
import sigima.params
import sigima.proc.image
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.qthelpers import is_qobject_valid
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import _
from datalab.env import execenv

if TYPE_CHECKING:
    from guidata.dataset.qtitemwidgets import AbstractDataSetWidget, SliderWidget
    from plotpy.plot import BasePlot

    from datalab.gui.main import DLMainWindow
    from datalab.gui.showcase.engine import ShowcaseContext
    from datalab.widgets.processingpreview import ProcessingPreviewDialog

logger = logging.getLogger("datalab.showcase")

#: Driver of a modal dialog: called with the dialog while it is executed, it is
#: expected to close it (e.g. by clicking on *OK*)
DialogDriver = Callable[[QW.QDialog], None]


@contextmanager
def dialog_delay(seconds: float) -> Generator[None, None, None]:
    """Set how long the demo mode shows the dialogs, within the context

    In demo mode, the dialogs executed with ``exec_dialog`` are closed (accepted)
    once ``execenv.delay`` has elapsed.

    Args:
        seconds: Delay (s) after which the demo mode closes the dialogs

    Yields:
        None
    """
    previous = execenv.delay
    execenv.delay = max(0, round(seconds * 1000))
    try:
        yield
    finally:
        execenv.delay = previous


def drive_dialogs(
    ctx: ShowcaseContext,
    trigger: Callable[[], None],
    drivers: Iterable[DialogDriver],
    timeout: float = 120.0,
) -> None:
    """Trigger an action showing modal dialogs, and drive each of them

    A timer is started before the (blocking) action is triggered: it waits for
    each modal dialog shown by the action, and calls the next driver with it while
    the dialog is executed. The driven dialogs are exempted from the modal-dialog
    guard of the showcase. As a safety net, the demo mode closes the dialogs left
    open after *timeout*.

    Args:
        ctx: Showcase context
        trigger: Callable triggering the action (e.g. ``ctx.trigger_action``)
        drivers: Drivers of the dialogs shown by the action, in this order
        timeout: Delay (s) after which the demo mode closes the dialogs

    Raises:
        RuntimeError: If a dialog is not shown, or if a driver fails
    """
    pending = list(drivers)
    driven: list[QW.QDialog] = []
    errors: list[Exception] = []
    busy = False

    def drive_next_dialog() -> None:
        """Drive the next modal dialog, once shown"""
        nonlocal busy
        if busy or not pending:
            return
        dialog = QW.QApplication.activeModalWidget()
        if (
            not isinstance(dialog, QW.QDialog)
            or not dialog.isVisible()
            or any(dialog is other for other in driven)
        ):
            return
        driven.append(dialog)
        driver = pending.pop(0)
        busy = True
        try:
            with ctx.unguarded(dialog):
                driver(dialog)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Showcase: cannot drive %s", type(dialog).__name__)
            errors.append(exc)
            if is_qobject_valid(dialog) and dialog.isVisible():
                dialog.reject()
        finally:
            busy = False

    timer = QC.QTimer(ctx.win)
    timer.setInterval(50)
    timer.timeout.connect(drive_next_dialog)
    timer.start()
    try:
        with dialog_delay(timeout):
            trigger()
    finally:
        timer.stop()
        timer.deleteLater()
    if errors:
        raise RuntimeError("A dialog could not be driven") from errors[0]
    if pending:
        raise RuntimeError(f"{len(pending)} expected dialog(s) not shown")


def get_edit_layouts(dialog: QW.QDialog) -> list[DataSetEditLayout]:
    """Return the edit layouts of a parameter dialog

    Args:
        dialog: Parameter dialog (``DataSetEditDialog`` or
         ``ProcessingPreviewDialog``)

    Returns:
        Edit layouts of the dialog
    """
    layouts = dialog.edit_layout
    return [layouts] if isinstance(layouts, DataSetEditLayout) else list(layouts)


def find_item_widget(
    layouts: DataSetEditLayout | Iterable[DataSetEditLayout], name: str
) -> AbstractDataSetWidget:
    """Return the widget editing a data item of a parameter form

    Args:
        layouts: Edit layout(s) of the form
        name: Name of the data item (e.g. ``'create_rois'``)

    Returns:
        Widget of the data item

    Raises:
        LookupError: If the form has no widget for this data item
    """
    if isinstance(layouts, DataSetEditLayout):
        layouts = [layouts]
    for layout in layouts:
        for widget in layout.get_terminal_widgets():
            if widget.item.item.get_name() == name:
                return widget
    raise LookupError(f"No widget for the data item {name!r}")


def click_button(
    ctx: ShowcaseContext,
    dialog: QW.QDialog,
    button: QW.QDialogButtonBox.StandardButton = QW.QDialogButtonBox.Ok,
) -> None:
    """Click on a standard button of a dialog (e.g. *OK*)

    Args:
        ctx: Showcase context
        dialog: Dialog
        button: Standard button

    Raises:
        LookupError: If the dialog has no such visible button
    """
    for box in dialog.findChildren(QW.QDialogButtonBox):
        widget = box.button(button)
        if widget is not None and widget.isVisible():
            ctx.click(widget)
            return
    raise LookupError(f"No such button in {type(dialog).__name__}")


def set_check_box(
    ctx: ShowcaseContext, dialog: QW.QDialog, name: str, checked: bool = True
) -> None:
    """Check (or uncheck) the check box of a boolean data item, with a click

    Args:
        ctx: Showcase context
        dialog: Parameter dialog
        name: Name of the boolean data item
        checked: State of the check box
    """
    checkbox: QW.QCheckBox = find_item_widget(get_edit_layouts(dialog), name).checkbox
    if checkbox.isChecked() != checked:
        ctx.click(checkbox)
    else:
        ctx.move_cursor_to(checkbox)
    ctx.pause(0.6)


def get_choice_index(widget: AbstractDataSetWidget, key: Any) -> int:
    """Return the index of a choice in the list of a choice data item

    Args:
        widget: Widget of the choice data item
        key: Key of the choice (an enum member, its name or its value for the
         choices of an enum)

    Returns:
        Index of the choice

    Raises:
        LookupError: If the data item has no such choice
    """
    keys = [key] + ([key.name, key.value] if isinstance(key, enum.Enum) else [])
    choices = widget.item.get_prop_value("data", "choices")
    for index, choice in enumerate(choices):
        if choice[0] in keys:
            return index
    raise LookupError(f"No choice {key!r} for {widget.item.item.get_name()!r}")


def choose(ctx: ShowcaseContext, dialog: QW.QDialog, name: str, key: Any) -> None:
    """Choose a value of a choice data item in its list, as a user would do

    Args:
        ctx: Showcase context
        dialog: Parameter dialog
        name: Name of the choice data item
        key: Key of the choice (see :func:`get_choice_index`)
    """
    widget = find_item_widget(get_edit_layouts(dialog), name)
    index = get_choice_index(widget, key)
    combobox: QW.QComboBox = widget.combobox
    ctx.move_cursor_to(combobox)
    combobox.showPopup()
    ctx.pause(0.7)
    combobox.setCurrentIndex(index)
    ctx.pause(0.3)
    combobox.hidePopup()
    ctx.pause(0.4)


def type_value(ctx: ShowcaseContext, dialog: QW.QDialog, name: str, text: str) -> None:
    """Type the value of a data item edited in a line editor

    Args:
        ctx: Showcase context
        dialog: Parameter dialog
        name: Name of the data item
        text: Value, as typed by the user
    """
    editor: QW.QLineEdit = find_item_widget(get_edit_layouts(dialog), name).edit
    editor.clear()
    ctx.type_text(editor, text, cps=8.0)
    ctx.pause(0.5)


def get_slider_handle_center(slider: QW.QSlider) -> QC.QPoint:
    """Return the center of the handle of a horizontal slider (widget coordinates)

    Args:
        slider: Slider

    Returns:
        Approximate center of the slider handle
    """
    style = slider.style()
    length = style.pixelMetric(QW.QStyle.PM_SliderLength, None, slider)
    span = max(0, slider.width() - length)
    offset = QW.QStyle.sliderPositionFromValue(
        slider.minimum(), slider.maximum(), slider.value(), span
    )
    return QC.QPoint(offset + length // 2, slider.height() // 2)


def drag_slider(
    ctx: ShowcaseContext,
    widget: SliderWidget,
    values: Iterable[float],
    duration: float = 2.0,
    hold: float = 1.0,
) -> None:
    """Drag the slider of a numeric data item through values, one after the other

    The slider is held down during the whole gesture, as with the mouse: the
    live preview of a processing parameter dialog is updated while dragging.

    Args:
        ctx: Showcase context
        widget: Widget of the numeric data item, with a slider
        values: Values reached one after the other, from the current value
        duration: Duration (s) of the move from a value to the next one, at
         normal pace
        hold: Pause (s) on each reached value, at normal pace
    """
    slider: QW.QSlider = widget.slider
    positions = [slider.value()] + [widget.value_to_slider(v) for v in values]
    ctx.move_cursor_to(slider, get_slider_handle_center(slider))
    slider.setSliderDown(True)
    try:
        for start, end in zip(positions, positions[1:]):
            count = max(2, round(duration * 20))
            for index in range(1, count + 1):
                ratio = index / count
                eased = ratio * ratio * (3.0 - 2.0 * ratio)
                slider.setValue(round(start + (end - start) * eased))
                ctx.move_cursor_to(slider, get_slider_handle_center(slider), 0.0)
                ctx.pause(duration / count)
            ctx.pause(hold)
    finally:
        slider.setSliderDown(False)


def send_mouse_event(
    widget: QW.QWidget,
    event_type: QC.QEvent.Type,
    pos: QC.QPoint,
    buttons: QC.Qt.MouseButtons,
) -> None:
    """Send a mouse event of the left button to a widget

    Args:
        widget: Widget
        event_type: Mouse event type (press, move or release)
        pos: Position, in widget coordinates
        buttons: Mouse buttons held down during the event
    """
    button = QC.Qt.NoButton if event_type == QC.QEvent.MouseMove else QC.Qt.LeftButton
    event = QG.QMouseEvent(
        event_type, QC.QPointF(pos), button, buttons, QC.Qt.NoModifier
    )
    QW.QApplication.sendEvent(widget, event)


def click_plot(ctx: ShowcaseContext, plot: BasePlot, x: float, y: float) -> None:
    """Click on a plot, at a point given in axis coordinates

    The plot tools (e.g. the cross section tool) create their shapes once the
    mouse is moved with the button down: the button is pressed, the mouse is
    moved slightly, then the button is released where it was pressed.

    Args:
        ctx: Showcase context
        plot: Plot
        x: X coordinate (bottom axis)
        y: Y coordinate (left axis)
    """
    canvas = plot.canvas()
    xpix = plot.transform(plot.get_axis_id("bottom"), x)
    ypix = plot.transform(plot.get_axis_id("left"), y)
    pos = QC.QPoint(round(xpix), round(ypix))
    ctx.move_cursor_to(canvas, pos)
    send_mouse_event(canvas, QC.QEvent.MouseButtonPress, pos, QC.Qt.LeftButton)
    for offset in (1, 2, 1, 0):
        moved = pos + QC.QPoint(offset, 0)
        send_mouse_event(canvas, QC.QEvent.MouseMove, moved, QC.Qt.LeftButton)
        ctx.pause(0.05)
    send_mouse_event(canvas, QC.QEvent.MouseButtonRelease, pos, QC.Qt.NoButton)
    QW.QApplication.processEvents()


def zoom_plot(
    ctx: ShowcaseContext,
    plot: BasePlot,
    xmin: float,
    xmax: float,
    duration: float = 1.2,
) -> None:
    """Zoom smoothly on an interval of the bottom axis of a plot

    Args:
        ctx: Showcase context
        plot: Plot
        xmin: Lower bound of the interval
        xmax: Upper bound of the interval
        duration: Duration (s) of the zoom, at normal pace
    """
    axis = plot.get_axis_id("bottom")
    vmin, vmax = plot.get_axis_limits(axis)
    count = max(2, round(duration * 25))
    for index in range(1, count + 1):
        ratio = index / count
        eased = ratio * ratio * (3.0 - 2.0 * ratio)
        plot.set_axis_limits(
            axis, vmin + (xmin - vmin) * eased, vmax + (xmax - vmax) * eased
        )
        plot.replot()
        ctx.pause(duration / count)


class PreviewWatcher(QC.QObject):
    """Watch the live preview of a processing parameter dialog

    Args:
        dialog: Processing parameter dialog with a live preview
    """

    #: Duration (s) during which the preview must stay up to date, once updated
    STABLE_DURATION = 0.5

    def __init__(self, dialog: ProcessingPreviewDialog) -> None:
        super().__init__(dialog)
        self.preview = dialog.preview
        #: Number of preview results shown
        self.count = 0
        self.preview.controller.SIG_RESULT.connect(self.__count_result)

    def __count_result(self, *_args: Any) -> None:
        """Count a preview result"""
        self.count += 1

    def is_up_to_date(self) -> bool:
        """Return True if the preview matches the current parameters"""
        return self.preview.status.text() == _("Preview up to date")

    def wait_for_update(
        self, ctx: ShowcaseContext, since: int, timeout: float = 30.0
    ) -> None:
        """Wait until the preview is updated, and matches the current parameters

        The first preview of a session may take a few seconds: the preview runs
        in a separate process, started on demand (see :func:`warm_up_preview`).

        Args:
            ctx: Showcase context
            since: Number of preview results shown before the update
            timeout: Maximum waiting time (s)

        Raises:
            RuntimeError: If the preview is not updated in time
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            if not ctx.wait_until(
                lambda: self.count > since and self.is_up_to_date(), remaining
            ):
                raise RuntimeError("The live preview was not updated")
            count = self.count
            ctx.wait_until(
                lambda: self.count != count or not self.is_up_to_date(),
                self.STABLE_DURATION,
            )
            if self.count == count and self.is_up_to_date():
                return


def enable_preview(
    ctx: ShowcaseContext, dialog: ProcessingPreviewDialog
) -> PreviewWatcher:
    """Enable the live preview of a processing parameter dialog, with a click

    Args:
        ctx: Showcase context
        dialog: Processing parameter dialog with a live preview

    Returns:
        Watcher of the preview, once the first preview is shown
    """
    watcher = PreviewWatcher(dialog)
    checkbox = dialog.preview.enabled
    if not checkbox.isChecked():
        ctx.click(checkbox)
        watcher.wait_for_update(ctx, since=0)
    return watcher


def warm_up_preview(win: DLMainWindow) -> None:
    """Start the process computing the live previews, in the background

    The live previews are computed in a separate process, started on demand: the
    first preview of a session takes a few seconds. This process is started (and
    the computing modules are imported) beforehand, then kept for the previews.

    Args:
        win: DataLab main window
    """
    cache = win.preview_executor_cache
    executor = cache.acquire()
    try:
        image = sigima.objects.create_image("", np.zeros((8, 8)))
        param = sigima.params.ButterworthParam()
        executor.submit(sigima.proc.image.butterworth, (image, param))
    finally:
        cache.release(executor)
