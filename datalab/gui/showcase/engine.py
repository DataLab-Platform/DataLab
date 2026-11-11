# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Showcase engine
===============

The :mod:`datalab.gui.showcase.engine` module provides the building blocks of the
showcase mode: live and unattended demos of DataLab, for exhibitions and booths,
written like end-to-end test scenarios.

A scenario is a :class:`ShowcaseScenario` whose ``run`` callable drives the real
DataLab user interface through a :class:`ShowcaseContext`. The same scenario
runs in two modes:

- At *showcase speed* (``speed > 0``): human pace, captions, title cards,
  spotlights, animated mouse pointer and visible menu walks.
- In *check mode* (``speed == 0``): no delay and no visual effect, to check that
  the scenario still works (e.g. in the test suite, on the offscreen platform).

Captions follow the language of the user interface: write them with ``_()``.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import sigima.proc.signal as sips
from guidata.configtools import get_icon
from guidata.qthelpers import is_qobject_valid
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.objects import create_signal

from datalab.config import _
from datalab.env import execenv
from datalab.gui.processor.base import WorkerState
from datalab.gui.tour import Cover

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

    from datalab.gui.main import DLMainWindow
    from datalab.gui.panel.base import BaseDataPanel
    from datalab.gui.processor.base import Worker

logger = logging.getLogger("datalab.showcase")

#: Qt platforms without any screen: the mouse pointer is not animated
HEADLESS_PLATFORMS = ("offscreen", "minimal")


def _clamp(value: float, vmin: float, vmax: float) -> float:
    """Return a value clamped to an interval

    Args:
        value: Value to clamp
        vmin: Lower bound
        vmax: Upper bound

    Returns:
        Clamped value
    """
    return max(vmin, min(vmax, value))


def is_headless() -> bool:
    """Return True if the Qt platform has no screen (e.g. offscreen tests)"""
    return QG.QGuiApplication.platformName() in HEADLESS_PLATFORMS


def process_events() -> None:
    """Process the pending events, then delete the objects scheduled for deletion

    The showcase never returns to the main event loop, which deletes the objects
    scheduled for deletion (``deleteLater``): like ``QTest::qWait``, the waiting
    loops of the showcase delete them, otherwise they would linger (e.g. replaced
    widgets painted over the new ones).
    """
    QW.QApplication.processEvents()
    QW.QApplication.sendPostedEvents(None, QC.QEvent.DeferredDelete)


def plain_text(text: str) -> str:
    """Return a menu or action text without its mnemonic markers

    Args:
        text: Action text (e.g. ``'&File'``)

    Returns:
        Text without mnemonic markers (e.g. ``'File'``)
    """
    marker = chr(0)
    return text.replace("&&", marker).replace("&", "").replace(marker, "&")


class ScenarioSkipped(Exception):
    """Raised when a scenario cannot be played

    The showcase skips the scenario, with a warning in the log (e.g. when a module
    or a plugin required by the scenario is not available).
    """


@dataclass
class ShowcaseScenario:
    """Showcase scenario

    Args:
        id: Short identifier (e.g. ``'d1'``), used to select the scenarios to play
         (e.g. with the ``--showcase`` command-line option)
        title: Title, shown on the title card before the scenario
        run: Callable playing the scenario with a :class:`ShowcaseContext`
        subtitle: Subtitle of the title card
        requires: Names of the modules required by the scenario: if one of them
         is not available, the scenario is skipped
        duration_hint: Approximate duration (seconds) at normal pace
    """

    id: str  # pylint: disable=invalid-name
    title: str
    run: Callable[[ShowcaseContext], None]
    subtitle: str | None = None
    requires: tuple[str, ...] = ()
    duration_hint: float | None = None

    def missing_requirements(self) -> list[str]:
        """Return the required modules which are not available

        Returns:
            Names of the missing modules
        """
        missing = []
        for name in self.requires:
            try:
                found = importlib.util.find_spec(name) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing.append(name)
        return missing

    def check_requirements(self) -> None:
        """Check that the modules required by the scenario are available

        Raises:
            ScenarioSkipped: If a required module is not available
        """
        missing = self.missing_requirements()
        if missing:
            names = ", ".join(missing)
            raise ScenarioSkipped(
                f"Scenario {self.id} requires missing modules: {names}"
            )


class _Overlay(QW.QWidget):
    """Base class of the showcase overlays, painted over the main window

    Overlays are child widgets of the main window: they never show up as separate
    windows. They let the mouse events through and follow the main window size.

    Args:
        win: Main window
    """

    def __init__(self, win: QW.QMainWindow) -> None:
        super().__init__(win)
        self.setAttribute(QC.Qt.WA_TransparentForMouseEvents)
        self.setFocusPolicy(QC.Qt.NoFocus)
        win.installEventFilter(self)

    def eventFilter(self, obj: QC.QObject, event: QC.QEvent) -> bool:
        """Follow the size of the main window

        Args:
            obj: Watched object
            event: Event

        Returns:
            False: events are never filtered out
        """
        if (
            obj is self.parentWidget()
            and event.type() == QC.QEvent.Resize
            and self.isVisible()
        ):
            self.relayout()
        return False

    def relayout(self) -> None:
        """Update the geometry of the overlay from the main window size"""
        self.setGeometry(self.parentWidget().rect())

    def scaled_font(
        self, ratio: float, vmin: float, vmax: float, bold: bool = False
    ) -> QG.QFont:
        """Return a font whose size follows the main window height

        Args:
            ratio: Font size (points) per pixel of main window height
            vmin: Minimum font size (points)
            vmax: Maximum font size (points)
            bold: True for a bold font

        Returns:
            Font
        """
        font = QG.QFont(self.font())
        height = self.parentWidget().height()
        font.setPointSizeF(_clamp(height * ratio, vmin, vmax))
        font.setBold(bold)
        return font

    def popup(self) -> None:
        """Show the overlay over the other widgets of the main window"""
        self.relayout()
        self.show()
        self.raise_()
        self.update()


#: Placement of the caption banner in the main window: top, left and width (None:
#: default value, i.e. bottom of the main window, centered, 80% of its width)
Placement = tuple[int | None, int | None, int | None]


class CaptionBanner(_Overlay):
    """Caption banner, shown at the bottom of the main window

    The banner shows a header line (scenario title and step number) above the
    caption text. It may be moved elsewhere in the main window, and narrowed (e.g.
    while a dialog covers the bottom of the main window), see :meth:`set_placement`.
    """

    PADDING = 22
    SPACING = 6
    FLAGS = QC.Qt.AlignHCenter | QC.Qt.TextWordWrap
    #: Default placement: at the bottom of the main window, centered
    DEFAULT_PLACEMENT: Placement = (None, None, None)

    def __init__(self, win: QW.QMainWindow) -> None:
        super().__init__(win)
        self.__text = ""
        self.__header = ""
        self.__placement: Placement = self.DEFAULT_PLACEMENT

    @property
    def text(self) -> str:
        """Caption text"""
        return self.__text

    @property
    def placement(self) -> Placement:
        """Placement of the banner (see :meth:`set_placement`)"""
        return self.__placement

    @property
    def moved(self) -> bool:
        """True if the banner is not at its default placement"""
        return self.__placement != self.DEFAULT_PLACEMENT

    def top_limit(self) -> int:
        """Return the highest top of the banner: just below the menu bar"""
        win = self.parentWidget()
        return win.menuBar().geometry().bottom() + int(win.height() * 0.02)

    def default_width(self) -> int:
        """Return the default width of the banner"""
        return int(min(self.parentWidget().width() * 0.8, 1500.0))

    def set_placement(
        self, top: int | None = None, left: int | None = None, width: int | None = None
    ) -> None:
        """Move (and narrow) the banner in the main window

        Args:
            top: Top of the banner in the main window (default: at the bottom of the
             main window)
            left: Left of the banner in the main window (default: centered)
            width: Width of the banner (default: :meth:`default_width`)
        """
        placement = (top, left, width)
        if placement != self.__placement:
            self.__placement = placement
            if self.isVisible():
                self.relayout()

    def set_text(self, text: str, header: str = "") -> None:
        """Set the caption

        Args:
            text: Caption text
            header: Header line, shown above the text
        """
        self.__text = text
        self.__header = header
        if self.isVisible():
            self.relayout()
        self.update()

    def __fonts(self) -> tuple[QG.QFont, QG.QFont]:
        """Return the fonts of the header and of the text"""
        text_font = self.scaled_font(0.02, 14.0, 24.0)
        header_font = QG.QFont(text_font)
        header_font.setPointSizeF(text_font.pointSizeF() * 0.62)
        header_font.setBold(True)
        return header_font, text_font

    def geometry_at(
        self, top: int | None = None, left: int | None = None, width: int | None = None
    ) -> QC.QRect:
        """Return the geometry of the banner at a given placement

        Args:
            top: Top of the banner in the main window (default: at the bottom of the
             main window)
            left: Left of the banner in the main window (default: centered)
            width: Width of the banner (default: :meth:`default_width`)

        Returns:
            Geometry, in main window coordinates (the height follows the width)
        """
        win = self.parentWidget()
        header_font, text_font = self.__fonts()
        width = self.default_width() if width is None else width
        inner = QC.QRect(0, 0, max(width - 2 * self.PADDING, 1), 10000)
        metrics = QG.QFontMetrics(text_font)
        height = metrics.boundingRect(inner, self.FLAGS, self.__text).height()
        height += 2 * self.PADDING
        if self.__header:
            height += QG.QFontMetrics(header_font).height() + self.SPACING
        left = (win.width() - width) // 2 if left is None else left
        if top is not None:
            return QC.QRect(left, top, width, height)
        bottom = win.height() - int(win.height() * 0.06)
        status_bar = win.statusBar()
        if status_bar.isVisible():
            bottom -= status_bar.height()
        return QC.QRect(left, bottom - height, width, height)

    def relayout(self) -> None:
        """Place the banner in the main window (at the bottom, by default)"""
        self.setGeometry(self.geometry_at(*self.__placement))

    def paintEvent(self, event: QG.QPaintEvent) -> None:  # pylint: disable=W0613
        """Paint the banner"""
        header_font, text_font = self.__fonts()
        painter = QG.QPainter(self)
        painter.setRenderHint(QG.QPainter.Antialiasing)
        painter.setPen(QC.Qt.NoPen)
        painter.setBrush(QG.QColor(16, 22, 34, 225))
        painter.drawRoundedRect(QC.QRectF(self.rect()), 18.0, 18.0)
        pad = self.PADDING
        rect = self.rect().adjusted(pad, pad, -pad, -pad)
        if self.__header:
            painter.setFont(header_font)
            painter.setPen(QG.QColor(130, 185, 255))
            header_height = QG.QFontMetrics(header_font).height()
            header_rect = QC.QRect(rect.left(), rect.top(), rect.width(), header_height)
            header = QG.QFontMetrics(header_font).elidedText(
                self.__header, QC.Qt.ElideMiddle, rect.width()
            )
            painter.drawText(header_rect, QC.Qt.AlignCenter, header)
            rect.setTop(rect.top() + header_height + self.SPACING)
        painter.setFont(text_font)
        painter.setPen(QG.QColor(255, 255, 255))
        painter.drawText(rect, self.FLAGS, self.__text)
        painter.end()


class TitleCard(_Overlay):
    """Card covering the main window: scenario title card, or end card"""

    FLAGS = QC.Qt.AlignHCenter | QC.Qt.TextWordWrap

    def __init__(self, win: QW.QMainWindow) -> None:
        super().__init__(win)
        self.__title = ""
        self.__subtitle = ""
        self.__footer = ""

    def set_content(self, title: str, subtitle: str = "", footer: str = "") -> None:
        """Set the content of the card

        Args:
            title: Title
            subtitle: Subtitle
            footer: Footer
        """
        self.__title, self.__subtitle, self.__footer = title, subtitle, footer
        self.update()

    def paintEvent(self, event: QG.QPaintEvent) -> None:  # pylint: disable=W0613
        """Paint the card"""
        painter = QG.QPainter(self)
        painter.setRenderHint(QG.QPainter.Antialiasing)
        painter.setRenderHint(QG.QPainter.SmoothPixmapTransform)
        width, height = self.width(), self.height()
        gradient = QG.QLinearGradient(0.0, 0.0, 0.0, float(height))
        gradient.setColorAt(0.0, QG.QColor(14, 32, 64, 248))
        gradient.setColorAt(1.0, QG.QColor(6, 10, 22, 248))
        painter.fillRect(self.rect(), QG.QBrush(gradient))
        logo_size = int(_clamp(height * 0.16, 64.0, 220.0))
        top = int(height * 0.14)
        pixmap = get_icon("DataLab.svg").pixmap(logo_size, logo_size)
        logo_rect = QC.QRect((width - logo_size) // 2, top, logo_size, logo_size)
        painter.drawPixmap(logo_rect, pixmap)
        top += logo_size + int(height * 0.06)
        margin = int(width * 0.1)
        lines = (
            (self.__title, self.scaled_font(0.045, 24.0, 52.0, True), (255, 255, 255)),
            (self.__subtitle, self.scaled_font(0.026, 16.0, 30.0), (140, 195, 255)),
            (self.__footer, self.scaled_font(0.02, 12.0, 24.0), (200, 205, 215)),
        )
        for text, font, color in lines:
            if not text:
                continue
            painter.setFont(font)
            painter.setPen(QG.QColor(*color))
            bounds = QC.QRect(margin, top, width - 2 * margin, max(height - top, 1))
            used = painter.boundingRect(bounds, self.FLAGS, text)
            painter.drawText(bounds, self.FLAGS, text)
            top += used.height() + int(height * 0.035)
        painter.end()


class StatusBadge(_Overlay):
    """Badge shown at the top right corner of the main window (e.g. pause)"""

    def __init__(self, win: QW.QMainWindow) -> None:
        super().__init__(win)
        self.__text = ""

    def set_text(self, text: str) -> None:
        """Set the badge text

        Args:
            text: Text
        """
        self.__text = text
        if self.isVisible():
            self.relayout()
        self.update()

    def __font(self) -> QG.QFont:
        """Return the font of the badge"""
        return self.scaled_font(0.016, 11.0, 18.0, True)

    def relayout(self) -> None:
        """Place the badge at the top right corner of the main window"""
        win = self.parentWidget()
        metrics = QG.QFontMetrics(self.__font())
        width = metrics.horizontalAdvance(self.__text) + 2 * metrics.height()
        height = int(metrics.height() * 1.8)
        top = win.menuBar().geometry().bottom() + 16
        self.setGeometry(win.width() - width - 24, top, width, height)

    def paintEvent(self, event: QG.QPaintEvent) -> None:  # pylint: disable=W0613
        """Paint the badge"""
        painter = QG.QPainter(self)
        painter.setRenderHint(QG.QPainter.Antialiasing)
        painter.setPen(QC.Qt.NoPen)
        painter.setBrush(QG.QColor(240, 150, 30, 235))
        painter.drawRoundedRect(QC.QRectF(self.rect()), 10.0, 10.0)
        painter.setFont(self.__font())
        painter.setPen(QG.QColor(20, 20, 20))
        painter.drawText(self.rect(), QC.Qt.AlignCenter, self.__text)
        painter.end()


class _Hotkeys(QC.QObject):
    """Application event filter handling the operator hotkeys of the showcase

    - F9: pause or resume the showcase
    - Shift+F12, or Escape twice quickly: stop the showcase after the current
      scenario

    Args:
        ctx: Showcase context
    """

    #: Maximum interval (s) between the two presses of a double Escape
    DOUBLE_ESCAPE_INTERVAL = 0.8

    def __init__(self, ctx: ShowcaseContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.__last_press: tuple[int, int] | None = None
        self.__last_escape = 0.0

    def __is_new_press(self, event: QG.QKeyEvent) -> bool:
        """Return True for the first delivery of a key press

        An application event filter sees a key press once per receiver, while the
        event is propagated to the parent widgets.
        """
        press = (event.timestamp(), event.key())
        if press[0] and press == self.__last_press:
            return False
        self.__last_press = press
        return True

    def eventFilter(self, obj: QC.QObject, event: QC.QEvent) -> bool:
        """Handle the operator hotkeys

        Args:
            obj: Receiver of the event
            event: Event

        Returns:
            True if the event is consumed
        """
        if (
            event.type() != QC.QEvent.KeyPress
            or not obj.isWidgetType()
            or event.isAutoRepeat()
        ):
            return False
        key = event.key()
        if key == QC.Qt.Key_F9:
            if self.__is_new_press(event):
                self.ctx.toggle_pause()
            return True
        if key == QC.Qt.Key_F12 and event.modifiers() & QC.Qt.ShiftModifier:
            if self.__is_new_press(event):
                self.ctx.request_stop()
            return True
        if key == QC.Qt.Key_Escape and self.__is_new_press(event):
            now = time.monotonic()
            if now - self.__last_escape <= self.DOUBLE_ESCAPE_INTERVAL:
                self.__last_escape = 0.0
                self.ctx.request_stop()
            else:
                self.__last_escape = now
        return False


class ShowcaseContext:
    """Context of a showcase scenario, driving the DataLab user interface

    At showcase speed (``speed > 0``), the context shows captions, title cards and
    spotlights, moves the mouse pointer and opens the menus before triggering
    their actions. In check mode (``speed == 0``), it performs the same actions and
    checks, without any delay or visual effect.

    While it is started (see :meth:`start`, or use it as a context manager), the
    context runs a *modal-dialog guard*: a modal dialog which is not closed by the
    demo mode of DataLab (e.g. a message box shown directly by a plugin) is
    accepted after a while, so that the showcase never gets stuck.

    Args:
        win: DataLab main window
        speed: Delay factor: 1 for the normal pace, 1.5 for a slower pace, 0.7 for
         a faster pace, 0 for check mode

    Raises:
        ValueError: If the speed is negative
    """

    #: Interval (ms) between two checks of the modal-dialog guard
    GUARD_INTERVAL_MS = 250
    #: Space (pixels) between the caption and a window moved to keep it visible
    CAPTION_MARGIN = 8
    #: Minimum delay (s) between the closing of a dialog by the demo mode and its
    #: acceptance by the modal-dialog guard (at normal pace)
    GUARD_GRACE = 2.0
    #: Maximum waiting time (s) for the warm-up of the processing worker
    WARM_UP_TIMEOUT = 60.0

    def __init__(self, win: DLMainWindow, speed: float = 1.0) -> None:
        if speed < 0:
            raise ValueError("Showcase speed must be positive or zero")
        self.win = win
        self.speed = float(speed)
        #: Scenario being played (see :meth:`begin_scenario`)
        self.scenario: ShowcaseScenario | None = None
        #: Number of captions shown since the beginning of the scenario
        self.step = 0
        #: True while the operator pauses the showcase (F9)
        self.paused = False
        #: True once the operator asked to stop the showcase (Shift+F12)
        self.stop_requested = False
        self.__caption: CaptionBanner | None = None
        self.__card: TitleCard | None = None
        self.__badge: StatusBadge | None = None
        self.__covers: list[tuple[Cover, list[QW.QWidget]]] = []
        self.__guard: QC.QTimer | None = None
        self.__guarded: dict[int, tuple[QW.QWidget, float, float]] = {}
        self.__unguarded: list[QW.QWidget] = []
        self.__hotkeys: _Hotkeys | None = None
        self.__warm_up_worker: Worker | None = None

    # ------ Mode and life cycle ---------------------------------------------------
    @property
    def check(self) -> bool:
        """True in check mode (``speed == 0``): no delay and no visual effect"""
        return self.speed == 0

    @property
    def caption_banner(self) -> CaptionBanner | None:
        """Caption banner, once a caption was shown at showcase speed"""
        return self.__caption

    @property
    def modal_timeout(self) -> float:
        """Delay (s) after which the guard accepts a modal dialog left open

        The delay follows the showcase speed, as the pauses. A dialog (except a
        message box) is never accepted before the demo mode of DataLab closes it
        (see :meth:`guard_timeout`).
        """
        return max(4.0, 6.0 * self.speed)

    def guard_timeout(self, widget: QW.QWidget) -> float:
        """Return the delay (s) after which the guard accepts a modal widget

        A message box is accepted after :attr:`modal_timeout`. A dialog is accepted
        after :attr:`modal_timeout` too, but never before the demo mode of DataLab
        had the opportunity to close it (if it was shown with ``exec_dialog``), even
        if the scenario lengthened the delay of the demo mode: the delay of the demo
        mode, when the dialog is shown, is followed by a grace period.

        Args:
            widget: Modal dialog or message box, just shown

        Returns:
            Delay (s)
        """
        timeout = self.modal_timeout
        if isinstance(widget, QW.QDialog) and not isinstance(widget, QW.QMessageBox):
            grace = self.GUARD_GRACE * max(self.speed, 1.0)
            timeout = max(timeout, execenv.delay / 1000.0 + grace)
        return timeout

    def start(self) -> None:
        """Start the modal-dialog guard

        While the guard runs, a modal dialog (or a message box) which stays open
        longer than :attr:`modal_timeout` is accepted, and logged: the dialogs
        shown with ``exec_dialog`` are closed earlier by the demo mode of DataLab,
        but other ones (e.g. ``QMessageBox.warning`` or ``dialog.exec()``) are not.
        A dialog is not accepted before the demo mode had the opportunity to close
        it, even if the scenario lengthened the delay of the demo mode. The dialogs
        driven by the scenario are exempted from the guard (see :meth:`unguarded`).

        The guard also keeps the overlays above the docks, and keeps the captions
        visible while other windows cover them (see :meth:`say`).
        """
        if self.__guard is None:
            self.__guard = QC.QTimer(self.win)
            self.__guard.setInterval(self.GUARD_INTERVAL_MS)
            self.__guard.timeout.connect(self.__check_modal_widgets)
            self.__guard.timeout.connect(self.__keep_overlays_on_top)
            self.__guard.timeout.connect(self.__place_caption)
        self.__guarded = {}
        self.__guard.start()

    def stop(self) -> None:
        """Stop the modal-dialog guard and the hotkeys, and hide the overlays"""
        self.finish_warm_up()
        if self.__guard is not None:
            self.__guard.stop()
        self.__guarded = {}
        self.disable_hotkeys()
        for overlay in (self.__caption, self.__card, self.__badge):
            if overlay is not None:
                overlay.hide()

    def __enter__(self) -> ShowcaseContext:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def enable_hotkeys(self) -> None:
        """Enable the operator hotkeys

        - **F9**: pause or resume the showcase
        - **Shift+F12**, or **Escape** twice quickly: stop the showcase after the
          current scenario
        """
        if self.__hotkeys is None:
            self.__hotkeys = _Hotkeys(self)
            QW.QApplication.instance().installEventFilter(self.__hotkeys)

    def disable_hotkeys(self) -> None:
        """Disable the operator hotkeys"""
        if self.__hotkeys is not None:
            QW.QApplication.instance().removeEventFilter(self.__hotkeys)
            self.__hotkeys = None

    def toggle_pause(self) -> None:
        """Pause or resume the showcase (F9 hotkey)

        The pause takes effect at the next step of the scenario (caption, pause...).
        """
        self.paused = not self.paused and not self.stop_requested
        logger.info(
            "Showcase %s by the operator", "paused" if self.paused else "resumed"
        )
        self.__update_badge()

    def request_stop(self) -> None:
        """Stop the showcase after the current scenario (Shift+F12 hotkey)"""
        if not self.stop_requested:
            self.stop_requested = True
            self.paused = False
            logger.info("Showcase stop requested by the operator")
            self.__update_badge()

    def __update_badge(self) -> None:
        """Show the state of the showcase (paused or stopping) to the operator"""
        if self.check:
            return
        if self.stop_requested:
            text = _("Stopping after this demo")
        elif self.paused:
            text = _("Paused (F9 to resume)")
        else:
            if self.__badge is not None:
                self.__badge.hide()
            return
        if self.__badge is None:
            self.__badge = StatusBadge(self.win)
        self.__badge.set_text(text)
        self.__badge.popup()

    def begin_scenario(self, scenario: ShowcaseScenario) -> None:
        """Begin a scenario (the scenario title is shown above the captions)

        Args:
            scenario: Scenario
        """
        self.scenario = scenario
        self.step = 0

    def end_scenario(self) -> None:
        """End the current scenario"""
        self.hide_caption()
        self.scenario = None
        self.step = 0

    # ------ Processing worker warm-up ----------------------------------------------
    def start_warm_up(self) -> None:
        """Start warming the processing worker up, in the background

        With process isolation (default setting), the first computation of a
        DataLab session waits for the processing worker to start (a few seconds,
        with a progress dialog). Instead, a tiny computation is started here, while
        the first title card is shown: the title card waits for its end (see
        :meth:`finish_warm_up`). Nothing is done in check mode, or once the worker
        was warmed up.
        """
        if self.check or self.__warm_up_worker is not None:
            return
        worker = self.win.signalpanel.processor.worker
        if worker is None or worker.state_machine.current_state != WorkerState.IDLE:
            return
        signal = create_signal("warm-up", np.arange(2.0), np.zeros(2))
        try:
            worker.run(sips.absolute, (signal,))
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Showcase: the processing worker cannot warm up")
            return
        logger.info("Showcase: warming the processing worker up")
        self.__warm_up_worker = worker

    def finish_warm_up(self) -> None:
        """Wait for the end of the warm-up of the processing worker, if started

        After :attr:`WARM_UP_TIMEOUT`, the warm-up computation is cancelled.
        """
        worker = self.__warm_up_worker
        if worker is None:
            return
        self.__warm_up_worker = None
        start = time.monotonic()
        if self.wait_until(worker.is_computation_finished, self.WARM_UP_TIMEOUT):
            if worker.has_result_available():
                worker.get_result()
            logger.info(
                "Showcase: processing worker ready after %.1f s",
                time.monotonic() - start,
            )
        else:
            logger.warning("Showcase: the processing worker did not warm up in time")
            worker.restart()

    # ------ Timing -----------------------------------------------------------------
    def pause(self, seconds: float = 1.0) -> None:
        """Pause the scenario while processing the events of the user interface

        Args:
            seconds: Duration at normal pace, scaled by the showcase speed (in check
             mode, pending events are processed once)
        """
        if self.speed > 0:
            self.__wait(seconds * self.speed)
        else:
            process_events()
        self.wait_while_paused()

    def wait_while_paused(self) -> None:
        """Wait while the operator pauses the showcase (F9)"""
        while self.paused and not self.stop_requested:
            process_events()
            time.sleep(0.02)

    def wait_until(self, condition: Callable[[], bool], timeout: float = 10.0) -> bool:
        """Process events until a condition is met, whatever the speed

        Args:
            condition: Callable returning True when the condition is met
            timeout: Maximum waiting time (s)

        Returns:
            True if the condition is met, False after the timeout
        """
        end = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= end:
                return False
            process_events()
            time.sleep(0.01)
        return True

    def __wait(self, duration: float) -> None:
        """Process events during a duration (s), frozen while paused"""
        end = time.monotonic() + duration
        while True:
            process_events()
            if self.paused:
                remaining = end - time.monotonic()
                self.wait_while_paused()
                end = time.monotonic() + max(remaining, 0.0)
            now = time.monotonic()
            if now >= end:
                return
            time.sleep(min(0.01, end - now))

    # ------ Captions and cards -------------------------------------------------------
    def say(self, text: str, hold: float = 2.5) -> None:
        """Show a caption, then pause

        The caption stays shown until the next caption or card (or until
        :meth:`hide_caption` is called). In check mode, it is only logged. The
        caption is shown at the bottom of the main window, unless another window
        (e.g. a dialog) covers it there: it is then moved where it is the least
        covered (top of the main window, above or below the other window, or
        narrowed beside it), and a modal dialog still covering it is moved down, if
        the main window has room for both.

        Args:
            text: Caption text (translated with ``_()``)
            hold: Pause duration (s) at normal pace
        """
        self.step += 1
        logger.info("Showcase caption %d: %s", self.step, text)
        if not self.check:
            if self.__caption is None:
                self.__caption = CaptionBanner(self.win)
            self.__caption.set_text(text, self.__caption_header())
            self.__caption.popup()
            self.__place_caption()
            if self.__covers:
                cover, targets = self.__covers[-1]
                cover.exclude(targets + [self.__caption])
                cover.update_geometry()
        self.pause(hold)

    def hide_caption(self) -> None:
        """Hide the caption"""
        if self.__caption is not None:
            self.__caption.hide()

    def __covering_windows(self) -> list[QW.QWidget]:
        """Return the visible windows which may cover the main window (e.g. dialogs,
        floating docks or tool windows), except menus, tooltips and spotlights"""
        windows = []
        for widget in QW.QApplication.topLevelWidgets():
            if (
                widget is self.win
                or not is_qobject_valid(widget)
                or not widget.isVisible()
                or isinstance(widget, (QW.QMenu, Cover))
                or widget.windowType() in (QC.Qt.ToolTip, QC.Qt.Popup)
            ):
                continue
            windows.append(widget)
        return windows

    def __place_caption(self) -> None:
        """Keep the caption visible while other windows cover the main window
        (guard timer)

        The caption stays at the bottom of the main window, unless another window
        (e.g. a dialog) covers it: it is then moved where it is the least covered,
        among the top of the main window, the places just above or just below the
        other windows, and (narrowed) the places on their left or on their right.
        If it is still covered by a modal dialog, the dialog is moved down, if the
        main window has room for both. Once uncovered, the caption gets back to the
        bottom of the main window.
        """
        caption = self.__caption
        if caption is None or not caption.isVisible():
            return
        win = self.win
        windows = self.__covering_windows()
        rects = []
        for window in windows:
            frame = window.frameGeometry()
            rects.append(QC.QRect(win.mapFromGlobal(frame.topLeft()), frame.size()))

        def covered_area(rect: QC.QRect) -> int:
            """Return the area of a caption geometry covered by the windows"""
            area = 0
            for other in rects:
                common = rect.intersected(other)
                if not common.isEmpty():
                    area += common.width() * common.height()
            return area

        best = CaptionBanner.DEFAULT_PLACEMENT
        best_area = covered_area(caption.geometry_at())
        if best_area > 0:
            for placement in self.__caption_placements(caption, rects):
                rect = caption.geometry_at(*placement)
                if rect.top() < caption.top_limit() or rect.bottom() >= win.height():
                    continue
                area = covered_area(rect)
                if area < best_area:
                    best, best_area = placement, area
            if best_area > 0:
                self.__make_room(caption.geometry_at(*best), windows)
        if best != caption.placement:
            caption.set_placement(*best)
            for cover, _targets in self.__covers:
                if is_qobject_valid(cover):
                    cover.update_geometry()

    def __caption_placements(
        self, caption: CaptionBanner, rects: list[QC.QRect]
    ) -> list[Placement]:
        """Return the placements of the caption avoiding windows, by preference

        Args:
            caption: Caption banner
            rects: Geometries of the windows, in main window coordinates

        Returns:
            Placements: top of the main window, above or below each window, then
             (narrowed) on the left or on the right of each window, at the bottom
             or at the top of the main window
        """
        margin = self.CAPTION_MARGIN
        height = caption.geometry_at().height()
        placements: list[Placement] = [(caption.top_limit(), None, None)]
        for rect in rects:
            placements.append((rect.top() - height - margin, None, None))
            placements.append((rect.bottom() + margin, None, None))
        min_width = max(320, self.win.width() // 4)
        for rect in rects:
            sides = [
                (margin, rect.left() - margin),
                (rect.right() + margin, self.win.width() - margin),
            ]
            # The side with the most room first
            sides.sort(key=lambda side: side[0] - side[1])
            for start, stop in sides:
                width = min(stop - start, caption.default_width())
                if width >= min_width:
                    left = start + (stop - start - width) // 2
                    placements.append((None, left, width))
                    placements.append((caption.top_limit(), left, width))
        return placements

    def __make_room(self, rect: QC.QRect, windows: list[QW.QWidget]) -> None:
        """Move down the modal dialogs covering the caption, if the main window has
        room for them below the caption

        Args:
            rect: Geometry of the caption, in main window coordinates
            windows: Windows which may cover the caption
        """
        win = self.win
        for window in windows:
            if not (isinstance(window, QW.QDialog) and window.isModal()):
                continue
            frame = window.frameGeometry()
            frame.moveTopLeft(win.mapFromGlobal(frame.topLeft()))
            if not frame.intersects(rect) or frame.top() < rect.top():
                continue
            top = rect.bottom() + self.CAPTION_MARGIN
            if top + frame.height() <= win.height():
                window.move(window.pos() + QC.QPoint(0, top - frame.top()))

    def __caption_header(self) -> str:
        """Return the header line of the captions"""
        step = _("Step %d") % self.step
        if self.scenario is None:
            return step
        return self.scenario.title + "  •  " + step

    def title_card(
        self, title: str, subtitle: str | None = None, hold: float = 3.0
    ) -> None:
        """Show a card covering the main window, then pause

        Args:
            title: Title
            subtitle: Subtitle
            hold: Pause duration (s) at normal pace
        """
        logger.info("Showcase title card: %s", title)
        self.__show_card(title, subtitle or "", "", hold)

    def end_card(self, hold: float = 5.0) -> None:
        """Show the end card, inviting the audience to try DataLab, then pause

        Args:
            hold: Pause duration (s) at normal pace
        """
        title, footer = _("Try DataLab"), _("DataLab Web: in your browser")
        self.__show_card(title, "datalab-platform.com", footer, hold)

    def __show_card(self, title: str, subtitle: str, footer: str, hold: float) -> None:
        """Show a card covering the main window, then pause

        The card stays shown until the processing worker is warmed up (see
        :meth:`start_warm_up`).
        """
        if self.check:
            self.pause(0.0)
            return
        self.hide_caption()
        if self.__card is None:
            self.__card = TitleCard(self.win)
        self.__card.set_content(title, subtitle, footer)
        self.__card.popup()
        try:
            self.pause(hold)
            self.finish_warm_up()
        finally:
            self.__card.hide()

    # ------ Spotlight and mouse pointer ------------------------------------------
    def __is_window_part(self, widget: QW.QWidget | None) -> bool:
        """Return True if a widget is a visible part of the main window"""
        return (
            isinstance(widget, QW.QWidget)
            and is_qobject_valid(widget)
            and widget.isVisible()
            and (widget is self.win or self.win.isAncestorOf(widget))
        )

    def __owned_window(self, widget: QW.QWidget | None) -> QW.QWidget | None:
        """Return the separate window of a visible widget, if the main window owns
        it (e.g. a floating panel or a dialog of the main window), else None"""
        if not (
            isinstance(widget, QW.QWidget)
            and is_qobject_valid(widget)
            and widget.isVisible()
        ):
            return None
        window = widget.window()
        if window is self.win:
            return None
        parent = window.parentWidget()
        while parent is not None:
            if parent is self.win:
                return window
            parent = parent.parentWidget()
        return None

    @contextmanager
    def spotlight(self, *widgets: QW.QWidget | None) -> Generator[None, None, None]:
        """Dim the main window, except some widgets, within the context

        Hidden widgets and None values are ignored. Widgets shown in separate
        windows owned by the main window (e.g. a floating panel or a dialog) stay
        bright: their windows are raised above the dimmed main window. Widgets of
        other windows are ignored. Captions shown within the context are not
        dimmed. In check mode, nothing is done.

        Args:
            widgets: Widgets to keep bright

        Yields:
            None
        """
        targets = [widget for widget in widgets if self.__is_window_part(widget)]
        windows: list[QW.QWidget] = []
        for widget in widgets:
            window = self.__owned_window(widget)
            if window is not None and window not in windows:
                windows.append(window)
        if self.check or not (targets or windows):
            yield
            return
        excluded = list(targets)
        if self.__caption is not None and self.__caption.isVisible():
            excluded.append(self.__caption)
        cover = Cover(self.win)
        cover.setAttribute(QC.Qt.WA_ShowWithoutActivating)
        cover.setWindowFlags(cover.windowFlags() | QC.Qt.WindowTransparentForInput)
        cover.exclude(excluded)
        cover.show()
        cover.update_geometry()
        for window in windows:
            window.raise_()
        self.__covers.append((cover, targets))
        QW.QApplication.processEvents()
        try:
            yield
        finally:
            self.__covers.remove((cover, targets))
            if is_qobject_valid(cover):
                cover.close()
                cover.deleteLater()
            QW.QApplication.processEvents()

    def move_cursor_to(
        self, widget: QW.QWidget, pos: QC.QPoint | None = None, duration: float = 0.6
    ) -> None:
        """Move the mouse pointer smoothly onto a widget

        Nothing is done in check mode, on a platform without screen or if the
        widget is hidden.

        Args:
            widget: Target widget
            pos: Target position, in widget coordinates (default: widget center)
            duration: Duration (s) of the move at normal pace
        """
        if (
            self.check
            or is_headless()
            or widget is None
            or not is_qobject_valid(widget)
            or not widget.isVisible()
        ):
            return
        target = widget.mapToGlobal(widget.rect().center() if pos is None else pos)
        start = QG.QCursor.pos()
        duration *= self.speed
        steps = max(1, int(duration * 60))
        for index in range(1, steps + 1):
            ratio = index / steps
            eased = ratio * ratio * (3.0 - 2.0 * ratio)
            QG.QCursor.setPos(
                round(start.x() + (target.x() - start.x()) * eased),
                round(start.y() + (target.y() - start.y()) * eased),
            )
            self.__wait(duration / steps)

    @staticmethod
    def __send_mouse_event(
        widget: QW.QWidget, event_type: QC.QEvent.Type, pos: QC.QPoint
    ) -> None:
        """Send a left mouse button event to a widget"""
        pressed = event_type == QC.QEvent.MouseButtonPress
        buttons = QC.Qt.LeftButton if pressed else QC.Qt.NoButton
        event = QG.QMouseEvent(
            event_type, QC.QPointF(pos), QC.Qt.LeftButton, buttons, QC.Qt.NoModifier
        )
        QW.QApplication.sendEvent(widget, event)

    def click(self, widget: QW.QWidget, pos: QC.QPoint | None = None) -> None:
        """Click on a widget, after moving the mouse pointer onto it

        Buttons are clicked with ``QAbstractButton.click`` (and shown pressed for
        a while beforehand, at showcase speed). Other widgets receive left mouse
        button press and release events.

        Args:
            widget: Widget to click
            pos: Click position, in widget coordinates (default: widget center)

        Raises:
            RuntimeError: If the widget is disabled
        """
        if not widget.isEnabled():
            raise RuntimeError(f"Cannot click on a disabled {type(widget).__name__}")
        self.move_cursor_to(widget, pos)
        if isinstance(widget, QW.QAbstractButton) and pos is None:
            if not self.check:
                widget.setDown(True)
                self.__wait(0.15 * self.speed)
                widget.setDown(False)
            widget.click()
        else:
            local = widget.rect().center() if pos is None else pos
            self.__send_mouse_event(widget, QC.QEvent.MouseButtonPress, local)
            if not self.check:
                self.__wait(0.15 * self.speed)
            self.__send_mouse_event(widget, QC.QEvent.MouseButtonRelease, local)
        QW.QApplication.processEvents()

    def type_text(self, widget: QW.QWidget, text: str, cps: float = 25.0) -> None:
        """Type text in an editor, character by character

        Supported editors: ``QLineEdit``, ``QPlainTextEdit`` (e.g. the macro
        editor of DataLab) and ``QTextEdit``. The text is inserted at the cursor
        position. In check mode, the text is inserted at once.

        Args:
            widget: Editor
            text: Text to type
            cps: Typing speed (characters per second) at normal pace

        Raises:
            TypeError: If the widget is not a supported editor
        """
        if isinstance(widget, QW.QLineEdit):
            insert = widget.insert
        elif isinstance(widget, (QW.QPlainTextEdit, QW.QTextEdit)):
            insert = widget.insertPlainText
        else:
            raise TypeError(f"Cannot type text in a {type(widget).__name__}")
        self.move_cursor_to(widget)
        widget.setFocus()
        if self.check or cps <= 0:
            insert(text)
        else:
            delay = self.speed / cps
            for char in text:
                insert(char)
                if not isinstance(widget, QW.QLineEdit):
                    widget.ensureCursorVisible()
                self.__wait(delay)
        QW.QApplication.processEvents()

    # ------ Menus, actions and processing features ----------------------------------
    def current_panel(self) -> BaseDataPanel:
        """Return the current data panel (Signal Panel or Image Panel)"""
        return self.win.tabwidget.currentWidget()

    def __get_panel(self, panel: BaseDataPanel | str | None) -> BaseDataPanel:
        """Return a data panel from its name (``'signal'`` or ``'image'``)"""
        if panel is None:
            return self.current_panel()
        if isinstance(panel, str):
            panels = {"signal": self.win.signalpanel, "image": self.win.imagepanel}
            if panel not in panels:
                raise ValueError(f"Unknown data panel: {panel}")
            return panels[panel]
        return panel

    def find_feature_action(
        self, name: str, panel: BaseDataPanel | str | None = None
    ) -> QW.QAction | None:
        """Return the action of a processing feature

        Args:
            name: Feature name, as for ``run_feature`` of the panel processor
             (e.g. ``'normalize'``)
            panel: Data panel, ``'signal'`` or ``'image'`` (default: current panel)

        Returns:
            Action of the feature (the action titled like the feature, among the
             actions of the panel), or None if the feature has no action
        """
        data_panel = self.__get_panel(panel)
        title = data_panel.processor.get_feature(name).action_title
        for action in data_panel.actions():
            if action.text() == title:
                return action
        return None

    def find_menu_path(self, action: QW.QAction) -> list[QW.QAction] | None:
        """Return the path to an action in the menus of the main window

        Menus populated when they are about to show (as most DataLab menus) are
        populated first.

        Args:
            action: Action to find

        Returns:
            Menu actions from the menu bar to the action (included), or None if
             the action is in no menu
        """
        tops = list(self.win.menuBar().actions())
        # The File menu is searched last: it creates new submenus when populated
        tops.sort(key=lambda item: item.menu() is self.win.file_menu)
        visited: dict[int, QW.QMenu] = {}
        for top in tops:
            if top is action:
                return [top]
            menu = top.menu()
            if menu is not None:
                path = self.__search_menu(menu, action, visited)
                if path is not None:
                    return [top] + path
        return None

    def __search_menu(
        self, menu: QW.QMenu, action: QW.QAction, visited: dict[int, QW.QMenu]
    ) -> list[QW.QAction] | None:
        """Return the path to an action in a menu and its submenus"""
        if id(menu) in visited:
            return None
        visited[id(menu)] = menu
        menu.aboutToShow.emit()  # Populate the menu, if it is populated on demand
        for item in menu.actions():
            if item is action:
                return [item]
            submenu = item.menu()
            if submenu is not None:
                path = self.__search_menu(submenu, action, visited)
                if path is not None:
                    return [item] + path
        return None

    @staticmethod
    def __match_item(menu: QW.QMenu, item: QW.QAction) -> QW.QAction | None:
        """Return the item of a menu matching an item (rebuilt menus included)"""
        actions = menu.actions()
        for candidate in actions:
            if candidate is item:
                return candidate
        if not is_qobject_valid(item):
            return None
        text, is_menu = item.text(), item.menu() is not None
        for candidate in actions:
            if candidate.text() == text and (candidate.menu() is not None) == is_menu:
                return candidate
        return None

    def __walk_menus(self, path: list[QW.QAction]) -> None:
        """Open the menus of a path one after the other, highlighting its items"""
        menubar = self.win.menuBar()
        opened: list[QW.QMenu] = []
        parent: QW.QMenu | None = None
        try:
            for index, item in enumerate(path):
                last = index == len(path) - 1
                if parent is None:
                    rect = menubar.actionGeometry(item)
                    self.move_cursor_to(menubar, rect.center())
                    menubar.setActiveAction(item)
                    menu = item.menu()
                    if menu is not None and not last:
                        self.pause(0.15)
                        if not menu.isVisible():
                            menu.popup(menubar.mapToGlobal(rect.bottomLeft()))
                        opened.append(menu)
                        parent = menu
                else:
                    item = self.__match_item(parent, item)
                    if item is None:
                        break
                    rect = parent.actionGeometry(item)
                    self.move_cursor_to(parent, rect.center(), duration=0.4)
                    parent.setActiveAction(item)
                    submenu = item.menu()
                    if submenu is not None and not last:
                        self.pause(0.15)
                        if not submenu.isVisible():
                            submenu.popup(parent.mapToGlobal(rect.topRight()))
                        opened.append(submenu)
                        parent = submenu
                self.pause(1.0 if last else 0.5)
        finally:
            for menu in reversed(opened):
                menu.hide()
            QW.QApplication.processEvents()

    def __point_tool_button(self, action: QW.QAction) -> None:
        """Point at the toolbar button of an action, if any"""
        for button in self.win.findChildren(QW.QToolButton):
            if button.defaultAction() is action and button.isVisible():
                self.move_cursor_to(button)
                button.setDown(True)
                self.pause(0.3)
                button.setDown(False)
                return

    def point_action(self, action: QW.QAction) -> None:
        """Show where an action is, without triggering it

        At showcase speed, the menus containing the action are opened one after
        the other, the action is highlighted, then the menus are closed (an action
        in no menu is shown by pointing at its toolbar button, if any). In check
        mode, nothing is done.

        Args:
            action: Action to show
        """
        if self.check:
            return
        path = self.find_menu_path(action)
        if path is not None:
            self.__walk_menus(path)
        else:
            self.__point_tool_button(action)
        QW.QApplication.processEvents()

    def trigger_action(self, action: QW.QAction) -> None:
        """Trigger an action, after walking through the menus leading to it

        At showcase speed, the menus containing the action are opened one after
        the other (including menus populated when they are about to show), the
        action is highlighted, then the menus are closed and the action is
        triggered. An action in no menu is triggered after pointing at its toolbar
        button (if any). In check mode, the action is triggered at once.

        Args:
            action: Action to trigger

        Raises:
            RuntimeError: If the action is disabled
        """
        path = self.find_menu_path(action)
        if not self.check:
            if path is not None:
                self.__walk_menus(path)
            else:
                self.__point_tool_button(action)
        QW.QApplication.processEvents()
        text = plain_text(action.text())
        if not action.isEnabled():
            raise RuntimeError(f"Action {text!r} is disabled")
        logger.info("Showcase action: %s", text)
        action.trigger()
        QW.QApplication.processEvents()

    def run_feature(
        self,
        name: str,
        param: Any = None,
        panel: BaseDataPanel | str | None = None,
        *,
        obj2: SignalObj | ImageObj | None = None,
        edit: bool | None = None,
    ) -> SignalObj | ImageObj | None:
        """Run a processing feature, showing the menu path of its action

        At showcase speed, the menus leading to the action of the feature are
        opened one after the other. Then, without parameters, the action is
        triggered: its parameter dialog (if any) is shown, then accepted by the
        demo mode of DataLab. With parameters (or a second operand), the feature is
        run with them, without any dialog. In check mode, the feature is run without
        any parameter dialog (with the default parameters, if not given).

        Args:
            name: Feature name, as for ``run_feature`` of the panel processor
             (e.g. ``'normalize'``, ``'moving_average'``)
            param: Feature parameters (a list of parameters for features creating
             several objects, e.g. ROI extraction)
            panel: Data panel, ``'signal'`` or ``'image'`` (default: current
             panel): it is made current first
            obj2: Second operand (features combining two objects)
            edit: True to show the parameter dialog even with parameters

        Returns:
            Current object of the panel after the processing (i.e. the new object
             for processing features)
        """
        data_panel = self.__get_panel(panel)
        if data_panel is not self.current_panel():
            self.win.set_current_panel(data_panel)
            self.pause(0.5)
        processor = data_panel.processor
        feature = processor.get_feature(name)
        action = self.find_feature_action(name, data_panel)
        no_args = param is None and obj2 is None and edit is None
        if action is not None and no_args and not self.check:
            self.trigger_action(action)
        else:
            if action is not None:
                path = self.find_menu_path(action)
                if path is not None and not self.check:
                    self.__walk_menus(path)
                QW.QApplication.processEvents()
                if not action.isEnabled():
                    text = plain_text(action.text())
                    raise RuntimeError(f"Action {text!r} is disabled")
            kwargs: dict[str, Any] = {}
            if param is not None:
                if feature.pattern == "1_to_n":
                    params = param if isinstance(param, (list, tuple)) else [param]
                    kwargs["params"] = list(params)
                else:
                    kwargs["param"] = param
                kwargs["edit"] = False if edit is None else edit
            elif edit is not None:
                kwargs["edit"] = edit
            elif self.check:
                # Check mode: no parameter dialog, the default parameters are used
                kwargs["edit"] = False
            if obj2 is not None:
                kwargs["obj2"] = obj2
            logger.info("Showcase feature: %s", name)
            processor.run_feature(feature, **kwargs)
        QW.QApplication.processEvents()
        return data_panel.objview.get_current_object()

    def checkpoint(self, condition: bool, message: str) -> None:
        """Check a condition, like an assertion of a test

        Args:
            condition: Condition to check
            message: Message of the error raised if the condition is not met

        Raises:
            AssertionError: If the condition is not met
        """
        if not condition:
            raise AssertionError(message)
        logger.debug("Showcase checkpoint passed: %s", message)

    # ------ Modal-dialog guard, workspace and recovery ----------------------------
    @contextmanager
    def unguarded(self, widget: QW.QWidget) -> Generator[None, None, None]:
        """Exempt a modal dialog from the modal-dialog guard within the context

        For a scenario driving a modal dialog itself (e.g. moving a slider, then
        clicking on *OK*), for longer than :attr:`modal_timeout`: the scenario is
        responsible for closing the dialog.

        Args:
            widget: Modal dialog driven by the scenario

        Yields:
            None
        """
        self.__unguarded.append(widget)
        try:
            yield
        finally:
            self.__unguarded = [w for w in self.__unguarded if w is not widget]

    def __keep_overlays_on_top(self) -> None:
        """Raise the overlays hidden by a dock raised meanwhile (guard timer)

        DataLab raises a plot dock when an object is added (e.g. by a macro or a
        remote client): the overlays, child widgets of the main window, would stay
        hidden behind it until the next caption.
        """
        children = self.win.children()
        for overlay in (self.__caption, self.__badge, self.__card):
            if overlay is None or not overlay.isVisible():
                continue
            above = children[children.index(overlay) + 1 :]
            if any(isinstance(child, QW.QDockWidget) for child in above):
                overlay.raise_()
                children = self.win.children()

    def __check_modal_widgets(self) -> None:
        """Accept the modal dialogs and message boxes left open (guard timer)"""
        now = time.monotonic()
        candidates: list[QW.QWidget] = []
        modal = QW.QApplication.activeModalWidget()
        if modal is not None:
            candidates.append(modal)
        for widget in QW.QApplication.topLevelWidgets():
            if (
                isinstance(widget, QW.QMessageBox)
                and widget.isVisible()
                and widget is not modal
            ):
                candidates.append(widget)
        guarded: dict[int, tuple[QW.QWidget, float, float]] = {}
        for widget in candidates:
            if isinstance(widget, QW.QProgressDialog) or any(
                widget is unguarded for unguarded in self.__unguarded
            ):
                continue
            entry = self.__guarded.get(id(widget))
            if entry is None or entry[0] is not widget:
                entry = (widget, now, self.guard_timeout(widget))
            since, timeout = entry[1], entry[2]
            if now - since >= timeout:
                self.__accept_widget(widget, timeout)
            else:
                guarded[id(widget)] = entry
        self.__guarded = guarded

    def __accept_widget(self, widget: QW.QWidget, timeout: float) -> None:
        """Accept a dialog or a message box left open"""
        logger.warning(
            "Showcase guard: accepting %s %r, open for more than %.0f s",
            type(widget).__name__,
            widget.windowTitle(),
            timeout,
        )
        if isinstance(widget, QW.QMessageBox):
            button = widget.defaultButton()
            for standard in (QW.QMessageBox.Ok, QW.QMessageBox.Yes):
                if button is None:
                    button = widget.button(standard)
            if button is None:
                for candidate in widget.buttons():
                    role = widget.buttonRole(candidate)
                    if role in (QW.QMessageBox.AcceptRole, QW.QMessageBox.YesRole):
                        button = candidate
                        break
            if button is not None:
                button.click()
            else:
                widget.close()
        elif isinstance(widget, QW.QDialog):
            widget.accept()
        else:
            widget.close()

    def close_stray_windows(self) -> None:
        """Close the menus, dialogs and windows left open

        The main window and its dock widgets are kept.
        """
        for _index in range(10):
            popup = QW.QApplication.activePopupWidget()
            if popup is None:
                break
            popup.close()
        for widget in QW.QApplication.topLevelWidgets():
            if (
                widget is self.win
                or isinstance(widget, (QW.QDockWidget, QW.QMenu))
                or not is_qobject_valid(widget)
                or not widget.isVisible()
            ):
                continue
            logger.info(
                "Showcase: closing %s %r", type(widget).__name__, widget.windowTitle()
            )
            widget.close()
        QW.QApplication.processEvents()

    def reset_workspace(self) -> None:
        """Reset the workspace, without any confirmation

        Remove all the objects (signals, images and macros) and the history, close
        the windows left open and the Applications panel (whose pages forget the
        outcomes of the previous runs), then show the Signal Panel.
        """
        win = self.win
        self.hide_caption()
        self.close_stray_windows()
        win.docks[win.applicationspanel].hide()
        win.applicationspanel.clear_outcomes()
        win.reset_all()
        win.historypanel.remove_all_objects()
        win.set_current_panel("signal")
        process_events()

    def recover(self) -> None:
        """Recover from a scenario failure

        Close the windows left open, tell the audience that the showcase moves on,
        then reset the workspace.
        """
        self.close_stray_windows()
        if not self.check:
            self.say(_("Moving on to the next demo..."), hold=2.0)
        self.reset_workspace()
