# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Tour of DataLab features
------------------------

This module implements a tour of DataLab features.

A dialog box is displayed when the user launches DataLab for the first time or
when the user clicks on the "Show tour" entry in the "?" menu.

The tour user experience is the following:

- First, the DataLab main window is maximized, and it is grayed out (the whole window
  is covered with a gray area with an opacity of 50%). The user can not interact with
  the DataLab main window while the tour is running.

- A first modal dialog box is displayed with a short description of DataLab, and two
  buttons: "Start" and "Close". If the user clicks on "Start", the tour
  starts. If the user clicks on "Close", the tour is not started and the dialog box
  is closed.

- The tour is composed of several steps. Each step is a modal dialog box with a short
  description of a DataLab feature, and three buttons: "Previous", "Next" and "Close".
  If the user clicks on "Previous", the previous step is displayed. If the user clicks
  on "Next", the next step is displayed. If the user clicks on "Close", the tour is
  stopped and the dialog box is closed.

- At each step, the dialog box is moved to the center of the DataLab main window (i.e.
  the center of the screen), and the feature described in the step is highlighted in
  the DataLab main window. The highlight is a red rectangle around the feature, and
  the feature is the only part of the DataLab main window that is not grayed out.

- The last step of the tour is a modal dialog box with a conclusion message and two
  buttons: "Show tour again", "Close", and "Show demo". If the user clicks on "Show
  tour again", the tour starts again. If the user clicks on "Close", the tour is
  stopped and the dialog box is closed. If the user clicks on "Show demo", the demo
  is started.

"""

from __future__ import annotations

import abc
import enum
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from guidata.configtools import get_image_file_path
from guidata.qthelpers import is_dark_mode
from PyQt5.QtGui import QShowEvent
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

import cdl.obj
from cdl.config import _

if TYPE_CHECKING:
    from cdl.core.gui.main import CDLMainWindow


class Cover(QW.QWidget):
    """
    Widget that covers the main window with a gray area with an opacity of 50%.
    """

    def __init__(self, parent: CDLMainWindow) -> None:
        super().__init__(parent)
        self.setWindowFlags(QC.Qt.Tool | QC.Qt.FramelessWindowHint)
        self.setAttribute(QC.Qt.WA_TranslucentBackground)
        self.__opacity_factor = 1.0
        # Widgets to be excluded from the grayed out area:
        self.__excluded_widgets: list[QW.QWidget] | None = None
        # Path to be grayed out:
        self.__path: QG.QPainterPath = QG.QPainterPath()

    def set_opacity_factor(self, opacity_factor: float) -> None:
        """
        Set the opacity factor of the grayed out area.

        Args:
            opacity_factor: Opacity factor.
        """
        self.__opacity_factor = opacity_factor

    def exclude(self, widgets: list[QW.QWidget]) -> None:
        """
        Exclude widgets from the grayed out area.

        Args:
            widgets: Widgets to be excluded.
        """
        self.__excluded_widgets = widgets

    def update_geometry(self) -> None:
        """
        Update the geometry of the widget.
        """
        if os.name == "nt":
            QW.QApplication.processEvents()
        self.setGeometry(self.parent().geometry())
        self.__path = QG.QPainterPath()
        # Path is defined as the rectangle of the main window minus the rectangles of
        # the excluded widgets:
        self.__path.addRect(QC.QRectF(self.parent().rect()))
        if self.__excluded_widgets is not None:
            for widget in self.__excluded_widgets:
                widget.raise_()
                widget.show()
                geometry = widget.frameGeometry()
                width, height = geometry.width(), geometry.height()
                point = widget.mapTo(self.parent(), QC.QPoint(0, 0))
                x, y = point.x(), point.y()
                widget_path = QG.QPainterPath()
                widget_path.addRect(QC.QRectF(x, y, width, height))
                self.__path -= QG.QPainterPath(widget_path)
        self.repaint()

    def showEvent(self, a0: QShowEvent | None) -> None:
        """
        Event handler for the "show" event.

        Args:
            a0: Show event.
        """
        super().showEvent(a0)
        self.setGeometry(self.parent().geometry())

    def paintEvent(self, event: QG.QPaintEvent) -> None:
        """
        Event handler for the "paint" event.

        Args:
            event: Paint event.
        """
        super().paintEvent(event)
        painter = QG.QPainter(self)
        painter.setOpacity((0.75 if is_dark_mode() else 0.5) * self.__opacity_factor)
        painter.setBrush(QG.QBrush(QC.Qt.black))
        painter.setPen(QC.Qt.NoPen)
        painter.drawPath(self.__path)


class StepResult(enum.Enum):
    """
    Result of a step.
    """

    PREVIOUS = enum.auto()
    NEXT = enum.auto()
    CLOSE = enum.auto()
    DEMO = enum.auto()
    RESTART = enum.auto()


class StepDialog(QW.QDialog):
    """
    Base class for tour dialog boxes.

    Args:
        parent: Parent widget.
        step: Tour step.
    """

    def __init__(
        self,
        parent: QW.QWidget,
        step: TourStep,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(step.title)
        self.setWindowFlags(QC.Qt.Popup)
        self.setModal(True)
        self.step = step
        self.result: StepResult | None = None
        self._btnmap: dict[QW.QAbstractButton, StepResult] = {}
        self._bbox: QW.QDialogButtonBox | None = None
        self._setup_ui()

    def _btn(self, text: str, default: bool = False) -> QW.QPushButton:
        """
        Create a button.

        Args:
            text: Button text.

        Returns:
            Button.
        """
        btn = QW.QPushButton(text)
        btn.setAutoDefault(default)
        btn.setDefault(default)
        return btn

    def __set_default_button(self, btn: QW.QPushButton) -> None:
        """
        Set the default button in the button box.

        Args:
            btn: Button.
        """
        for button in self._bbox.buttons():
            button.setAutoDefault(button is btn)
            button.setDefault(button is btn)

    def __create_label(
        self,
        text: str,
        delta_fontsize: int,
        color: QC.Qt.GlobalColor,
        bold: bool = False,
    ) -> QW.QLabel:
        """
        Create a label.

        Args:
            text: Label text.
            delta_fontsize: Delta font size.
            color: Text color.
            bold: Bold text.

        Returns:
            Label.
        """
        label = QW.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QC.Qt.AlignCenter)
        # Change font size for something bigger:
        font = label.font()
        font.setPointSize(font.pointSize() + delta_fontsize)
        label.setFont(font)
        # Set text color:
        palette = label.palette()
        palette.setColor(QG.QPalette.WindowText, color)
        label.setPalette(palette)
        # Set bold text:
        font = label.font()
        font.setBold(bold)
        return label

    def __create_horizontal_line(self) -> QW.QFrame:
        """
        Create a horizontal line.

        Returns:
            Horizontal line.
        """
        line = QW.QFrame()
        line.setFrameShape(QW.QFrame.HLine)
        line.setFrameShadow(QW.QFrame.Sunken)
        line.setLineWidth(1)
        line.setMidLineWidth(0)
        palette = line.palette()
        palette.setColor(QG.QPalette.WindowText, QC.Qt.lightGray)
        line.setPalette(palette)
        return line

    def _setup_ui(self) -> None:
        """
        Setup the user interface.
        """
        self._bbox = bbox = QW.QDialogButtonBox()
        role = QW.QDialogButtonBox.AcceptRole
        if self.step.step_type == "introduction":
            next_btn = bbox.addButton(_("Start"), role)
            self._btnmap[next_btn] = StepResult.NEXT
            self.__set_default_button(next_btn)
        elif self.step.step_type == "conclusion":
            self._btnmap[bbox.addButton("<<", role)] = StepResult.RESTART
            self._btnmap[bbox.addButton("<", role)] = StepResult.PREVIOUS
            self._btnmap[bbox.addButton(_("Demo"), role)] = StepResult.DEMO
        else:
            self._btnmap[bbox.addButton("<<", role)] = StepResult.RESTART
            self._btnmap[bbox.addButton("<", role)] = StepResult.PREVIOUS
            next_btn = bbox.addButton(">", role)
            self._btnmap[next_btn] = StepResult.NEXT
            self.__set_default_button(next_btn)
        close_btn = bbox.addButton("X", role)
        if self.step.step_type == "introduction":
            close_btn.setText(_("Close"))
        self._btnmap[close_btn] = StepResult.CLOSE
        if self.step.step_type == "conclusion":
            self.__set_default_button(close_btn)
        bbox.clicked.connect(self.button_clicked)
        self._layout = QW.QVBoxLayout()
        if self.step.step_type in ("introduction", "conclusion"):
            logo = QW.QLabel()
            logo.setPixmap(QG.QPixmap(get_image_file_path("DataLab-Banner2-100.png")))
            logo.setAlignment(QC.Qt.AlignCenter)
            self._layout.addWidget(logo)
        title_color = QC.Qt.lightGray if is_dark_mode() else QC.Qt.darkGray
        title = self.__create_label(self.step.title, 1, title_color, True)
        self._layout.addWidget(title)
        self._layout.addWidget(self.__create_horizontal_line())
        label_color = QC.Qt.cyan if is_dark_mode() else QC.Qt.darkBlue
        label_dsize = 3 if self.step.step_type == "regular" else 4
        label = self.__create_label(self.step.text, label_dsize, label_color)
        if self.step.step_type == "regular":
            label.setAlignment(QC.Qt.AlignLeft)
        self._layout.addWidget(label)
        self._layout.addSpacing(5)
        self._layout.addWidget(bbox)
        if self.step.step_type == "introduction":
            self._layout.addSpacing(5)
            help_text = _(
                "Hit <b>Enter</b> to continue to the next step, or "
                "<b>Esc</b> to close the tour."
            )
            help_label = self.__create_label(help_text, -1, title_color)
            help_label.setAlignment(QC.Qt.AlignLeft)
            self._layout.addWidget(help_label)
        self.setLayout(self._layout)

    def button_clicked(self, button: QW.QAbstractButton) -> None:
        """
        Event handler for the "clicked" event on the buttons.

        Args:
            button: Button that was clicked.
        """
        self.result = self._btnmap[button]
        self.accept()

    def reject(self) -> None:
        """
        Event handler for the "reject" event.
        """
        self.result = StepResult.CLOSE
        super().reject()

    def paintEvent(self, event: QG.QPaintEvent) -> None:
        """
        Event handler for the "paint" event.

        Args:
            event: Paint event.
        """
        super().paintEvent(event)
        painter = QG.QPainter(self)
        painter.setOpacity(0.5)
        painter.setBrush(QG.QBrush(QC.Qt.black))
        painter.setPen(QC.Qt.NoPen)


class TourStep:
    """
    Tour step.

    Args:
        tour: Tour.
        title: Step title.
        text: Step text.
        widgets: Widgets to be highlighted. If None, no widget is highlighted, meaning
         that the step is probably an introduction or a conclusion.
        setup_callback: Callback function to be called before the step is displayed,
          which takes a single argument, the `CDLMainWindow` instance.
        teardown_callback: Callback function to be called after the step is displayed,
          which takes a single argument, the `CDLMainWindow` instance.
        step_type: Step type. Can be "regular", "introduction" or "conclusion".
    """

    def __init__(
        self,
        tour: BaseTour,
        title: str,
        text: str,
        widgets: list[QW.QWidget] | None = None,
        setup_callback: Callable | None = None,
        teardown_callback: Callable | None = None,
        step_type: str = "regular",
    ) -> None:
        self.tour = tour
        self.title = title
        self.text = text
        self.widgets = widgets
        self.setup_callback = setup_callback
        self.teardown_callback = teardown_callback
        self.step_type = step_type
        assert step_type in ["regular", "introduction", "conclusion"]

    def show(self) -> StepResult:
        """
        Show the step.

        Returns:
            Result of the step.
        """
        if self.setup_callback is not None:
            self.setup_callback(self.tour.win)
        self.update_cover(self.tour.cover)
        dialog = StepDialog(self.tour.win, self)
        dialog.exec()
        if self.teardown_callback is not None:
            self.teardown_callback(self.tour.win)
        return dialog.result

    def update_cover(self, cover: Cover) -> None:
        """
        Update the cover widget.

        Args:
            cover: Cover widget.
        """
        cover.set_opacity_factor(1.0 if self.step_type == "regular" else 0.7)
        cover.exclude(self.widgets)
        cover.update_geometry()


class BaseTourMeta(type(QW.QWidget), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class BaseTour(QW.QWidget, metaclass=BaseTourMeta):
    """
    Base class for the tour of DataLab features.

    Args:
        win: DataLab main window.
    """

    def __init__(self, win: CDLMainWindow) -> None:
        super().__init__(win)
        self.win = win
        self._steps: list[TourStep] = []
        self._current_step = 0
        self.cover = Cover(win)
        self.cover.show()
        self.__window_geometry: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.__window_prepared = False
        self.setup_tour(win)

    def __resize_window(self, factor: float) -> None:
        """
        Resize the window so that it is centered on the screen and its size is
        `factor` times the size of the screen.

        Args:
            factor: Factor by which the size of the window is multiplied.
        """
        desktop = QW.QApplication.desktop()
        screen = desktop.screenNumber(desktop.cursor().pos())
        screen_geometry = desktop.screenGeometry(screen)
        width = int(screen_geometry.width() * factor)
        height = int(screen_geometry.height() * factor)
        self.win.resize(width, height)
        self.win.move(screen_geometry.center() - QC.QPoint(width // 2, height // 2))

    def __save_window_geometry(self) -> None:
        """Save the window geometry (size and position)."""
        self.__window_geometry = self.win.saveGeometry()

    def __restore_window_geometry(self) -> None:
        """Restore the window geometry (size and position)."""
        if self.__window_geometry is not None:
            self.win.restoreGeometry(self.__window_geometry)

    @abc.abstractmethod
    def setup_tour(self, win: CDLMainWindow) -> None:
        """
        Setup the tour: add steps to the tour.
        """

    @abc.abstractmethod
    def cleanup_tour(self, win: CDLMainWindow) -> None:
        """
        Cleanup the tour.
        """

    def add_step(
        self,
        title: str,
        text: str,
        widgets: list[QW.QWidget] | None = None,
        setup_callback: Callable | None = None,
        teardown_callback: Callable | None = None,
        step_type: str | None = None,
    ) -> None:
        """
        Add a step to the tour.

        Args:
            title: Step title.
            text: Step text.
            widgets: Widgets to be highlighted. If None, no widget is highlighted,
             meaning that the step is probably an introduction or a conclusion.
            setup_callback: Callback function to be called before the step is displayed,
             which takes a single argument, the `CDLMainWindow` instance.
            teardown_callback: Callback function to be called after the step is displayed,
             which takes a single argument, the `CDLMainWindow` instance.
            step_type: Step type. Can be "regular", "introduction" or "conclusion".
             Defaults to None.
        """
        if step_type is None:
            if self._steps:
                step_type = "regular"
            else:
                step_type = "introduction"
        step = TourStep(
            self, title, text, widgets, setup_callback, teardown_callback, step_type
        )
        self._steps.append(step)

    def show_current_step(self) -> None:
        """
        Show the current step.
        """
        step = self._steps[self._current_step]
        result = step.show()
        if result is StepResult.PREVIOUS:
            self.previous_step()
        elif result is StepResult.NEXT:
            self.next_step()
        elif result is StepResult.CLOSE:
            self.end()
        elif result is StepResult.DEMO:
            self.end()
            from cdl.tests.scenarios import (
                demo,  # pylint: disable=import-outside-toplevel
            )

            demo.play_demo(self.win)
        elif result is StepResult.RESTART:
            self.start()
        else:
            raise RuntimeError(f"Unknown result: {result}")

    def start(self) -> None:
        """
        Start the tour.
        """
        self._current_step = 0
        self.show_current_step()

    def end(self) -> None:
        """
        End the tour.
        """
        self.cover.close()
        self.cleanup_tour(self.win)
        self.__restore_window_geometry()

    def next_step(self) -> None:
        """
        Go to the next step.
        """
        if self._current_step == 0 and not self.__window_prepared:
            self.__save_window_geometry()
            self.__resize_window(0.7)
            self.cover.update_geometry()
            self.__window_prepared = True
        self._current_step += 1
        self.show_current_step()

    def previous_step(self) -> None:
        """
        Go to the previous step.
        """
        self._current_step -= 1
        self.show_current_step()


class Tour(BaseTour):
    """
    Tour of DataLab features.

    Args:
        win: DataLab main window.
    """

    def prepare_signalpanel(self, win: CDLMainWindow) -> None:
        """
        Prepare the signal panel.

        Args:
            win: DataLab main window.
        """
        # Create a signal:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        sig = cdl.obj.create_signal(_("Test signal, for the tour"), x, y)
        win.add_object(sig)
        win.set_current_panel("signal")

    def prepare_imagepanel(self, win: CDLMainWindow) -> None:
        """
        Prepare the image panel.

        Args:
            win: DataLab main window.
        """
        # Create an image:
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx) * np.cos(yy)
        img = cdl.obj.create_image(_("Test image, for the tour"), zz)
        win.add_object(img)
        win.set_current_panel("image")

    def popup_menu(self, win: CDLMainWindow, menu: QW.QMenu) -> None:
        """
        Popup a menu.

        Args:
            win: DataLab main window.
            menu_name: Name of the menu to popup.
        """
        menu.popup(win.mapToGlobal(QC.QPoint(50, 50)))

    def cleanup_tour(self, win: CDLMainWindow) -> None:
        """
        Cleanup the tour.

        Args:
            win: DataLab main window.
        """
        if len(win.signalpanel.objmodel) == 1:
            win.signalpanel.remove_all_objects()
        else:
            win.signalpanel.remove_object()
        if len(win.imagepanel.objmodel) == 1:
            win.imagepanel.remove_all_objects()
        else:
            win.imagepanel.remove_object()

    def setup_tour(self, win: CDLMainWindow) -> None:
        """
        Setup the tour: add steps to the tour.
        """
        self.add_step(
            _("Welcome to DataLab!"),
            _("This tour will guide you through the main features of DataLab."),
        )
        self.add_step(
            _("DataLab main window"),
            _(
                "This is the main window of DataLab. It is composed of several parts "
                "that we will describe in the following steps."
            ),
        )
        self.add_step(
            _("DataLab main window"),
            _(
                "Menus and toolbars regroup the main actions that can be performed "
                "in DataLab. Their content is adapted to the current panel, as we "
                "will see in the following steps."
            ),
            [win.menuBar(), win.main_toolbar, win.signal_toolbar, win.image_toolbar],
        )
        self.add_step(
            _("DataLab main window"),
            _(
                "The main window is composed of two main panels: the Signal Panel, "
                "and the Image Panel.<br>"
                "Switching between panels is done using the highlighted tabs."
            ),
            [win.tabwidget.tabBar()],
        )
        self.add_step(
            _("Signal Panel"),
            _(
                "The <b>Signal Panel</b> is used to manage 1D signals."
                "It is composed of the elements shown in the following steps."
            ),
        )
        self.add_step(
            _("Signal Panel") + " – " + _("List and properties"),
            _(
                "In the highlighted area, signals are listed at the top, and their "
                "properties may be displayed and edited at the bottom.<br><br>"
                "Signals are numbered (but may be reorganized) and put together in "
                "numbered groups."
            ),
            [win.tabwidget.tabBar(), win.signalpanel],
            self.prepare_signalpanel,
        )
        self.add_step(
            _("Signal Panel") + " – " + _("View"),
            _(
                "Signals are plotted in the <b>Signal View</b>.<br><br>"
                "Curves may be customized using context menus or the vertical "
                "toolbar on the left (appearance settings are saved in the "
                "signal metadata)."
            ),
            [win.docks[win.signalpanel]],
        )
        self.add_step(
            _("Signal Panel") + " – " + _("File menu"),
            _(
                "The <b>File</b> menu contains actions to import and export signals "
                "individually (various formats) or to save or restore the whole "
                "workspace (HDF5 files)."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.file_menu),
            lambda win: win.file_menu.hide(),
        )
        self.add_step(
            _("Signal Panel") + " – " + _("Edit menu"),
            _(
                "The <b>Edit</b> menu contains actions to edit signals individually "
                "or in groups."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.edit_menu),
            lambda win: win.edit_menu.hide(),
        )
        self.add_step(
            _("Signal Panel") + " – " + _("Operations menu"),
            _(
                "The <b>Operations</b> menu is focused on arithmetic operations, "
                "data type conversions, peak detection, ROI extraction, ..."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.operation_menu),
            lambda win: win.operation_menu.hide(),
        )
        self.add_step(
            _("Signal Panel") + " – " + _("Processing menu"),
            _("The <b>Processing</b> menu regroups 1->1 signal processing actions."),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.processing_menu),
            lambda win: win.processing_menu.hide(),
        )
        self.add_step(
            _("Signal Panel") + " – " + _("Computing menu"),
            _(
                "The <b>Computing</b> menu regroups 1->0 signal computing actions "
                "(that is, actions that do not modify the signals, but compute "
                "a result, e.g. scalar values), with optional ROI selection."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.computing_menu),
            lambda win: win.computing_menu.hide(),
        )
        self.add_step(
            _("Image Panel"),
            _(
                "The <b>Image Panel</b> is used to manage 2D images. It is composed "
                "of the elements shown in the following steps."
            ),
            [],
            lambda win: win.set_current_panel("image"),
        )
        self.add_step(
            _("Image Panel") + " – " + _("List and properties"),
            _(
                "In the highlighted area, images are listed at the top, and their "
                "properties may be displayed and edited at the bottom.<br><br>"
                "Images are numbered (but may be reorganized) and put together in "
                "numbered groups."
            ),
            [win.tabwidget.tabBar(), win.imagepanel],
            self.prepare_imagepanel,
        )
        self.add_step(
            _("Image Panel") + " – " + _("View"),
            _(
                "Images are shown in the <b>Image View</b>.<br><br>"
                "The displayed images may be customized using context menus "
                "or the vertical toolbar on the left (appearance settings "
                "are saved in the image metadata)."
            ),
            [win.docks[win.imagepanel]],
        )
        self.add_step(
            _("Image Panel") + " – " + _("File menu"),
            _(
                "The <b>File</b> menu contains actions to import and export images "
                "individually (various formats) or to save or restore the whole "
                "workspace (HDF5 files)."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.file_menu),
            lambda win: win.file_menu.hide(),
        )
        self.add_step(
            _("Image Panel") + " – " + _("Edit menu"),
            _(
                "The <b>Edit</b> menu contains actions to edit images individually "
                "or in groups."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.edit_menu),
            lambda win: win.edit_menu.hide(),
        )
        self.add_step(
            _("Image Panel") + " – " + _("Operations menu"),
            _(
                "The <b>Operations</b> menu is focused on arithmetic operations, "
                "data type conversions, pixel binning, resize, ROI extraction ..."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.operation_menu),
            lambda win: win.operation_menu.hide(),
        )
        self.add_step(
            _("Image Panel") + " – " + _("Processing menu"),
            _(
                "The <b>Processing</b> menu regroups 1->1 image processing actions "
                "(that is, actions that modify the images)."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.processing_menu),
            lambda win: win.processing_menu.hide(),
        )
        self.add_step(
            _("Image Panel") + " – " + _("Computing menu"),
            _(
                "The <b>Computing</b> menu regroups 1->0 image computing actions "
                "(that is, actions that do not modify the images, but compute "
                "a result, e.g. circle coordinates), with optional ROI selection."
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.computing_menu),
            lambda win: win.computing_menu.hide(),
        )
        self.add_step(
            _("Extensions"),
            _(
                "DataLab is designed to be easily extended with new features, "
                "by using <b>Macros</b>, <b>Plugins</b> or <b>Remote Control</b>."
            ),
        )
        self.add_step(
            _("Extensions") + " – " + _("Macros"),
            _(
                "The <b>Macro manager</b> allows to create, edit and run macros.<br>"
                "Macros are saved together with the DataLab workspace (HDF5 file)."
            ),
            [win.docks[win.macropanel]],
            lambda win: win.set_current_panel("macro"),
            lambda win: win.set_current_panel("signal"),
        )
        self.add_step(
            _("Extensions") + " – " + _("Plugins"),
            _(
                "The <b>Plugins</b> menu regroups features that are not part of "
                "the core of DataLab, but that are provided as plugins.<br>"
                "(See the documentation for more information about plugins.)"
            ),
            [win.menuBar()],
            lambda win: self.popup_menu(win, win.plugins_menu),
            lambda win: win.plugins_menu.hide(),
        )
        self.add_step(
            _("This is the end of the tour!"),
            _("You can show the tour again, or close this dialog box."),
            step_type="conclusion",
        )


def start(win: CDLMainWindow) -> None:
    """
    Start the tour of DataLab features.

    Args:
        win: DataLab main window.
    """
    tour = Tour(win)
    tour.start()


def test_dialogs() -> None:
    """
    Test the tour dialog boxes.
    """
    # pylint: disable=wrong-import-position
    from guidata.qthelpers import qt_app_context

    from cdl.core.gui.main import CDLMainWindow

    with qt_app_context():
        win = CDLMainWindow()
        win.show()
        base_dlg = StepDialog(win, "Title", "Text")
        base_dlg._cover.exclude_widget(win.docks[win.signalpanel])
        base_dlg._cover.update_geometry()
        # QC.QTimer.singleShot(0, base_dlg._cover.update_geometry)
        base_dlg.exec()
        win.close()


def test_tour() -> None:
    """
    Test the tour of DataLab features.
    """
    # pylint: disable=wrong-import-position
    from cdl.tests import cdltest_app_context

    with cdltest_app_context() as win:
        start(win)


if __name__ == "__main__":
    # test_dialogs()
    test_tour()
