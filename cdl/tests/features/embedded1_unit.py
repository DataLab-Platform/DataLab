# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Application embedded test 1

DataLab main window is destroyed when closing application.
It is rebuilt from scratch when reopening application.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import abc

from guidata.qthelpers import (
    get_std_icon,
    qt_app_context,
    win32_fix_title_bar_background,
)
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtWidgets as QW

import cdl.obj
from cdl.config import _
from cdl.core.gui.main import CDLMainWindow
from cdl.tests import data as test_data


class HostWidget(QW.QWidget):
    """Host widget: menu with action buttons, log viewer"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.button_layout = QW.QVBoxLayout()
        self.logwidget = CodeEditor(self)
        self.logwidget.setMinimumWidth(500)
        grid_layout = QW.QGridLayout()
        grid_layout.addLayout(self.button_layout, 0, 0)
        grid_layout.addWidget(self.logwidget, 0, 1)
        self.setLayout(grid_layout)

    def log(self, message):
        """Log message"""
        self.logwidget.appendPlainText(message)

    def add_spacing(self, spacing: int) -> None:
        """Add spacing to button box"""
        self.button_layout.addSpacing(spacing)

    def add_label(self, text: str) -> None:
        """Add label to button box"""
        self.button_layout.addWidget(QW.QLabel(text))

    def add_widget(self, obj: QW.QWidget, spacing_before: int = 0) -> None:
        """Add widget (QWidget) to button box"""
        if spacing_before > 0:
            self.add_spacing(spacing_before)
        self.button_layout.addWidget(obj)

    def add_button(self, title, slot, spacing_before=0, icon=None):
        """Add button"""
        btn = QW.QPushButton(title)
        if icon is not None:
            btn.setIcon(get_std_icon(icon))
        btn.clicked.connect(lambda _checked=False: slot())
        self.add_widget(btn, spacing_before=spacing_before)
        return btn

    def add_stretch(self):
        """Add stretch to button box"""
        self.button_layout.addStretch()


class AbstractClientWindowMeta(type(QW.QMainWindow), abc.ABCMeta):
    """Mixed metaclass to avoid conflicts"""


class AbstractClientWindow(QW.QMainWindow, metaclass=AbstractClientWindowMeta):
    """Abstract client window, to embed DataLab or connect to it"""

    PURPOSE = None
    INIT_BUTTON_LABEL = None
    SIG_TITLES = ("Oscilloscope", "Digitizer", "Radiometer", "Voltmeter", "Sensor")
    IMA_TITLES = (
        "Camera",
        "Streak Camera",
        "Image Scanner",
        "Laser Beam Profiler",
        "Gated Imaging Camera",
    )

    def __init__(self):
        super().__init__()
        win32_fix_title_bar_background(self)
        self.setWindowTitle(_("Host application"))
        self.setWindowIcon(get_std_icon("ComputerIcon"))
        self.cdl: CDLMainWindow = None
        self.host = HostWidget(self)
        self.setCentralWidget(self.host)
        self.setup_window()
        self.host.add_stretch()
        self.index_sigtitle = -1
        self.index_imatitle = -1

    @property
    def sigtitle(self):
        """Return current signal title index"""
        self.index_sigtitle = idx = (self.index_sigtitle + 1) % len(self.SIG_TITLES)
        return self.SIG_TITLES[idx]

    @property
    def imatitle(self):
        """Return current image title index"""
        self.index_imatitle = idx = (self.index_imatitle + 1) % len(self.IMA_TITLES)
        return self.IMA_TITLES[idx]

    def setup_window(self):
        """Setup window"""
        self.host.add_label(self.PURPOSE)
        add_btn = self.host.add_button
        add_btn(self.INIT_BUTTON_LABEL, self.init_cdl, 10, "DialogApplyButton")
        add_btn(_("Raise window"), self.raise_cdl, 0, "FileDialogToParent")
        self.add_additional_buttons()
        add_btn(_("Add signal objects"), self.add_signals, 10, "CommandLink")
        add_btn(_("Add image objects"), self.add_images, 0, "CommandLink")
        add_btn(_("Remove all objects"), self.remove_all, 5, "MessageBoxWarning")
        add_btn(_("Close DataLab"), self.close_cdl, 10, "DialogCloseButton")

    def add_additional_buttons(self):
        """Add additional buttons"""

    @abc.abstractmethod
    def init_cdl(self):
        """Open DataLab test"""

    def raise_cdl(self):
        """Raise DataLab window"""
        if self.cdl is not None:
            self.cdl.raise_window()
            self.host.log("=> Raised DataLab window")

    @abc.abstractmethod
    def close_cdl(self):
        """Close DataLab window"""

    def add_object(self, obj):
        """Add object to DataLab"""
        if self.cdl is not None:
            self.cdl.add_object(obj)

    def add_signals(self):
        """Add signals to DataLab"""
        if self.cdl is not None:
            for func in (
                test_data.create_paracetamol_signal,
                test_data.create_noisy_signal,
            ):
                obj = func(title=self.sigtitle)
                self.add_object(obj)
                self.host.log(f"Added signal: {obj.title}")

    def add_images(self):
        """Add images to DataLab"""
        if self.cdl is not None:
            p = cdl.obj.new_image_param(height=2000, width=2000, title=self.imatitle)
            for func in (
                test_data.create_sincos_image,
                test_data.create_noisygauss_image,
                test_data.create_multigauss_image,
            ):
                obj = func(p)
                self.add_object(obj)
                self.host.log(f"Added image: {obj.title}")

    @abc.abstractmethod
    def remove_all(self):
        """Remove all objects from DataLab"""


class AbstractHostWindow(AbstractClientWindow):  # pylint: disable=abstract-method
    """Abstract host window, embedding DataLab"""

    PURPOSE = _("This the host application, which embeds DataLab.")
    INIT_BUTTON_LABEL = _("Open DataLab")

    def remove_all(self):
        """Remove all objects from DataLab"""
        if self.cdl is not None:
            for panel in self.cdl.panels:
                panel.remove_all_objects()
                self.host.log(f"Removed objects from {panel.PANEL_STR}")

    def add_additional_buttons(self):
        """Add additional buttons"""
        add_btn = self.host.add_button
        add_btn(_("Import signal from DataLab"), self.import_signal, 10, "ArrowLeft")
        add_btn(_("Import image from DataLab"), self.import_image, 0, "ArrowLeft")

    def import_object(self, panel, title):
        """Import object from DataLab"""
        self.host.log(f"get_object_with_dialog ({title}):")
        obj = panel.get_object_with_dialog(title, parent=self.host)
        if obj is not None:
            self.host.log(f"  -> {obj.title}:")
            self.host.log(str(obj))
        else:
            self.host.log("  -> canceled")

    def import_signal(self):
        """Import signal from DataLab"""
        if self.cdl is not None:
            self.import_object(self.cdl.signalpanel, self.sender().text())

    def import_image(self):
        """Import image from DataLab"""
        if self.cdl is not None:
            self.import_object(self.cdl.imagepanel, self.sender().text())


class HostWindow(AbstractHostWindow):
    """Test main view"""

    def init_cdl(self):
        """Open DataLab test"""
        if self.cdl is None:
            self.cdl = CDLMainWindow(console=False)
            self.cdl.SIG_CLOSING.connect(self.cdl_was_closed)
            self.cdl.show()
            self.host.log("✨Initialized DataLab window")
        else:
            try:
                self.cdl.show()
                self.cdl.raise_()
                self.host.log("=> Shown DataLab window")
            except RuntimeError:
                self.cdl = None
                self.init_cdl()

    def cdl_was_closed(self):
        """DataLab was closed"""
        self.cdl = None
        self.host.log("✨DataLab window was closed by user")

    def close_cdl(self):
        """Close DataLab window"""
        if self.cdl is not None:
            self.host.log("=> Closed DataLab")
            self.cdl.SIG_CLOSING.disconnect(self.cdl_was_closed)
            self.cdl.close()
            self.cdl.deleteLater()
            self.cdl = None


def run_host_window(klass):
    """Running host window to test DataLab embedding feature"""
    with qt_app_context(exec_loop=True):
        window: HostWindow = klass()
        window.resize(800, 800)
        window.show()


def test_embedded_feature():
    """Testing embedded feature"""
    run_host_window(HostWindow)


if __name__ == "__main__":
    test_embedded_feature()
