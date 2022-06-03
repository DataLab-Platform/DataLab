# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Application embedded test 1

CodraFT main window is destroyed when closing application.
It is rebuilt from scratch when reopening application.
"""

from guidata.qthelpers import get_std_icon
from guidata.widgets.codeeditor import CodeEditor
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from codraft.config import _
from codraft.core.gui.main import CodraFTMainWindow
from codraft.tests import data as test_data
from codraft.utils.qthelpers import qt_app_context

SHOW = True  # Show test in GUI-based test launcher


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
        btn.clicked.connect(slot)
        self.add_widget(btn, spacing_before=spacing_before)

    def add_stretch(self):
        """Add stretch to button box"""
        self.button_layout.addStretch()


class BaseHostWindow(QW.QMainWindow):
    """Main window"""

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
        self.setWindowTitle(_("Host application"))
        self.setWindowIcon(get_std_icon("ComputerIcon"))
        self.codraft = None
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
        self.host.add_label(_("This the host application, which embeds CodraFT."))
        add_btn = self.host.add_button
        add_btn(_("Open CodraFT"), self.open_codraft, 10, "DialogApplyButton")
        add_btn(_("Import signal from CodraFT"), self.import_signal, 10, "ArrowLeft")
        add_btn(_("Import image from CodraFT"), self.import_image, 0, "ArrowLeft")
        add_btn(_("Add signal objects"), self.add_signals, 10, "CommandLink")
        add_btn(_("Add image objects"), self.add_images, 0, "CommandLink")
        add_btn(_("Remove all objects"), self.remove_all, 5, "MessageBoxWarning")
        add_btn(_("Close CodraFT"), self.close_codraft, 10, "DialogCloseButton")

    def open_codraft(self):
        """Open CodraFT test"""
        raise NotImplementedError

    def close_codraft(self):
        """Close CodraFT window"""
        if self.codraft is not None:
            self.host.log("=> Closed CodraFT")
            self.codraft.close()
            self.codraft = None

    def import_object(self, panel, title):
        """Import object from CodraFT"""
        self.host.log(f"get_object_dialog ({title}):")
        obj = panel.get_object_dialog(self.host, title)
        if obj is not None:
            self.host.log(f"  -> {obj.title}:")
            self.host.log(str(obj))
        else:
            self.host.log("  -> canceled")

    def import_signal(self):
        """Import signal from CodraFT"""
        if self.codraft is not None:
            self.import_object(self.codraft.signalpanel, self.sender().text())

    def import_image(self):
        """Import image from CodraFT"""
        if self.codraft is not None:
            self.import_object(self.codraft.imagepanel, self.sender().text())

    def add_signals(self):
        """Add signals to CodraFT"""
        if self.codraft is not None:
            for func in (test_data.create_test_signal1, test_data.create_test_signal2):
                obj = func(title=self.sigtitle)
                self.codraft.signalpanel.add_object(obj)
                self.host.log(f"Added signal: {obj.title}")

    def add_images(self):
        """Add images to CodraFT"""
        if self.codraft is not None:
            size = 2000
            for func in (
                test_data.create_test_image1,
                test_data.create_test_image2,
                test_data.create_test_image3,
            ):
                obj = func(size, title=self.imatitle)
                self.codraft.imagepanel.add_object(obj)
                self.host.log(f"Added image: {obj.title}")

    def remove_all(self):
        """Remove all objects from CodraFT"""
        if self.codraft is not None:
            for panel in self.codraft.panels:
                objn = len(panel.objlist)
                panel.remove_all_objects()
                self.host.log(f"Removed {objn} objects from {panel.PANEL_STR}")


class HostWindow(BaseHostWindow):
    """Test main view"""

    def open_codraft(self):
        """Open CodraFT test"""
        if self.codraft is None:
            self.codraft = CodraFTMainWindow(console=False)
            self.codraft.setAttribute(QC.Qt.WA_DeleteOnClose, True)
            self.codraft.show()
            self.host.log("✨Initialized CodraFT window")
        else:
            try:
                self.codraft.show()
                self.codraft.raise_()
                self.host.log("=> Shown CodraFT window")
            except RuntimeError:
                self.codraft = None
                self.open_codraft()


def test_embedded_feature(klass):
    """Testing embedded feature"""
    with qt_app_context(exec_loop=True, enable_logs=False):
        window = klass()
        window.resize(800, 800)
        window.show()


if __name__ == "__main__":
    test_embedded_feature(HostWindow)
