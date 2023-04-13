# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
CobraDataLab main window status bar widgets
"""


import psutil
from guidata.qthelpers import get_std_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import DEBUG, _
from cdl.env import execenv


class MemoryStatus(QW.QWidget):
    """
    Memory status widget

    :param int threshold: available memory thresold (MB)
    :param int delay: update interval (s)
    :param QWidget parent: parent widget
    """

    SIG_MEMORY_ALARM = QC.Signal(bool)

    def __init__(self, threshold: int = 500, delay: int = 2, parent: QW.QWidget = None):
        super().__init__(parent)
        self.demo_mode = False
        self.ko_icon = get_std_icon("MessageBoxWarning")
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.icon = QW.QLabel()
        self.label = QW.QLabel()
        layout.addWidget(self.icon)
        layout.addWidget(self.label)
        self.__threshold = threshold * (1024**2)
        self.label.setMinimumWidth(self.label.fontMetrics().width("000%"))
        self.timer = QC.QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(delay * 1000)
        self.update_status()

    def set_demo_mode(self, state):
        """Set demo mode state (used when taking screenshots)"""
        self.demo_mode = state
        self.update_status()

    def update_status(self):
        """Update status widget"""
        mem = psutil.virtual_memory()
        memok = mem.available > self.__threshold
        self.SIG_MEMORY_ALARM.emit(not memok)
        if DEBUG and not memok:
            execenv.print(f"=== Memory available: {mem.available//(1024**2)} MB ===")
        self.label.setStyleSheet("" if memok else "color: red")
        size = self.label.sizeHint().height()
        self.icon.setPixmap(QG.QPixmap() if memok else self.ko_icon.pixmap(size, size))
        mem_percent = 65 if self.demo_mode else int(mem.percent)
        self.label.setText(_("Memory:") + f" {mem_percent}%")
