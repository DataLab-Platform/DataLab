# -*- coding: utf-8 -*-
#
# Copyright © 2023 Codra
# Pierre Raybaut

"""
Module providing a connection dialog box for the DataLab proxy client
"""

from __future__ import annotations

import time
from collections.abc import Callable

from guidata.configtools import get_icon, get_image_file_path
from guidata.qthelpers import (
    get_std_icon,
    qt_app_context,
    win32_fix_title_bar_background,
)
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import _


class ConnectionThread(QC.QThread):
    """DataLab Connection thread"""

    SIG_CONNECTION_OK = QC.Signal()
    SIG_CONNECTION_KO = QC.Signal()

    def __init__(self, connect_callback: Callable, parent: QC.QObject = None) -> None:
        super().__init__(parent)
        self.connect_callback = connect_callback

    def run(self) -> None:
        """Run thread"""
        try:
            self.connect_callback()
            self.SIG_CONNECTION_OK.emit()
        except ConnectionRefusedError:
            self.SIG_CONNECTION_KO.emit()


class ConnectionDialog(QW.QDialog):
    """DataLab Connection dialog

    Args:
        connect_callback: Callback function to connect to DataLab server
        parent: Parent widget. Defaults to None.
    """

    def __init__(self, connect_callback: Callable, parent: QW.QWidget = None) -> None:
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.setWindowTitle(_("Connection to DataLab"))
        self.setWindowIcon(get_icon("DataLab.svg"))
        self.resize(300, 50)
        self.host_label = QW.QLabel()
        pixmap = QG.QPixmap(get_image_file_path("DataLab-Banner-200.png"))
        self.host_label.setPixmap(pixmap)
        self.host_label.setAlignment(QC.Qt.AlignCenter)
        self.progress_bar = QW.QProgressBar()
        self.progress_bar.setRange(0, 0)
        status = QW.QWidget()
        status_layout = QW.QHBoxLayout()
        status.setLayout(status_layout)
        self.status_label = QW.QLabel("Waiting for connection...")
        self.status_icon = QW.QLabel()
        status_layout.addWidget(self.status_icon)
        status_layout.addWidget(self.status_label)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.addStretch()
        layout = QW.QVBoxLayout()
        layout.addWidget(self.host_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(status)
        self.setLayout(layout)
        self.thread = ConnectionThread(connect_callback)
        self.thread.SIG_CONNECTION_OK.connect(self.__on_connection_successful)
        self.thread.SIG_CONNECTION_KO.connect(self.__on_connection_failed)
        button_box = QW.QDialogButtonBox(QW.QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def __set_status_icon(self, name: str) -> None:
        """Set status icon with standard Qt icon name"""
        self.status_icon.setPixmap(QG.QPixmap(get_std_icon(name).pixmap(24)))

    def __connect_to_server(self) -> None:
        """Connect to server"""
        self.progress_bar.setRange(0, 0)
        self.__set_status_icon("BrowserReload")
        self.status_label.setText("Connecting to server...")
        self.thread.start()

    def __on_connection_successful(self) -> None:
        """Connection successful"""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.__set_status_icon("DialogApplyButton")
        self.status_label.setText("Connection successful!")
        QC.QTimer.singleShot(1000, self.accept)

    def __on_connection_failed(self) -> None:
        """Connection failed"""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.__set_status_icon("MessageBoxCritical")
        self.status_label.setText("Connection failed.")
        QC.QTimer.singleShot(2000, self.reject)

    def exec(self) -> int:
        """Execute dialog"""
        self.__connect_to_server()
        return super().exec()


if __name__ == "__main__":

    def fake_connect():
        """Fake connection"""
        time.sleep(5)
        # raise ConnectionRefusedError("Connection refused")

    with qt_app_context():
        dlg = ConnectionDialog(fake_connect)
        dlg.exec()
