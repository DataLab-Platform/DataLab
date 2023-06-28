# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab main window status bar widgets
"""

from __future__ import annotations

import psutil
from guidata.configtools import get_icon
from guidata.qthelpers import get_std_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from cdl.config import DEBUG, Conf, _
from cdl.env import execenv
from cdl.plugins import PluginRegistry


class BaseStatus(QW.QWidget):
    """Base status widget.

    Args:
        delay (int | None): update interval (s). If None, widget will not be updated.
        parent (QWidget): parent widget
    """

    def __init__(
        self, delay: int | None = None, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        layout = QW.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.icon = QW.QLabel()
        self.label = QW.QLabel()
        layout.addWidget(self.icon)
        layout.addWidget(self.label)
        if delay is not None:
            self.timer = QC.QTimer()
            self.timer.timeout.connect(self.update_status)
            self.timer.start(delay * 1000)

    def set_icon(self, icon: QG.QIcon | str | None) -> None:
        """Set icon.

        Args:
            icon (QIcon | None): icon
        """
        size = self.label.sizeHint().height()
        if isinstance(icon, str):
            icon = get_icon(icon)
        pixmap = QG.QPixmap() if icon is None else icon.pixmap(size, size)
        self.icon.setPixmap(pixmap)

    def update_status(self) -> None:
        """Update status widget"""
        raise NotImplementedError


class MemoryStatus(BaseStatus):
    """Memory status widget.

    Args:
        threshold (int): available memory thresold (MB)
        delay (int): update interval (s)
        parent (QWidget): parent widget
    """

    SIG_MEMORY_ALARM = QC.Signal(bool)

    def __init__(
        self, threshold: int = 500, delay: int = 2, parent: QW.QWidget | None = None
    ) -> None:
        super().__init__(delay, parent)
        self.demo_mode = False
        self.ko_icon = get_std_icon("MessageBoxWarning")
        self.__threshold = threshold * (1024**2)
        self.label.setMinimumWidth(self.label.fontMetrics().width("000%"))
        self.update_status()

    def set_demo_mode(self, state: bool) -> None:
        """Set demo mode state (used when taking screenshots).
        The demo mode allows to take screenshots which always look the same.
        (this will set memory usage to a constant value).
        If demo mode is set to False, memory usage will be set to actual value.

        Args:
            state (bool): demo mode state
        """
        self.demo_mode = state
        self.update_status()

    def update_status(self) -> None:
        """Update status widget"""
        mem = psutil.virtual_memory()
        memok = mem.available > self.__threshold
        self.SIG_MEMORY_ALARM.emit(not memok)
        if DEBUG and not memok:
            execenv.log(self, f"Memory available: {mem.available//(1024**2)} MB")
        self.label.setStyleSheet("" if memok else "color: red")
        self.set_icon("libre-tech-ram.svg" if memok else self.ko_icon)
        mem_percent = 65 if self.demo_mode else int(mem.percent)
        self.label.setText(_("Memory:") + f" {mem_percent}%")


class PluginStatus(BaseStatus):
    """Plugin status widget.
    Shows the number of plugins loaded.

    Args:
        parent (QWidget): parent widget
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.set_icon("libre-gui-plugin.svg")
        self.update_status()

    def update_status(self) -> None:
        """Update status widget"""
        text = _("Plugins:") + " "
        if Conf.main.plugins_enabled.get():
            nplugins = len(PluginRegistry.get_plugins())
            text += str(nplugins)
        else:
            text += "-"
        self.label.setText(text)


class XMLRPCStatus(BaseStatus):
    """XML-RPC status widget.
    Shows the XML-RPC server status and port number.

    Args:
        parent (QWidget): parent widget
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.port: int | None = None

    def set_port(self, port: int | None):
        """Set XML-RPC server port number.

        Args:
            port (int | None): XML-RPC server port number.
                If None, XML-RPC server is disabled.
        """
        self.port = port
        self.update_status()

    def update_status(self) -> None:
        """Update status widget"""
        text = _("XML-RPC:") + " "
        if self.port is None:
            self.label.setText(text + "-")
            self.set_icon("libre-gui-unlink.svg")
        else:
            self.label.setText(text + str(self.port))
            self.set_icon("libre-gui-link.svg")
