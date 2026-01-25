# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab main window status bar widgets
"""

from __future__ import annotations

import os

import psutil
from guidata.configtools import get_icon
from guidata.qthelpers import get_std_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import DEBUG, Conf, _
from datalab.env import execenv
from datalab.plugins import PluginRegistry


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


class ConsoleStatus(BaseStatus):
    """Console status widget.

    Shows a message if an error or warning has been logged to the console.
    Shows a button to show the console, only if the console is hidden.

    Args:
        parent (QWidget): parent widget
    """

    SIG_SHOW_CONSOLE = QC.Signal()

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.label.setText(_("Internal console"))
        self.label.setToolTip(
            _(
                "Click to show the internal console.\n"
                "The icon will turn red if an error or warning is logged."
            )
        )
        self.label.setCursor(QG.QCursor(QC.Qt.PointingHandCursor))
        self.label.mouseReleaseEvent = self.on_click
        self.ok_icon = get_std_icon("MessageBoxInformation")
        self.ko_icon = get_std_icon("MessageBoxWarning")
        self.has_errors = False
        self.update_status()

    def on_click(self, event: QG.QMouseEvent) -> None:
        """Handle mouse click event on label.

        Args:
            event: mouse event
        """
        if event.button() == QC.Qt.LeftButton:
            self.SIG_SHOW_CONSOLE.emit()

    def console_visibility_changed(self, visible: bool) -> None:
        """Handle console visibility changed event.

        Args:
            visible (bool): console visibility
        """
        if visible:
            # Hide this status widget when console is visible
            self.hide()
        else:
            self.show()
            self.update_status()

    def exception_occurred(self) -> None:
        """Handle exception occurred event"""
        self.has_errors = True
        self.update_status()

    def update_status(self) -> None:
        """Update status widget"""
        if self.has_errors:
            self.set_icon(self.ko_icon)
            self.label.setStyleSheet("color: red")
            self.label.setToolTip(
                _(
                    "Click to show the internal console.\n"
                    "An error or warning has been logged."
                )
            )
        else:
            self.set_icon(self.ok_icon)
            self.label.setStyleSheet("")
            self.label.setToolTip(
                _(
                    "Click to show the internal console.\n"
                    "No error or warning has been logged."
                )
            )


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
        txtlist = [
            f"%s {mem.available // (1024**2)} MB" % _("Memory available:"),
            f"%s {mem.used // (1024**2)} MB" % _("Memory used:"),
            f"%s {self.__threshold // (1024**2)} MB" % _("Alarm threshold:"),
        ]
        txt = os.linesep.join(txtlist)
        self.setToolTip(txt)
        if DEBUG and not memok:
            execenv.log(self, txt)
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
        self.setToolTip(PluginRegistry.get_plugin_info())


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


class WebAPIStatus(BaseStatus):
    """Web API status widget.
    Shows the Web API server status and port number.

    Args:
        parent (QWidget): parent widget
    """

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.port: int | None = None
        self.url: str | None = None

    def set_status(self, url: str | None, port: int | None) -> None:
        """Set Web API server status.

        Args:
            url: Web API server URL (e.g. "http://127.0.0.1:8080").
                If None, Web API server is not running.
            port: Web API server port number.
        """
        self.url = url
        self.port = port
        self.update_status()

    def update_status(self) -> None:
        """Update status widget"""
        text = _("Web API:") + " "
        if self.port is None:
            self.label.setText(text + "-")
            self.set_icon("libre-gui-unlink.svg")
            self.setToolTip(_("Web API server is not running"))
        else:
            self.label.setText(text + str(self.port))
            self.set_icon("libre-gui-link.svg")
            tooltip = _("Web API server is running") + "\n"
            if self.url:
                tooltip += f"URL: {self.url}"
            self.setToolTip(tooltip)
