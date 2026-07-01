# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab main window status bar widgets
"""

from __future__ import annotations

import sigimax.widgets.status
from guidata.configtools import get_icon
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW

from datalab.config import PLUGIN_ERROR_COLOR, Conf, _
from datalab.plugins import PluginRegistry


class PluginStatus(sigimax.widgets.status.BaseStatus):
    """Plugin status widget.
    Shows the number of plugins loaded.

    Args:
        parent (QWidget): parent widget
    """

    #: Size used for creating the tinted icon pixmap
    ICON_TINT_SIZE: int = 64

    #: Error color for tinting the icon when plugins have failed
    ERROR_COLOR = QG.QColor(PLUGIN_ERROR_COLOR)

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.ok_icon = get_icon("libre-gui-plugin.svg")
        self.ko_icon = self._make_red_icon(self.ok_icon)
        self.update_status()

    @classmethod
    def _make_red_icon(cls, icon: QG.QIcon) -> QG.QIcon:
        """Create a red-tinted version of the given icon.

        Args:
            icon: Source icon

        Returns:
            Red-tinted icon
        """
        size = cls.ICON_TINT_SIZE
        pixmap = icon.pixmap(size, size)
        red_pixmap = QG.QPixmap(pixmap.size())
        red_pixmap.fill(QC.Qt.transparent)
        painter = QG.QPainter(red_pixmap)
        painter.drawPixmap(0, 0, pixmap)
        painter.setCompositionMode(QG.QPainter.CompositionMode_SourceIn)
        painter.fillRect(red_pixmap.rect(), cls.ERROR_COLOR)
        painter.end()
        return QG.QIcon(red_pixmap)

    def update_status(self) -> None:
        """Update status widget"""
        text = _("Plugins:") + " "
        if Conf.main.plugins_enabled.get():
            nplugins = len(PluginRegistry.get_plugins())
            nfailed = len(PluginRegistry.get_failed_plugins())
            ntotal = nplugins + nfailed
            text += f"{nplugins}/{ntotal}"
            has_errors = nfailed > 0
            self.setEnabled(True)
        else:
            text += "-"
            has_errors = False
            self.setEnabled(False)
        self.label.setText(text)
        self.set_icon(self.ko_icon if has_errors else self.ok_icon)
        if has_errors:
            self.label.setStyleSheet("color: red")
        else:
            self.label.setStyleSheet("")
        self.setToolTip(PluginRegistry.get_plugin_info())


class XMLRPCStatus(sigimax.widgets.status.BaseStatus):
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


class WebAPIStatus(sigimax.widgets.status.BaseStatus):
    """Web API status widget.
    Shows the Web API server status and port number.

    Args:
        parent (QWidget): parent widget
    """

    SIG_SHOW_INFO = QC.Signal()  # Signal to show connection info
    SIG_START_SERVER = QC.Signal()  # Signal to propose starting server

    def __init__(self, parent: QW.QWidget | None = None) -> None:
        super().__init__(None, parent)
        self.port: int | None = None
        self.url: str | None = None
        self.label.setCursor(QG.QCursor(QC.Qt.PointingHandCursor))
        self.label.mouseReleaseEvent = self.on_click
        self.update_status()  # Initialize widget state

    def on_click(self, event: QG.QMouseEvent) -> None:
        """Handle mouse click event on label.

        Args:
            event: mouse event
        """
        if event.button() == QC.Qt.LeftButton:
            if self.port is None:
                self.SIG_START_SERVER.emit()
            else:
                self.SIG_SHOW_INFO.emit()

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
        if self.port is None:
            self.label.setText(_("Web API"))
            self.set_icon("libre-gui-unlink.svg")
            self.setToolTip(
                _("Web API server is not running") + "\n" + _("Click to start")
            )
        else:
            self.label.setText(_("Web API:") + " " + str(self.port))
            self.set_icon("libre-gui-link.svg")
            tooltip = _("Web API server is running") + "\n"
            if self.url:
                tooltip += f"URL: {self.url}\n"
            tooltip += _("Click to view connection info")
            self.setToolTip(tooltip)
