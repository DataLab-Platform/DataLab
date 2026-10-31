# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plugin status bar widget tests."""

from __future__ import annotations

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtTest as QT
from qtpy import QtWidgets as QW

from datalab.config import PLUGIN_ERROR_COLOR, Conf, _
from datalab.plugins import FailedPluginInfo, PluginRegistry
from datalab.widgets.status import PluginStatus


class HighDpiIcon:
    """Icon returning its pixmaps at a device pixel ratio of 2, as `QIcon` does
    with a 200% screen scaling"""

    RATIO = 2.0

    def pixmap(self, width: int, height: int) -> QG.QPixmap:
        """Return an opaque pixmap of the given logical size"""
        pixmap = QG.QPixmap(int(width * self.RATIO), int(height * self.RATIO))
        pixmap.setDevicePixelRatio(self.RATIO)
        pixmap.fill(QC.Qt.black)
        return pixmap


def test_error_icon_covers_whole_icon_at_high_dpi() -> None:
    """The red-tinted icon (failed plugins) has the size and content of the
    original icon at high DPI, instead of covering its top-left part only."""
    with qt_app_context():
        red_icon = PluginStatus._make_red_icon(HighDpiIcon())
        sizes = red_icon.availableSizes()
        assert len(sizes) == 1
        image = red_icon.pixmap(sizes[0]).toImage()
        size = int(PluginStatus.ICON_TINT_SIZE * HighDpiIcon.RATIO)
        assert image.width() == image.height() == size
        error_color = QG.QColor(PLUGIN_ERROR_COLOR).rgb()
        corners = ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        for x, y in corners:
            assert image.pixelColor(x, y).rgb() == error_color


@pytest.mark.parametrize("has_errors", [False, True])
def test_plugin_status_label_requests_configuration(
    monkeypatch: pytest.MonkeyPatch, has_errors: bool
) -> None:
    """Only a left click on the enabled label requests plugin configuration."""
    info = "<i>Plugin information</i>"
    failed = [FailedPluginInfo("Failed plugin", "plugin.py", "Load error")]
    monkeypatch.setattr(PluginRegistry, "get_plugins", lambda: [])
    monkeypatch.setattr(
        PluginRegistry, "get_failed_plugins", lambda: failed if has_errors else []
    )
    monkeypatch.setattr(PluginRegistry, "get_plugin_info", lambda: info)
    Conf.plugins_enabled.set(True)

    with qt_app_context():
        widget = PluginStatus()
        requests: list[bool] = []
        widget.SIG_CONFIGURE_PLUGINS.connect(lambda: requests.append(True))
        widget.show()
        QW.QApplication.processEvents()
        try:
            assert widget.label.cursor().shape() == QC.Qt.PointingHandCursor
            assert widget.icon.cursor().shape() == QC.Qt.ArrowCursor
            assert widget.label.text() == _("Plugins:") + f" 0/{int(has_errors)}"
            assert widget.toolTip() == info + "<br>" + _("Click to configure plugins")
            assert widget.label.styleSheet() == ("color: red" if has_errors else "")
            expected_icon = widget.ko_icon if has_errors else widget.ok_icon
            assert (
                widget.icon.pixmap().cacheKey()
                == expected_icon.pixmap(widget.label.sizeHint().height()).cacheKey()
            )

            for button in (QC.Qt.RightButton, QC.Qt.MiddleButton):
                QT.QTest.mouseClick(widget.label, button)
            QT.QTest.mouseClick(widget.icon, QC.Qt.LeftButton)
            assert requests == []

            QT.QTest.mouseClick(widget.label, QC.Qt.LeftButton)
            assert requests == [True]
        finally:
            widget.close()
            widget.deleteLater()
            QW.QApplication.processEvents()


def test_disabled_plugin_status_does_not_request_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The disabled status stays inert and becomes clickable when re-enabled."""
    info = "<i>Plugin information</i>"
    monkeypatch.setattr(PluginRegistry, "get_plugins", lambda: [])
    monkeypatch.setattr(PluginRegistry, "get_failed_plugins", lambda: [])
    monkeypatch.setattr(PluginRegistry, "get_plugin_info", lambda: info)
    Conf.plugins_enabled.set(False)

    with qt_app_context():
        widget = PluginStatus()
        requests: list[bool] = []
        widget.SIG_CONFIGURE_PLUGINS.connect(lambda: requests.append(True))
        widget.show()
        QW.QApplication.processEvents()
        try:
            assert not widget.isEnabled()
            assert widget.label.text() == _("Plugins:") + " -"
            assert widget.label.cursor().shape() == QC.Qt.ArrowCursor
            assert widget.toolTip() == info
            QT.QTest.mouseClick(widget.label, QC.Qt.LeftButton)
            assert requests == []

            Conf.plugins_enabled.set(True)
            widget.update_status()
            assert widget.isEnabled()
            assert widget.label.cursor().shape() == QC.Qt.PointingHandCursor
            QT.QTest.mouseClick(widget.label, QC.Qt.LeftButton)
            assert requests == [True]

            Conf.plugins_enabled.set(False)
            widget.update_status()
            QT.QTest.mouseClick(widget.label, QC.Qt.LeftButton)
            assert requests == [True]
            assert widget.toolTip() == info
        finally:
            widget.close()
            widget.deleteLater()
            QW.QApplication.processEvents()


if __name__ == "__main__":
    test_error_icon_covers_whole_icon_at_high_dpi()
