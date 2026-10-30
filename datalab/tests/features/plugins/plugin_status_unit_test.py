# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Plugin status bar widget tests."""

from __future__ import annotations

from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from qtpy import QtGui as QG

from datalab.config import PLUGIN_ERROR_COLOR
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


if __name__ == "__main__":
    test_error_icon_covers_whole_icon_at_high_dpi()
