# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Remote client connection dialog example
"""

# guitest: show,skip

from guidata.configtools import get_icon, get_image_file_path
from guidata.qthelpers import qt_app_context
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigimax.widgets.connection import ConnectionDialog

from datalab.control.proxy import RemoteProxy


def test_dialog():
    """Test connection dialog"""
    proxy = RemoteProxy(autoconnect=False)
    with qt_app_context():
        dlg = ConnectionDialog(
            proxy.connect,
            icon=get_icon("DataLab.svg"),
            banner=QG.QPixmap(get_image_file_path("DataLab-Banner-200.png")),
        )
        if dlg.exec():
            QW.QMessageBox.information(None, "Connection", "Successfully connected")
        else:
            QW.QMessageBox.critical(None, "Connection", "Connection failed")


if __name__ == "__main__":
    test_dialog()
