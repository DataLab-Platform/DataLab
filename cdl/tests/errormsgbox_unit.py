# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Error message box test
"""

from qtpy import QtWidgets as QW

from cdl.utils.qthelpers import exec_dialog, qt_app_context
from cdl.widgets.errormessagebox import ErrorMessageBox

SHOW = True  # Show test in GUI-based test launcher


def test_error_message_box():
    """Test error message box"""
    with qt_app_context():
        win = QW.QMainWindow()
        win.setWindowTitle("DataLab Error Message Box test")
        win.show()
        try:
            raise ValueError("Test error message box")
        except ValueError:
            context = "Test_error_message_box." * 5
            tip = "This error may occured when testing the error message box. " * 10
            dlg = ErrorMessageBox(win, context, tip)
            exec_dialog(dlg)


if __name__ == "__main__":
    test_error_message_box()
