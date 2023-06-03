# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Error message box test
"""

from qtpy import QtWidgets as QW

from cdl.utils.qthelpers import exec_dialog, qt_app_context
from cdl.widgets.warningerror import WarningErrorMessageBox

SHOW = True  # Show test in GUI-based test launcher


def test_error_message_box(category: str):
    """Test error message box

    Args:
        category (str): Error category
            Valid values are: "error", "warning"
    """
    with qt_app_context():
        win = QW.QMainWindow()
        win.setWindowTitle(f"DataLab {category.capitalize()} Message Box test")
        win.show()
        if category == "error":
            try:
                raise ValueError("Test error message box")
            except ValueError:
                context = "Test_error_message_box." * 5
                tip = "This error may occured when testing the error message box. " * 10
                dlg = WarningErrorMessageBox(win, "error", context, tip=tip)
                exec_dialog(dlg)
        elif category == "warning":
            context = "Test_warning_message_box." * 5
            message = "Test warning message box" * 10
            dlg = WarningErrorMessageBox(win, "warning", context, message)
            exec_dialog(dlg)
        else:
            raise ValueError(f"Invalid category: {category}")


if __name__ == "__main__":
    test_error_message_box("error")
    test_error_message_box("warning")
