# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Error message box test
"""

# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context
from qtpy import QtWidgets as QW

from datalab.widgets.warningerror import WarningErrorMessageBox


def error_message_box(category: str):
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


def test_error_message_box():
    """Test error message box"""
    error_message_box("error")
    error_message_box("warning")


if __name__ == "__main__":
    test_error_message_box()
