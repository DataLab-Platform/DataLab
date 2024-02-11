# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Import wizard test
"""

# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context
from qtpy import QtWidgets as QW

from cdl.tests.data import get_test_fnames
from cdl.widgets.textimport import TextImportWizard


def file_to_clipboard(filename: str) -> None:
    """Copy file content to clipboard"""
    with open(filename, "r") as file:
        text = file.read()
    QW.QApplication.clipboard().setText(text)


def test_import_wizard():
    """Test the import wizard"""
    with qt_app_context():
        for destination, fname in (
            ("signal", "paracetamol.txt"),
            ("image", "fiber.txt"),
        ):
            path = get_test_fnames(fname)[0]
            file_to_clipboard(path)
            wizard = TextImportWizard(destination=destination)
            wizard.resize(800, 600)
            if exec_dialog(wizard):
                for obj in wizard.get_objs():
                    print(obj)


if __name__ == "__main__":
    test_import_wizard()
