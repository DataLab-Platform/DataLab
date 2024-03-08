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

from cdl.env import execenv
from cdl.obj import ImageObj, SignalObj
from cdl.tests.data import get_test_fnames
from cdl.widgets.textimport import TextImportWizard


def file_to_clipboard(filename: str) -> None:
    """Copy file content to clipboard"""
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    QW.QApplication.clipboard().setText(text)


def test_import_wizard():
    """Test the import wizard"""
    with qt_app_context():
        for destination, fname, otype in (
            ("image", "fiber.txt", ImageObj),
            ("signal", "paracetamol.txt", SignalObj),
        ):
            path = get_test_fnames(fname)[0]
            if not execenv.unattended:
                # Do not test clipboard in unattended mode, would fail:
                # - Windows: OleSetClipboard: Failed to set mime data (text/plain)
                #            on clipboard: COM error 0xffffffff800401d0
                # - Linux:  QXcbClipboard: Unable to receive an event from the clipboard
                #           manager in a reasonable time
                file_to_clipboard(path)
            wizard = TextImportWizard(destination=destination)
            if execenv.unattended:
                wizard.show()
                srcpge = wizard.source_page
                srcpge.source_widget.file_edit.setText(path)
                srcpge.source_widget.file_edit.editingFinished.emit()
                wizard.go_to_next_page()
                datapge = wizard.data_page
                if fname == "fiber.txt":
                    datapge.param.delimiter_choice = " "
                else:
                    datapge.param.skip_rows = 10
                datapge.param_widget.get()
                datapge.update_preview()
                wizard.go_to_next_page()
                wizard.go_to_previous_page()  # For test purpose only
                wizard.go_to_next_page()
                wizard.go_to_next_page()
                wizard.accept()
                assert len(wizard.get_objs()) == 1
                assert isinstance(wizard.get_objs()[0], otype)
            elif exec_dialog(wizard):
                for obj in wizard.get_objs():
                    execenv.print(obj)


if __name__ == "__main__":
    test_import_wizard()
