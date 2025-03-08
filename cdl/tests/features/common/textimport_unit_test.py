# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(filename, "r", encoding=encoding) as file:
                text = file.read()
            QW.QApplication.clipboard().setText(text)
            return
        except UnicodeDecodeError:
            pass


def test_import_wizard():
    """Test the import wizard"""
    with qt_app_context():
        for destination, fname, otype in (
            ("image", "fiber.txt", ImageObj),
            ("signal", "multiple_curves.csv", SignalObj),
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
                srcpge.param.path = path
                srcpge.param_widget.get()
                wizard.go_to_next_page()
                datapge = wizard.data_page
                n_objs = 1
                if fname == "fiber.txt":
                    datapge.param.delimiter_choice = " "
                elif fname == "multiple_curves.csv":
                    datapge.param.delimiter_choice = ";"
                    datapge.param.skip_rows = 1
                    n_objs = 5
                else:
                    datapge.param.skip_rows = 10
                datapge.param_widget.get()
                datapge.update_preview()
                wizard.go_to_next_page()
                wizard.go_to_previous_page()  # For test purpose only
                wizard.go_to_next_page()
                wizard.go_to_next_page()
                wizard.accept()
                assert len(wizard.get_objs()) == n_objs
                assert isinstance(wizard.get_objs()[0], otype)
            elif exec_dialog(wizard):
                for obj in wizard.get_objs():
                    execenv.print(obj)


if __name__ == "__main__":
    test_import_wizard()
