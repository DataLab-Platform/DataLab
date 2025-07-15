# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Import wizard test
"""

# guitest: show

import numpy as np
from guidata.dataset import update_dataset
from guidata.qthelpers import exec_dialog, qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj
from sigima.tests.data import get_test_fnames

from datalab.env import execenv
from datalab.widgets.textimport import ImageParam, SignalParam, TextImportWizard


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


TEST_SIGNAL_PARAM = SignalParam.create(
    xlabel="X Signal axis",
    ylabel="Y Signal axis",
    xunit="X Signal unit",
    yunit="Y Signal unit",
)

TEST_IMAGE_PARAM = ImageParam.create(
    xlabel="X Image axis",
    ylabel="Y Image axis",
    zlabel="Z Image axis",
    xunit="X Image unit",
    yunit="Y Image unit",
    zunit="Z Image unit",
)


def test_import_wizard():
    """Test the import wizard"""
    with qt_app_context():
        for destination, fname, otype in (
            ("image", "fiber.txt", ImageObj),
            ("signal", "multiple_curves.csv", SignalObj),
            ("signal", "paracetamol.txt", SignalObj),
            ("signal", "spectrum.mca", SignalObj),
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
                srcpge = wizard.source_page  # `SourcePage`
                srcpge.param.path = path
                srcpge.param_widget.get()
                wizard.go_to_next_page()
                datapge = wizard.data_page  # `DataPreviewPage`
                n_objs = 1
                if destination == "image":
                    datapge.param.dtype_str = "uint8"
                else:
                    datapge.param.dtype_str = "float32"
                if fname == "fiber.txt":
                    datapge.param.delimiter_choice = " "
                elif fname == "multiple_curves.csv":
                    datapge.param.delimiter_choice = ";"
                    datapge.param.skip_rows = 1
                    n_objs = 5
                elif fname == "spectrum.mca":
                    datapge.param.skip_rows = 18
                    datapge.param.max_rows = 2048
                    datapge.param.header = None
                    datapge.param.first_col_is_x = False
                else:
                    datapge.param.skip_rows = 10
                datapge.param_widget.get()
                datapge.update_preview()
                wizard.go_to_next_page()  # Go to `GraphicalRepresentationPage`
                wizard.go_to_previous_page()  # For test purpose only
                wizard.go_to_next_page()  # Go to `GraphicalRepresentationPage`
                labels_page = wizard.labels_page  # `LabelsPage`
                if destination == "image":
                    assert isinstance(labels_page.param, ImageParam)
                    update_dataset(labels_page.param, TEST_IMAGE_PARAM)
                else:
                    assert isinstance(labels_page.param, SignalParam)
                    update_dataset(labels_page.param, TEST_SIGNAL_PARAM)
                wizard.go_to_next_page()  # Go to `LabelsPage`
                wizard.accept()
                objs = wizard.get_objs()
                assert len(objs) == n_objs
                assert isinstance(objs[0], otype)
                # Check that the parameters are set correctly
                for obj in objs:
                    if isinstance(obj, ImageObj):
                        assert obj.data.dtype == np.uint8
                        assert obj.xlabel == TEST_IMAGE_PARAM.xlabel
                        assert obj.ylabel == TEST_IMAGE_PARAM.ylabel
                        assert obj.zlabel == TEST_IMAGE_PARAM.zlabel
                        assert obj.xunit == TEST_IMAGE_PARAM.xunit
                        assert obj.yunit == TEST_IMAGE_PARAM.yunit
                        assert obj.zunit == TEST_IMAGE_PARAM.zunit
                    elif isinstance(obj, SignalObj):
                        assert obj.data.dtype == np.float32
                        assert obj.xlabel == TEST_SIGNAL_PARAM.xlabel
                        assert obj.ylabel == TEST_SIGNAL_PARAM.ylabel
                        assert obj.xunit == TEST_SIGNAL_PARAM.xunit
                        assert obj.yunit == TEST_SIGNAL_PARAM.yunit
            elif exec_dialog(wizard):
                for obj in wizard.get_objs():
                    execenv.print(obj)


if __name__ == "__main__":
    test_import_wizard()
