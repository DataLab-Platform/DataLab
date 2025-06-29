# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 browser unit tests 1
-------------------------

Try and open all HDF5 test data available.
"""

# guitest: show

from __future__ import annotations

from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.tests.features.hdf5.h5browser_app_test import create_h5browser_dialog
from sigima_.tests.data import get_test_fnames


def test_h5browser_all_files(pattern=None):
    """HDF5 browser unit test for all available .h5 test files"""
    with qt_app_context():
        fnames = get_test_fnames("*.h5" if pattern is None else pattern)
        for index, fname in enumerate(fnames):
            dlg = create_h5browser_dialog([fname], toggle_all=True, select_all=True)
            dlg.setObjectName(dlg.objectName() + f"_{index:02d}")
            exec_dialog(dlg)


if __name__ == "__main__":
    test_h5browser_all_files()
