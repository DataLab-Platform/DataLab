# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
HDF5 Browser Application test
-----------------------------


"""

# guitest: show

from guidata.qthelpers import exec_dialog, qt_app_context

from cdl.env import execenv
from cdl.tests.data import get_test_fnames
from cdl.widgets.h5browser import H5BrowserDialog


def create_h5browser_dialog(
    fname: str, toggle_all: bool = False, select_all: bool = False
) -> H5BrowserDialog:
    """Create HDF5 browser dialog with all nodes expanded and selected

    Args:
        fname: HDF5 file name

    Returns:
        H5BrowserDialog instance
    """
    execenv.print(f"Opening: {fname}")
    dlg = H5BrowserDialog(None)
    dlg.setup(fname)
    dlg.browser.tree.toggle_all(toggle_all)
    dlg.browser.tree.select_all(select_all)
    return dlg


def test_h5browser() -> None:
    """Test HDF5 browser"""
    with qt_app_context():
        dlg = create_h5browser_dialog(get_test_fnames("*.h5")[-2])
        exec_dialog(dlg)


if __name__ == "__main__":
    test_h5browser()
