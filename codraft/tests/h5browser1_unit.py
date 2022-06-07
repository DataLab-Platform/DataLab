# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
HDF5 browser test 1

Try and open all HDF5 test data available.
"""


from codraft.tests.data import get_test_fnames
from codraft.utils.env import execenv
from codraft.utils.qthelpers import exec_dialog, qt_app_context
from codraft.widgets.h5browser import H5BrowserDialog

SHOW = True  # Show test in GUI-based test launcher


def h5browser_test():
    """HDF5 browser test"""
    with qt_app_context():
        for index, fname in enumerate(get_test_fnames("*.h5")):
            execenv.print(f"Opening: {fname}")
            dlg = H5BrowserDialog(None, size=(1050, 450))
            dlg.setup(fname)
            dlg.browser.tree.toggle_all(True)
            dlg.browser.tree.select_all(True)
            dlg.setObjectName(dlg.objectName() + f"_{index:02d}")
            exec_dialog(dlg)


if __name__ == "__main__":
    h5browser_test()
