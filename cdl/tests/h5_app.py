# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Application launcher test 1

Running application a few times in a row with different entry parameters.
"""

from cdl import app
from cdl.env import execenv
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.tests import get_test_fnames

SHOW = True  # Show test in GUI-based test launcher


def test(pattern=None):
    """Testing DataLab app launcher"""
    if pattern is None:
        pattern = "*.h5"
    execenv.print("HDF5 import test scenario:")
    execenv.print("[1] Loading all h5 files at once")
    with qt_app_context(exec_loop=True):
        app.create(h5files=get_test_fnames(pattern))
    execenv.print("[2] Loading h5 files one by one (only the first 3 files)")
    for fname in get_test_fnames(pattern)[:3]:
        with qt_app_context(exec_loop=True):
            execenv.print(f"      Opening: {fname}")
            app.create(h5files=[fname])


if __name__ == "__main__":
    test()
