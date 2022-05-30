# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Application launcher test 1

Running application a few times in a row with different entry parameters.
"""

from codraft import app
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.tests import get_test_fnames

SHOW = True  # Show test in GUI-based test launcher


def test(pattern=None):
    """Testing CodraFT app launcher"""
    if pattern is None:
        pattern = "*.h5"
    print("HDF5 import test scenario:")
    print("[1] Loading all h5 files at once")
    with qt_app_context(exec_loop=True):
        app.create(h5files=get_test_fnames(pattern))
    print("[2] Loading h5 files one by one")
    for fname in get_test_fnames(pattern):
        with qt_app_context(exec_loop=True):
            print(f"    Opening: {fname}")
            app.create(h5files=[fname])


if __name__ == "__main__":
    test()
