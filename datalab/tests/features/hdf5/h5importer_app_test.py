# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 Importer Application test
------------------------------

Running application a few times in a row with different entry parameters.
"""

# guitest: show

from datalab import app
from datalab.env import execenv
from datalab.utils.qthelpers import datalab_app_context
from sigima.tests.helpers import get_test_fnames


def test_h5importer_app(pattern=None):
    """Testing DataLab app launcher"""
    if pattern is None:
        pattern = "*.h5"
    execenv.print("HDF5 import test scenario:")
    execenv.print("[1] Loading all h5 files at once (only the first 5 files)")
    with datalab_app_context(exec_loop=True):
        app.create(h5files=get_test_fnames(pattern)[:5])
    execenv.print("[2] Loading h5 files one by one")
    for fname in get_test_fnames(pattern):
        with datalab_app_context(exec_loop=True):
            execenv.print(f"      Opening: {fname}")
            app.create(h5files=[fname])


if __name__ == "__main__":
    test_h5importer_app()
