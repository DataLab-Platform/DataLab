# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 import application test
"""

# guitest: show

from datalab import app
from datalab.env import execenv
from datalab.tests import helpers
from datalab.utils.qthelpers import datalab_app_context


def test_hdf5_import():
    """Testing DataLab app launcher"""
    with datalab_app_context(exec_loop=True):
        win = app.create(console=False)
        fname = helpers.get_test_fnames("*.h5")[-1]
        execenv.print(f"Importing HDF5 file: {fname}")
        win.import_h5_file(fname)


if __name__ == "__main__":
    test_hdf5_import()
