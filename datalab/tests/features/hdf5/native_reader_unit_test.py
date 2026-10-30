# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Native HDF5 reader unit tests
-----------------------------

The DataLab version that saved a native HDF5 file is read once, whatever the way
it is stored, and tells whether the file stores automatic curve styles.
"""

# guitest: skip

from __future__ import annotations

import h5py
import numpy as np
import pytest

from datalab.h5.native import DATALAB_VERSION_NAME, NativeH5Reader


@pytest.mark.parametrize(
    "stored, saved_version, legacy",
    [
        ("1.4.0", "1.4.0", False),
        ("1.3.2", "1.3.2", True),
        (np.bytes_(b"1.2.1"), "1.2.1", True),  # Stored as bytes
        ("not a version", "not a version", False),
        (42, None, False),  # Not a string
    ],
)
def test_saved_version(tmp_path, stored, saved_version, legacy: bool) -> None:
    """The saved version is decoded, and legacy curve styles are detected"""
    filename = str(tmp_path / "workspace.h5")
    with h5py.File(filename, "w") as h5file:
        h5file[DATALAB_VERSION_NAME] = stored
    reader = NativeH5Reader(filename)
    try:
        assert reader.saved_version == saved_version
        assert reader.has_legacy_curve_styles is legacy
    finally:
        reader.close()
