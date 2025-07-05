# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image data types
-------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np

from sigima_.algorithms.datatypes import clip_astype
from sigima_.obj import ImageDatatypes
from sigima_.tests.env import execenv


def get_integer_datatypes() -> list[ImageDatatypes]:
    """Return all integer data types."""
    return [dtype for dtype in ImageDatatypes if "int" in dtype.name.lower()]


def test_clip_astype():
    """Test `clip_astype` algorithm"""

    # Test that function do nothing for certain data types
    for dtype1 in ImageDatatypes:
        for dtype2 in ImageDatatypes:
            if not np.issubdtype(dtype2.value, np.integer):
                continue
            if np.issubdtype(dtype1.value, np.integer):
                info1 = np.iinfo(dtype1.value)
            else:
                info1 = np.finfo(dtype1.value)
            info2 = np.iinfo(dtype2.value)
            if info2.min < info1.min or info2.max > info1.max:
                continue
            data = np.array([0, 1, 2, 3, 4, 5], dtype=dtype1.value)
            txt = f"No change: {dtype1.value} -> {dtype2.value}"
            execenv.print(txt, end="... ")
            assert np.array_equal(clip_astype(data, dtype2.value), data), txt
            execenv.print("OK")

    # Test that function handles overflow for integer data types
    for dtype in get_integer_datatypes():
        maxval = np.iinfo(dtype.value).max
        data1 = np.array([maxval], dtype=dtype.value)
        data2 = clip_astype(data1.astype(float) + 1, dtype.value)
        txt = f"Overflow: {dtype.value}"
        execenv.print(txt, end="... ")
        assert data2[0] == maxval, txt
        execenv.print("OK")

    # Test that function handles underflow for integer data types
    for dtype in get_integer_datatypes():
        minval = np.iinfo(dtype.value).min
        data1 = np.array([minval], dtype=dtype.value)
        data2 = clip_astype(data1.astype(float) - 1, dtype.value)
        txt = f"Underflow: {dtype.value}"
        execenv.print(txt, end="... ")
        assert data2[0] == minval, txt
        execenv.print("OK")


if __name__ == "__main__":
    test_clip_astype()
