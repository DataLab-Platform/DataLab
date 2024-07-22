# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for image data types
-------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np

import cdl.obj
from cdl.algorithms.datatypes import clip_astype


def get_integer_datatypes() -> list[cdl.obj.ImageDatatypes]:
    """Return all integer data types."""
    return [dtype for dtype in cdl.obj.ImageDatatypes if "int" in dtype.name.lower()]


def test_clip_astype():
    """Test `clip_astype` algorithm"""

    # Test that function do nothing for certain data types
    for dtype1 in cdl.obj.ImageDatatypes:
        for dtype2 in cdl.obj.ImageDatatypes:
            data = np.array([0, 1, 2, 3, 4, 5], dtype=dtype1.value)
            assert np.array_equal(
                clip_astype(data, dtype2.value), data
            ), f"No change: {dtype1.value} -> {dtype2.value}"

    # Test that function handles overflow for integer data types
    for dtype in get_integer_datatypes():
        maxval = np.iinfo(dtype.value).max
        data1 = np.array([maxval], dtype=dtype.value)
        data2 = clip_astype(data1.astype(float) + 1, dtype.value)
        assert data2[0] == maxval, f"Overflow: {dtype.value}"

    # Test that function handles underflow for integer data types
    for dtype in get_integer_datatypes():
        minval = np.iinfo(dtype.value).min
        data1 = np.array([minval], dtype=dtype.value)
        data2 = clip_astype(data1.astype(float) - 1, dtype.value)
        assert data2[0] == minval, f"Underflow: {dtype.value}"


if __name__ == "__main__":
    test_clip_astype()
