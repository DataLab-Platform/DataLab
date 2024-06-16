# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab I/O conversion functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import skimage.util


def data_to_xy(data: np.ndarray) -> list[np.ndarray]:
    """Convert 2-D array into a list of 1-D array data (x, y, dx, dy).
    This is useful for importing data and creating a DataLab signal with it.

    Args:
        data (numpy.ndarray): 2-D array of data

    Returns:
        list[np.ndarray]: list of 1-D array data (x, y, dx, dy)
    """
    if len(data.ravel()) == len(data):
        return np.arange(len(data)), data.ravel(), None, None
    rows, cols = data.shape
    for colnb in (2, 3, 4):
        if cols == colnb and rows > colnb:
            data = data.T
            break
    if len(data) == 1:
        data = data.T
    if len(data) not in (2, 3, 4):
        raise ValueError(f"Invalid data: len(data)={len(data)} (expected 2, 3 or 4)")
    x, y = data[:2]
    dx, dy = None, None
    if len(data) == 3:
        dy = data[2]
    if len(data) == 4:
        dx, dy = data[2:]
    return x, y, dx, dy


def convert_array_to_standard_type(array: np.ndarray) -> np.ndarray:
    """Convert an integer array to a standard type
    (int8, int16, int32, uint8, uint16, uint32).

    Ignores floating point arrays.

    Args:
        array: array to convert

    Raises:
        ValueError: if array is not of integer type

    Returns:
        Converted array
    """
    # Determine the kind and size of the data type
    kind = array.dtype.kind
    itemsize = array.dtype.itemsize

    if kind in ["f", "c"]:  # 'f' for floating point, 'c' for complex
        return array

    if kind == "b":  # 'b' for boolean
        # Convert to uint8
        return skimage.util.img_as_ubyte(array)

    if kind in ["i", "u"]:  # 'i' for signed integers, 'u' for unsigned integers
        if itemsize == 1:  # 8-bit
            new_type = np.dtype(f"{kind}1").newbyteorder("=")
        elif itemsize == 2:  # 16-bit
            new_type = np.dtype(f"{kind}2").newbyteorder("=")
        elif itemsize == 4:  # 32-bit
            new_type = np.dtype(f"{kind}4").newbyteorder("=")
        else:
            raise ValueError("Unsupported item size for integer type")

        # Convert to the new type if it is different from the current type
        if array.dtype != new_type:
            return array.astype(new_type)
        return array

    raise ValueError("Unsupported data type")
