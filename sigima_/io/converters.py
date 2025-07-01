# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O conversion functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import Any

import numpy as np
import skimage


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


def to_string(obj: Any) -> str:
    """Convert to string, trying utf-8 then latin-1 codec"""
    if isinstance(obj, bytes):
        try:
            return obj.decode()
        except UnicodeDecodeError:
            return obj.decode("latin-1")
    try:
        return str(obj)
    except UnicodeDecodeError:
        return str(obj, encoding="latin-1")
