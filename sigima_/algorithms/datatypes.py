# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Data Type Conversion Algorithms (see parent package :mod:`sigima_.algorithms`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np


def is_integer_dtype(dtype: np.dtype) -> bool:
    """Return True if data type is an integer type

    Args:
        dtype: Data type to check

    Returns:
        True if data type is an integer type
    """
    return issubclass(np.dtype(dtype).type, np.integer)


def is_complex_dtype(dtype: np.dtype) -> bool:
    """Return True if data type is a complex type

    Args:
        dtype: Data type to check

    Returns:
        True if data type is a complex type
    """
    return issubclass(np.dtype(dtype).type, complex)


def clip_astype(data: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Convert array to a new data type, after having clipped values to the new
    data type's range if it is an integer type.
    If data type is not integer, this is equivalent to ``data.astype(dtype)``.

    Args:
        data: Array to convert
        dtype: Data type to convert to

    Returns:
        Array converted to new data type
    """
    if is_integer_dtype(dtype):
        return np.clip(data, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return data.astype(dtype)
