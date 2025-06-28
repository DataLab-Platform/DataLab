# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Data Type Conversion Algorithms (see parent package :mod:`sigima_.algorithms`)
"""

from __future__ import annotations

import numpy as np


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
    if np.issubdtype(dtype, np.integer):
        return np.clip(data, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)
    return data.astype(dtype)
