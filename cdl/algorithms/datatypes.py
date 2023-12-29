# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
.. Data Type Conversion Algorithms (see parent package :mod:`cdl.algorithms`)
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
