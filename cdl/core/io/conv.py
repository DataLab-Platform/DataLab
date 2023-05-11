# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O conversion functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np


def data_to_xy(data: np.ndarray) -> list[np.ndarray]:
    """Convert 2-D array into a list of 1-D array data (x, y, dx, dy).
    This is useful for importing data and creating a DataLab signal with it."""
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
