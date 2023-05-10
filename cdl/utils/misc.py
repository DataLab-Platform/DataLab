# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Miscelleneous utilities
"""

import os.path as osp

import numpy as np


def to_string(obj):
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


def is_integer_dtype(dtype):
    """Return True if data type is an integer type"""
    return issubclass(np.dtype(dtype).type, np.integer)


def is_complex_dtype(dtype):
    """Return True if data type is a complex type"""
    return issubclass(np.dtype(dtype).type, complex)


def reduce_path(filename: str) -> str:
    """Reduce a file path to a relative path"""
    return osp.relpath(filename, osp.join(osp.dirname(filename), osp.pardir))
