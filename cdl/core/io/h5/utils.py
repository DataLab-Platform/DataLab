# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Utilities for exogenous HDF5 format support
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np

from cdl.utils.misc import to_string


def fix_ldata(fuzzy):
    """Fix label data"""
    if fuzzy is not None:
        if fuzzy and isinstance(fuzzy, np.void) and len(fuzzy) > 1:
            #  Shouldn't happen (invalid LMJ fmt)
            fuzzy = fuzzy[0]
        if isinstance(fuzzy, (np.string_, bytes)):
            fuzzy = to_string(fuzzy)
        if isinstance(fuzzy, str):
            return fuzzy
    return None


def fix_ndata(fuzzy):
    """Fix numeric data"""
    if fuzzy is not None:
        if fuzzy and isinstance(fuzzy, np.void) and len(fuzzy) > 1:
            #  Shouldn't happen (invalid LMJ fmt)
            fuzzy = fuzzy[0]
        try:
            if float(fuzzy) == int(fuzzy):
                return int(fuzzy)
            return float(fuzzy)
        except (TypeError, ValueError):
            pass
    return None


def process_scalar_value(dset, name, callback):
    """Process dataset numeric/str value `name`"""
    try:
        scdata = dset[name][()]
        if scdata is not None:
            return callback(scdata)
    except (KeyError, ValueError):
        pass
    return None


def process_label(dset, name):
    """Process dataset label `name`"""
    try:
        ldata = dset[name][()]
        if ldata is not None:
            xldata, yldata, zldata = None, None, None
            if len(ldata) == 2:
                xldata, yldata = ldata
            elif len(ldata) == 3:
                xldata, yldata, zldata = ldata
            return fix_ldata(xldata), fix_ldata(yldata), fix_ldata(zldata)
    except KeyError:
        pass
    return None, None, None


def process_xy_values(dset, name):
    """Process dataset x,y values `name`"""
    try:
        ldata = dset[name][()]
        if ldata is not None:
            return fix_ndata(ldata[0]), fix_ndata(ldata[1])
    except (KeyError, ValueError):
        pass
    return None, None


def is_supported_num_dtype(data):
    """Return True if data type is a numerical type supported by DataLab"""
    return data.dtype.name.startswith(("int", "uint", "float", "complex"))


def is_single_str_array(data):
    """Return True if data is a single-item string array"""
    return np.issctype(data) and data.shape == (1,) and isinstance(data[0], str)


def is_supported_str_dtype(data):
    """Return True if data type is a string type supported by preview"""
    return data.dtype.name.startswith("string") or is_single_str_array(data)
