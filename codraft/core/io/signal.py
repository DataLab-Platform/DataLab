# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)


"""
CodraFT Signal I/O module
"""

import os.path as osp
import pandas

import numpy as np
from codraft.core.io.conv import data_to_xy

from codraft.core.model.signal import SignalParam, create_signal


# ==============================================================================
# Signal I/O functions
# ==============================================================================

# TODO: Add a mechanism to allow extending features to other file formats


def read_signal(filename: str) -> SignalParam:
    """Read CSV or NumPy files, return a signal object (`SignalParam` instance)"""
    signal = create_signal(osp.basename(filename))
    if osp.splitext(filename)[1] == ".npy":
        xydata = np.load(filename)
    else:
        xydata_df = pandas.read_csv(filename, comment="#")
        xydata = xydata_df.to_numpy()
        signal.xlabel, signal.ylabel = xydata_df.columns[:2]
    assert len(xydata.shape) in (1, 2), "Data not supported"
    if len(xydata.shape) == 1:
        signal.set_xydata(np.arange(xydata.size), xydata)
    else:
        x, y, dx, dy = data_to_xy(xydata)
        signal.set_xydata(x, y, dx, dy)
    return signal


def write_signal(obj: SignalParam, filename: str) -> None:
    """Write signal object to CSV or NumPy file"""
    if osp.splitext(filename)[1] == ".npy":
        np.save(filename, obj.xydata.T)
    else:
        np.savetxt(
            filename,
            obj.xydata.T,
            header=",".join([obj.xlabel or "X", obj.ylabel or "Y"]),
            delimiter=",",
            comments="",
        )
