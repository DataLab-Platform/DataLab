# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)


"""
CodraFT Signal I/O module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import os.path as osp
import re

import numpy as np

from codraft.core.io.conv import data_to_xy
from codraft.core.model.signal import SignalParam, create_signal

# ==============================================================================
# Signal I/O functions
# ==============================================================================

# TODO: Add a mechanism to allow extending features to other file formats

HEADER_KEY = "HEADER"


def read_signal(filename: str) -> SignalParam:
    """Read CSV or NumPy files, return a signal object (`SignalParam` instance)"""
    reducepath = osp.relpath(filename, osp.join(osp.dirname(filename), osp.pardir))
    signal = create_signal(reducepath)
    if osp.splitext(filename)[1] == ".npy":
        xydata = np.load(filename)
    else:
        for delimiter, comments in zip(("\t", ",", " ", ";"), (None, "#")):
            try:
                # Load everything readable (titles are eventually converted as NaNs)
                xydata = np.genfromtxt(
                    filename, delimiter=delimiter, comments=comments, dtype=float
                )
                # Removing lines with NaNs
                xydata = xydata[~np.isnan(xydata).any(axis=1), :]
                # Trying to read X,Y titles
                line0 = delimiter.join([str(val) for val in xydata[0]])
                header = ""
                with open(filename, "r", encoding="utf-8") as fdesc:
                    lines = fdesc.readlines()
                    for rawline in lines:
                        if rawline.startswith(comments):
                            header += rawline
                            continue
                        line = rawline.replace(" ", "")
                        if line == line0:
                            break
                        try:
                            xlabel, ylabel = rawline.split(delimiter)
                            signal.xlabel = xlabel.strip()
                            signal.ylabel = ylabel.strip()
                            # Trying to parse X,Y units
                            pattern = r"([\S ]*) \(([\S]*)\)"  # Matching "Label (unit)"
                            match = re.match(pattern, signal.xlabel)
                            if match is not None:
                                signal.xlabel, signal.xunit = match.groups()
                            match = re.match(pattern, signal.ylabel)
                            if match is not None:
                                signal.ylabel, signal.yunit = match.groups()
                        except ValueError:
                            pass
                        break
                if header:
                    signal.metadata[HEADER_KEY] = header
                break
            except ValueError:
                continue
        else:
            raise ValueError("Unable to open CSV file")
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
        xlabel, ylabel = obj.xlabel or "X", obj.ylabel or "Y"
        if obj.xunit:
            xlabel += f" ({obj.xunit})"
        if obj.yunit:
            ylabel += f" ({obj.yunit})"
        delimiter = ","
        np.savetxt(
            filename,
            obj.xydata.T,
            header=delimiter.join([xlabel, ylabel]),
            delimiter=delimiter,
            comments=obj.metadata.get(HEADER_KEY, ""),
        )
