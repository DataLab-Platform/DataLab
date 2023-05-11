# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab I/O signal functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re

import numpy as np


def read_csv(filename: str) -> tuple[np.ndarray, str, str, str, str, str]:
    """Read CSV data, and return tuple (xydata, xlabel, xunit, ylabel, yunit, header).

    Parameters
    ----------
    filename: str
        CSV file name.

    Returns
    -------
    data: np.ndarray
        Data array.
    xlabel: str
        X axis label.
    xunit: str
        X axis unit.
    ylabel: str
        Y axis label.
    yunit: str
        Y axis unit.
    header: str
        Header.
    """
    xydata, xlabel, xunit, ylabel, yunit, header = None, "", "", "", "", ""
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
                        xlabel = xlabel.strip()
                        ylabel = ylabel.strip()
                        # Trying to parse X,Y units
                        pattern = r"([\S ]*) \(([\S]*)\)"  # Matching "Label (unit)"
                        match = re.match(pattern, xlabel)
                        if match is not None:
                            xlabel, xunit = match.groups()
                        match = re.match(pattern, ylabel)
                        if match is not None:
                            ylabel, yunit = match.groups()
                    except ValueError:
                        pass
                    break
            break
        except ValueError:
            continue
    return xydata, xlabel, xunit, ylabel, yunit, header


def write_csv(
    filename: str,
    xydata: np.ndarray,
    xlabel: str,
    xunit: str,
    ylabel: str,
    yunit: str,
    header: str,
) -> None:
    """Write CSV data.

    Parameters
    ----------
    filename: str
        CSV file name.
    xydata: np.ndarray
        Data array.
    xlabel: str
        X axis label.
    xunit: str
        X axis unit.
    ylabel: str
        Y axis label.
    yunit: str
        Y axis unit.
    header: str
        Header.
    """
    xlabel, ylabel = xlabel or "X", ylabel or "Y"
    if xunit:
        xlabel += f" ({xunit})"
    if yunit:
        ylabel += f" ({yunit})"
    delimiter = ","
    np.savetxt(
        filename,
        xydata.T,
        header=delimiter.join([xlabel, ylabel]),
        delimiter=delimiter,
        comments=header,
    )
