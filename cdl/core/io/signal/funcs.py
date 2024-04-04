# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab I/O signal functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re

import numpy as np


def read_csv(
    filename: str,
) -> tuple[
    np.ndarray, str | None, str | None, list[str] | None, list[str] | None, str | None
]:
    """Read CSV data, and return tuple (xydata, xlabel, xunit, ylabels, yunits, header).

    Args:
        filename: CSV file name

    Returns:
        Tuple (xydata, xlabel, xunit, ylabels, yunits, header)
    """
    xydata, xlabel, xunit, ylabels, yunits, header = None, None, None, None, None, None
    for delimiter, comments in (
        (x, y) for x in (";", "\t", ",", " ") for y in ("#", None)
    ):
        try:
            # Load everything readable (titles are eventually converted as NaNs)
            xydata = np.genfromtxt(
                filename, delimiter=delimiter, comments=comments, dtype=float
            )
            if np.all(np.isnan(xydata)):
                continue
            # Removing columns with all but NaNs
            xydata = xydata[:, ~np.all(np.isnan(xydata), axis=0)]
            # Removing lines with NaNs
            xydata = xydata[~np.isnan(xydata).any(axis=1), :]
            if xydata.size == 0:
                raise ValueError("No data")
            # Trying to read X,Y titles
            vals0 = [str(val) for val in xydata[0]]
            nb_of_y_columns = len(vals0) - 1
            line0 = delimiter.join(vals0)
            header = ""
            with open(filename, "r", encoding="utf-8") as fdesc:
                lines = fdesc.readlines()
                for rawline in lines:
                    if comments is not None and rawline.startswith(comments):
                        header += rawline
                        continue
                    line = rawline.replace(" ", "")
                    if line == line0:
                        break
                    try:
                        labels = rawline.split(delimiter)
                        if len(labels) == nb_of_y_columns + 1:
                            xlabel = labels[0]
                            ylabels = labels[1:]
                            xlabel = xlabel.strip()
                            ylabels = [label.strip() for label in ylabels]
                            yunits = [""] * len(ylabels)

                            # Trying to parse X,Y units
                            pattern = r"([\S ]*) \(([\S]*)\)"
                            match = re.match(pattern, xlabel)
                            if match is not None:
                                xlabel, xunit = match.groups()
                            for i, ylabel in enumerate(ylabels):
                                match = re.match(pattern, ylabel)
                                if match is not None:
                                    ylabels[i], yunits[i] = match.groups()
                    except ValueError:
                        pass
                    break
            break
        except ValueError:
            continue
    return xydata, xlabel, xunit, ylabels, yunits, header


def write_csv(
    filename: str,
    xydata: np.ndarray,
    xlabel: str | None,
    xunit: str | None,
    ylabels: list[str] | None,
    yunits: list[str] | None,
    header: str | None,
) -> None:
    """Write CSV data.

    Args:
        filename: CSV file name
        xydata: XY data
        xlabel: X label
        xunit: X unit
        ylabels: Y labels
        yunits: Y units
        header: Header
    """
    labels = ""
    delimiter = ","
    if len(ylabels) == 1:
        ylabels = ["Y"] if not ylabels[0] else ylabels
    elif ylabels:
        ylabels = [f"Y{i+1}" if not label else label for i, label in enumerate(ylabels)]
        if yunits:
            ylabels = [
                f"{label} ({unit})" if unit else label
                for label, unit in zip(ylabels, yunits)
            ]
    if ylabels:
        xlabel = xlabel or "X"
        if xunit:
            xlabel += f" ({xunit})"
        labels = delimiter.join([xlabel] + ylabels)
    np.savetxt(
        filename,
        xydata,
        header=labels,
        delimiter=delimiter,
        comments=header,
    )
