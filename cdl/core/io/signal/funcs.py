# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab I/O signal functions
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from cdl.utils.io import count_lines


def get_labels_units_from_dataframe(
    df: pd.DataFrame,
) -> tuple[str, list[str], str, list[str]]:
    """Get labels and units from a DataFrame.

    Args:
        df: DataFrame

    Returns:
        Tuple (xlabel, ylabels, xunit, yunits)
    """
    # Reading X,Y labels
    xlabel = str(df.columns[0])
    ylabels = [str(col) for col in df.columns[1:]]

    # Retrieving units from labels
    xunit = ""
    yunits = [""] * len(ylabels)
    pattern = r"([\S ]*) \(([\S]*)\)"
    match = re.match(pattern, xlabel)
    if match is not None:
        xlabel, xunit = match.groups()
    for i, ylabel in enumerate(ylabels):
        match = re.match(pattern, ylabel)
        if match is not None:
            ylabels[i], yunits[i] = match.groups()

    return xlabel, ylabels, xunit, yunits


def __read_csv_primitive(
    filename: str,
    progress_callback: Callable | None = None,
    delimiter: str | None = None,
    header: int | None = "infer",
    skiprows: int | None = None,
    nrows: int | None = None,
    comment: str | None = None,
    chunksize: int = 1000,
) -> pd.DataFrame:
    """Read CSV data with primitive options, using pandas read_csv function defaults,
    and reading data in chunks, using the iterator interface.

    Args:
        filename: CSV file name
        progress_callback: progress callback function (a function that takes a float
            between 0 and 1 as argument representing the progress, and returns a
            boolean indicating whether to cancel the operation)
        delimiter: Delimiter
        header: Header line
        skiprows: Skip rows
        nrows: Number of rows to read
        comment: Comment character
        chunksize: Chunk size

    Returns:
        DataFrame
    """
    nlines = count_lines(filename)
    # Read data in chunks, and concatenate them at the end, thus allowing to call the
    # progress callback function at each chunk read and to return an intermediate result
    # if the operation is canceled.
    chunks = []
    for chunk in pd.read_csv(
        filename,
        delimiter=delimiter,
        header=header,
        skiprows=skiprows,
        nrows=nrows,
        comment=comment,
        chunksize=chunksize,
    ):
        chunks.append(chunk)
        # Compute the progression based on the number of lines read so far
        progress = sum(len(chunk) for chunk in chunks) / nlines
        if progress_callback is not None and progress_callback(progress):
            break
    return pd.concat(chunks)


def read_csv(
    filename: str,
    progress_callback: Callable,
) -> tuple[
    np.ndarray, str | None, str | None, list[str] | None, list[str] | None, str | None
]:
    """Read CSV data, and return tuple (xydata, xlabel, xunit, ylabels, yunits, header).

    Args:
        filename: CSV file name
        progress_callback: progress callback function (a function that takes a float
         between 0 and 1 as argument representing the progress, and returns a
         boolean indicating whether to cancel the operation)

    Returns:
        Tuple (xydata, xlabel, xunit, ylabels, yunits, header)
    """
    xydata, xlabel, xunit, ylabels, yunits, header = None, None, None, None, None, None

    # The first attempt is to read the CSV file assuming it has no header because it
    # won't raise an error if the first line is data. If it fails, we try to read it
    # with a header, and if it fails again, we try to skip some lines before reading
    # the data.

    skiprows = None

    # First attempt: no header (try to read with different delimiters)
    for delimiter in (",", ";", "\t", " "):
        try:
            df = pd.read_csv(
                filename,
                dtype=float,
                delimiter=delimiter,
                header=None,
                comment="#",
                nrows=1000,  # Read only the first 1000 lines
            )
            read_without_header = True
            break
        except (pd.errors.ParserError, ValueError):
            df = None

    # Second attempt: with header
    if df is None:
        for delimiter in (",", ";", "\t", " "):
            # Headers are generally in the first 10 lines, so we try to skip the
            # minimum number of lines before reading the data:
            for skiprows in range(20):
                try:
                    df = pd.read_csv(
                        filename,
                        dtype=float,
                        delimiter=delimiter,
                        skiprows=skiprows,
                        comment="#",
                        nrows=1000,  # Read only the first 1000 lines
                    )
                    break
                except (pd.errors.ParserError, ValueError):
                    df = None
            if df is not None:
                break

        if df is None:
            raise ValueError("Unable to read CSV file (format not supported)")

        # At this stage, we have a DataFrame with column names, but we don't know
        # if the first line is a header or data. We try to read the first line as
        # a header, and if it fails, we read it as data.
        try:
            df.columns.astype(float)
            # This means that the first line is data, so we have to read it again, but
            # without the header:
            read_without_header = True
        except ValueError:
            read_without_header = False
            # This means that the first line is a header, so we already have the data
            # without missing values.
            # However, it also means that there could be text information preceding
            # the header. Let's try to read it and put it in `header` variable.

            # 1. We read only the first 1000 lines to avoid reading the whole file
            # 2. We keep only the lines beginning with a comment character
            # 3. We join the lines to create a single string
            header = ""
            with open(filename, "r") as file:
                for _ in range(1000):
                    line = file.readline()
                    if line.startswith("#"):
                        header += line
                    else:
                        break
            # Remove the last line if it contains the column names:
            if header and df.columns[0] in header.splitlines()[-1]:
                header = "\n".join(header.splitlines()[:-1])

    # Now we read the whole file with the correct options
    df = __read_csv_primitive(
        filename,
        progress_callback,
        delimiter=delimiter,
        header=None if read_without_header else "infer",
        skiprows=skiprows,
        comment="#",
    )

    # Remove rows and columns where all values are NaN in the DataFrame:
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Converting to NumPy array
    xydata = df.to_numpy(float)
    if xydata.size == 0:
        raise ValueError("Unable to read CSV file (no supported data after cleaning)")

    xlabel, ylabels, xunit, yunits = get_labels_units_from_dataframe(df)
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
    df = pd.DataFrame(xydata.T, columns=[xlabel] + ylabels)
    df.to_csv(filename, index=False, header=labels, sep=delimiter)
    # Add header if present
    if header:
        with open(filename, "r+") as file:
            content = file.read()
            file.seek(0, 0)
            file.write(header + "\n" + content)
