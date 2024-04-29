# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab I/O utilities
"""

from __future__ import annotations

from itertools import islice


def count_lines(filename: str) -> int:
    """Count the number of lines in a file

    Args:
        filename: File name

    Returns:
        The number of lines in the file
    """
    with open(filename, "r", encoding="utf-8") as file:
        line_count = sum(1 for line in file)
    return line_count


def read_first_n_lines(filename: str, n: int = 100000) -> str:
    """Read the first n lines of a file

    Args:
        filename: File name
        n: Number of lines to read

    Returns:
        The first n lines of the file
    """
    with open(filename, "r", encoding="utf-8") as file:
        lines = list(islice(file, n))
    return "".join(lines)
