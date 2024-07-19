# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab strings utilities
"""

from __future__ import annotations

import difflib
import os.path as osp
import webbrowser
from typing import Any


def to_string(obj: Any) -> str:
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


def reduce_path(filename: str) -> str:
    """Reduce a file path to a relative path"""
    return osp.relpath(filename, osp.join(osp.dirname(filename), osp.pardir))


def save_html_diff(
    text1: str, text2: str, desc1: str, desc2: str, filename: str
) -> None:
    """Generates HTML diff between two strings, saves it to a file, and opens it
    in a web browser (Windows only).

    Args:
        text1 (str): The first string to compare.
        text2 (str): The second string to compare.
        desc1 (str): Description of the first string.
        desc2 (str): Description of the second string.
        filename (str): The name of the file to save the HTML diff to.

    Returns:
        None
    """
    differ = difflib.HtmlDiff()
    diff_html = differ.make_file(text1.splitlines(), text2.splitlines(), desc1, desc2)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(diff_html)
    webbrowser.open("file://" + osp.realpath(filename))


def shorten_docstring(docstring: str) -> str:
    """Shorten a docstring to a single line

    Args:
        docstring: Docstring

    Returns:
        Shortened docstring
    """
    shorter = docstring.split("\n")[0].strip() if docstring else "-"
    for suffix in (".", ":", ",", "using", "with"):
        # TODO: Use string.removesuffix() when we drop Python 3.8 support
        if shorter.endswith(suffix):
            shorter = shorter[: -len(suffix)]
    return shorter
