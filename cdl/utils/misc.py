# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Miscelleneous utilities
"""

from __future__ import annotations

import difflib
import os.path as osp
import re
import subprocess
import webbrowser

import numpy as np

from cdl.config import Conf, get_mod_source_dir


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


def go_to_error(text: str) -> None:
    """Go to error: open file and go to line number.

    Args:
        text (str): Error text
    """
    pattern = r'File "(.+)", line (\d+),'
    match = re.search(pattern, text)
    if match:
        path = match.group(1)
        line_number = match.group(2)
        if not osp.isfile(path):
            otherpath = osp.join(get_mod_source_dir(), path)
            if not osp.isfile(otherpath):
                # TODO: For frozen app, go to error is  implemented only when the
                # source code is available locally (development mode).
                # How about using a web browser to open the source code on github?
                return
            path = otherpath
        if not osp.isfile(path):
            return  # File not found (unhandled case)
        fdict = {"path": path, "line_number": line_number}
        args = Conf.console.external_editor_args.get().format(**fdict).split(" ")
        editor_path = Conf.console.external_editor_path.get()
        subprocess.run([editor_path] + args, shell=True, check=False)


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
