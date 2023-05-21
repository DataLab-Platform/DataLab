# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Miscelleneous utilities
"""

import os.path as osp
import re
import subprocess

import numpy as np

from cdl.config import Conf


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
        fdict = {"path": path, "line_number": line_number}
        args = Conf.console.external_editor_args.get().format(**fdict).split(" ")
        editor_path = Conf.console.external_editor_path.get()
        subprocess.run([editor_path] + args, shell=True, check=False)
