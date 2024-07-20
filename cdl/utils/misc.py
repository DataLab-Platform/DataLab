# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Miscelleneous utilities
"""

from __future__ import annotations

import os.path as osp
import re
import subprocess

from cdl.config import Conf, get_mod_source_dir


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
        mod_src_dir = get_mod_source_dir()
        if not osp.isfile(path) and mod_src_dir is not None:
            otherpath = osp.join(mod_src_dir, path)
            if not osp.isfile(otherpath):
                # TODO: [P3] For frozen app, go to error is implemented only when the
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


def is_version_at_least(version1: str, version2: str) -> bool:
    """
    Compare two version strings to check if the first version is at least
    equal to the second. Limit the comparison to the minor version (e.g. 1.2.3 -> 1.2).

    Args:
        version1 (str): The first version string.
        version2 (str): The second version string.

    Returns:
        bool: True if version1 is greater than or equal to version2, False otherwise.

    .. note::

        Development, alpha, beta, and rc versions are considered to be equal
        to the corresponding release version.
    """
    # Split the version strings into parts
    parts1 = [part.strip() for part in version1.split(".")]
    parts2 = [part.strip() for part in version2.split(".")]

    for part1, part2 in zip(parts1, parts2):
        if part1.isdigit() and part2.isdigit():
            if int(part1) > int(part2):
                return True
            if int(part1) < int(part2):
                return False
        elif part1 > part2:
            return True
        elif part1 < part2:
            return False

    return len(parts1) >= len(parts2)


def compare_versions(version1: str, operator: str, version2: str) -> bool:
    """Compare module version with the given version.

    Args:
        version1: Version to compare (e.g., "1.2.3")
        operator: Comparison operator (e.g., "==", "<", ">", "<=", ">=")
        version2: Version to compare with (e.g., "1.2.3")

    Returns:
        True if the comparison is successful, False otherwise.
    """
    specs1, specs2 = version1.split("."), version2.split(".")
    assert len(specs2) <= len(specs1)
    specs2 = specs2[: len(specs1)]
    tuple1, tuple2 = tuple(map(int, specs1)), tuple(map(int, specs2))
    if operator == "==":
        return tuple1 == tuple2
    if operator == "<":
        return tuple1 < tuple2
    if operator == ">":
        return tuple1 > tuple2
    if operator == "<=":
        return tuple1 <= tuple2
    if operator == ">=":
        return tuple1 >= tuple2
    raise ValueError("Invalid operator")
