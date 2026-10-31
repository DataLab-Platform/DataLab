# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Icon tests

DataLab icons share the image name space of guidata, PlotPy and SigimaX: an icon
requested by its file name must match a single image file, otherwise guidata emits
an ``ImageNameConflictWarning`` (an error in the test suite) and the icon must be
designated explicitly with the ``"module:filename"`` syntax (e.g.
``"datalab:delete.svg"``). This requires DataLab icon file names to be unique, and
DataLab not to ship copies of the icons of the libraries it depends on.
"""

from __future__ import annotations

import collections
import hashlib
import os
import os.path as osp
import re

import plotpy.config  # noqa: F401  (registers PlotPy's icon path)
import pytest
import sigimax.config  # noqa: F401  (registers SigimaX's icon path)
from guidata import configtools

import datalab
import datalab.config  # noqa: F401  (registers DataLab's icon path)

#: Modules providing the icons used by DataLab, by increasing precedence
ICON_MODULES = ("guidata", "plotpy", "sigimax", "datalab")


def _icon_files(modname: str) -> dict[str, list[str]]:
    """Return the SVG image files registered by a module, by file name"""
    roots = configtools.IMG_MODULE_PATH[modname]
    files = collections.defaultdict(list)
    for root in roots:
        for filename in os.listdir(root):
            path = osp.join(root, filename)
            if filename.endswith(".svg") and osp.isfile(path):
                files[filename].append(path)
    return files


def _referenced_icon_names() -> set[str]:
    """Return the SVG image names referenced as string literals in DataLab's code
    (tests and comments excluded) and in its example plugins"""
    names = set()
    pattern = re.compile(r"[\"']((?:[\w.]+:)?[\w\-]+\.svg)[\"']")
    package_dir = osp.dirname(datalab.__file__)
    roots = (package_dir, osp.join(osp.dirname(package_dir), "plugins"))
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath == package_dir:
                dirnames.remove("tests")
            for filename in filenames:
                if filename.endswith(".py"):
                    with open(osp.join(dirpath, filename), encoding="utf-8") as fd:
                        for line in fd:
                            if not line.lstrip().startswith("#"):
                                names.update(pattern.findall(line))
    return names


def test_unique_icon_file_names() -> None:
    """Test that DataLab icon file names are unique"""
    files = _icon_files("datalab")
    assert not {name: paths for name, paths in files.items() if len(paths) > 1}


def test_no_copy_of_library_icons() -> None:
    """Test that DataLab does not ship copies of the icons of the libraries it
    depends on (they would have to be designated explicitly)"""

    def digest(path: str) -> str:
        with open(path, "rb") as fdesc:
            return hashlib.sha1(fdesc.read().replace(b"\r\n", b"\n")).hexdigest()

    library_digests = {
        digest(path): path
        for modname in ICON_MODULES[:-1]
        for paths in _icon_files(modname).values()
        for path in paths
    }
    copies = {
        path: library_digests[digest(path)]
        for paths in _icon_files("datalab").values()
        for path in paths
        if digest(path) in library_digests
    }
    assert not copies


def test_referenced_icons_are_unambiguous() -> None:
    """Test that the icons requested by DataLab exist and match a single image
    file, unless they are designated explicitly"""
    names = _referenced_icon_names()
    assert {"datalab:delete.svg", "sigimax:image.svg"} <= names
    missing, ambiguous = [], {}
    for name in sorted(names):
        if ":" in name:
            try:
                configtools.get_image_file_path(name, default=None)
            except RuntimeError:
                missing.append(name)
            continue
        paths = [
            path
            for modname in ICON_MODULES
            for path in _icon_files(modname).get(name, [])
        ]
        if not paths:
            missing.append(name)
        elif len(paths) > 1:
            ambiguous[name] = paths
    assert not missing
    assert not ambiguous


@pytest.mark.parametrize(
    "name, relpath",
    [
        ("datalab:delete.svg", "edit/delete.svg"),
        ("datalab:image.svg", None),
        ("sigimax:image.svg", None),
        ("datalab:create/square_wave.svg", "create/square_wave.svg"),
    ],
)
def test_explicit_designation(name: str, relpath: str | None) -> None:
    """Test that DataLab icons are designated explicitly, and that the image and
    signal icons are provided by SigimaX"""
    modname = name.split(":", 1)[0]
    if relpath is None and modname == "datalab":
        with pytest.raises(RuntimeError):
            configtools.get_image_file_path(name, default=None)
        return
    path = configtools.get_image_file_path(name, default=None)
    root = configtools.get_module_data_path(modname, "data/icons")
    if relpath is None:
        relpath = name.split(":", 1)[1]
    assert osp.normcase(path) == osp.normcase(osp.join(root, *relpath.split("/")))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
