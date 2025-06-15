# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

from __future__ import annotations

import os.path as osp

from cdl.env import execenv
from cdl.plugins import discover_plugins
from cdl.utils.strings import reduce_path
from cdl.utils.tests import CDLTemporaryDirectory, get_test_fnames
from sigima_.io.base import BaseIORegistry
from sigima_.io.image import ImageIORegistry
from sigima_.io.signal import SignalIORegistry

discover_plugins()


def __testfunc(
    title: str, registry: BaseIORegistry, pattern: str, in_folder: str
) -> None:
    """Test I/O features"""
    execenv.print(f"  {title}:")
    with CDLTemporaryDirectory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern, in_folder)
        objects = {}
        for fname in fnames:
            label = f"    Opening {reduce_path(fname)}"
            execenv.print(label + ": ", end="")
            try:
                obj = registry.read(fname)[0]
                execenv.print("OK")
                objects[fname] = obj
            except NotImplementedError:
                execenv.print("Skipped (not implemented)")
        execenv.print("    Saving:")
        for fname in fnames:
            obj = objects.get(fname)
            if obj is None:
                continue
            path = osp.join(tmpdir, osp.basename(fname))
            try:
                execenv.print(f"      {path}: ", end="")
                registry.write(path, obj)
                execenv.print("OK")
            except NotImplementedError:
                execenv.print("Skipped (not implemented)")


def test_io2():
    """I/O test"""
    execenv.print("I/O unit test:")
    __testfunc("Signals", SignalIORegistry, "*.*", "curve_formats")
    __testfunc("Images", ImageIORegistry, "*.*", "image_formats")


if __name__ == "__main__":
    test_io2()
