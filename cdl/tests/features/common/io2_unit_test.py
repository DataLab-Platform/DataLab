# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

import os.path as osp

from cdl.core.io.base import BaseIORegistry
from cdl.core.io.image import ImageIORegistry
from cdl.core.io.signal import SignalIORegistry
from cdl.env import execenv
from cdl.plugins import discover_plugins
from cdl.utils.strings import reduce_path
from cdl.utils.tests import CDLTemporaryDirectory, get_test_fnames

discover_plugins()


def __test_func(title: str, registry: BaseIORegistry, pattern: str) -> None:
    """Test I/O features"""
    execenv.print(f"  {title}:")
    with CDLTemporaryDirectory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern)
        execenv.print("    Opening:")
        objects = {}
        for fname in fnames:
            try:
                execenv.print(f"      {reduce_path(fname)}: ", end="")
                obj = registry.read(fname)[0]
                objects[fname] = obj
                execenv.print("OK")
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
    __test_func("Signals", SignalIORegistry, "curve_formats/*.*")
    __test_func("Images", ImageIORegistry, "image_formats/*.*")


if __name__ == "__main__":
    test_io2()
