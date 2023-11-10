# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

import os.path as osp

from cdlapp.core.io.base import BaseIORegistry
from cdlapp.core.io.image import ImageIORegistry
from cdlapp.core.io.signal import SignalIORegistry
from cdlapp.env import execenv
from cdlapp.plugins import discover_plugins
from cdlapp.utils.misc import reduce_path
from cdlapp.utils.tests import get_test_fnames, temporary_directory

discover_plugins()


def __test_func(title: str, registry: BaseIORegistry, pattern: str) -> None:
    """Test I/O features"""
    execenv.print(f"  {title}:")
    with temporary_directory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern)
        execenv.print("    Opening:")
        objects = {}
        for fname in fnames:
            try:
                execenv.print(f"      {reduce_path(fname)}: ", end="")
                obj = registry.read(fname)
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


def io_test():
    """I/O test"""
    execenv.print("I/O unit test:")
    __test_func("Signals", SignalIORegistry, "curve_formats/*.*")
    __test_func("Images", ImageIORegistry, "image_formats/*.*")


if __name__ == "__main__":
    io_test()
