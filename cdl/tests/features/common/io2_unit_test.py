# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O test

Testing DataLab specific formats.
"""

# guitest: show

import os.path as osp

from cdl.core.io.base import BaseIORegistry
from cdl.core.io.image import ImageIORegistry
from cdl.core.io.signal import SignalIORegistry
from cdl.core.model.image import ImageObj
from cdl.core.model.signal import SignalObj
from cdl.env import execenv
from cdl.plugins import discover_plugins
from cdl.utils.qthelpers import CallbackWorker, long_callback
from cdl.utils.strings import reduce_path
from cdl.utils.tests import CDLTemporaryDirectory, get_test_fnames

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

            def callback(worker: CallbackWorker) -> list[SignalObj] | list[ImageObj]:
                """Callback function"""
                return registry.read(fname, worker)[0]

            worker = CallbackWorker(callback)
            label = f"    Opening {reduce_path(fname)}"
            try:
                obj = long_callback(label, worker, fname.endswith(".sif"))
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
