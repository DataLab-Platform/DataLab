# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
I/O application test:

  - Testing signals I/O
  - Testing images I/O
"""

# guitest: show

import os.path as osp

from cdl.core.gui.panel.base import BaseDataPanel
from cdl.core.io.base import BaseIORegistry, IOAction
from cdl.core.io.image import ImageIORegistry
from cdl.core.io.signal import SignalIORegistry
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.utils.tests import CDLTemporaryDirectory, get_test_fnames


def __test_func(
    title: str, panel: BaseDataPanel, registry: BaseIORegistry, pattern: str
) -> None:
    """Test I/O features"""
    execenv.print(f"  {title}:")
    with CDLTemporaryDirectory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern)
        execenv.print("    Opening:")
        # TODO: This test does not support formats that return multiple objects
        # (e.g. SIF files with multiple images). As a consequence, it will not test
        # thoroughly the I/O functionalities for these formats (it will keep only the
        # first object in the list of returned objects)
        objs = []
        for fname in fnames:
            execenv.print(f"      {fname}: ", end="")
            try:
                registry.get_format(fname, IOAction.LOAD)
            except NotImplementedError:
                execenv.print("Skipped (not supported)")
                continue
            objs.append(panel.load_from_files([fname])[0])
            execenv.print("OK")
        execenv.print("    Saving:")
        for fname, obj in zip(fnames, objs):
            panel.objview.set_current_object(obj)
            path = osp.join(tmpdir, osp.basename(fname))
            execenv.print(f"      {path}: ", end="")
            try:
                registry.get_format(fname, IOAction.SAVE)
            except NotImplementedError:
                execenv.print("Skipped (not supported)")
                continue
            panel.save_to_files([path])
            execenv.print("OK")


def test_io_app() -> None:
    """Run image tools test scenario"""
    with cdltest_app_context() as win:
        execenv.print("I/O application test:")
        __test_func("Signals", win.signalpanel, SignalIORegistry, "curve_formats/*.*")
        __test_func("Images", win.imagepanel, ImageIORegistry, "image_formats/*.*")


if __name__ == "__main__":
    test_io_app()
