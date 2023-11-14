# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
I/O application test:

  - Testing signals I/O
  - Testing images I/O
"""

# guitest: show

import os.path as osp

from cdlapp.core.gui.panel.base import BaseDataPanel
from cdlapp.core.io.base import BaseIORegistry, IOAction
from cdlapp.core.io.image import ImageIORegistry
from cdlapp.core.io.signal import SignalIORegistry
from cdlapp.env import execenv
from cdlapp.tests import test_cdl_app_context
from cdlapp.utils.tests import get_test_fnames, temporary_directory


def __test_func(
    title: str, panel: BaseDataPanel, registry: BaseIORegistry, pattern: str
) -> None:
    """Test I/O features"""
    execenv.print(f"  {title}:")
    with temporary_directory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern)
        execenv.print("    Opening:")
        objs = []
        for fname in fnames:
            execenv.print(f"      {fname}: ", end="")
            try:
                registry.get_format(fname, IOAction.LOAD)
            except NotImplementedError:
                execenv.print("Skipped (not supported)")
                continue
            objs.append(panel.open_object(fname))
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
            panel.save_objects([path])
            execenv.print("OK")


def test() -> None:
    """Run image tools test scenario"""
    with test_cdl_app_context() as win:
        execenv.print("I/O application test:")
        __test_func("Signals", win.signalpanel, SignalIORegistry, "curve_formats/*.*")
        __test_func("Images", win.imagepanel, ImageIORegistry, "image_formats/*.*")


if __name__ == "__main__":
    test()
