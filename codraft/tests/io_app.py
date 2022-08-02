# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
I/O application test:

  - Testing signals I/O
  - Testing images I/O
"""

import os.path as osp

from guiqwt.io import iohandler

from codraft.env import execenv
from codraft.tests import codraft_app_context
from codraft.utils.tests import get_test_fnames, temporary_directory

SHOW = True  # Show test in GUI-based test launcher


def __test_io_features(title, panel, pattern, getreadfunc=None, getwritefunc=None):
    """ "Test I/O features"""
    execenv.print(f"  {title}:")
    with temporary_directory() as tmpdir:
        # os.startfile(tmpdir)
        fnames = get_test_fnames(pattern)
        execenv.print("    Opening:")
        for fname in fnames:
            if getreadfunc is not None:
                try:
                    getreadfunc(osp.splitext(fname)[1])
                except RuntimeError:
                    execenv.print(f"      Skipping {fname}")
                    continue
            execenv.print(f"      {fname}")
            panel.open_object(fname)
        execenv.print("    Saving:")
        for row, fname in enumerate(fnames):
            panel.objlist.set_current_row(row)
            if getwritefunc is not None:
                try:
                    getwritefunc(osp.splitext(fname)[1])
                except RuntimeError:
                    execenv.print(f"      Skipping {fname}")
                    continue
            path = osp.join(tmpdir, osp.basename(fname))
            execenv.print(f"      {path}")
            panel.save_objects([path])


def test():
    """Run image tools test scenario"""
    with codraft_app_context() as win:
        execenv.print("I/O application test:")

        # === Testing Signal I/O ---------------------------------------------------
        __test_io_features("Signals", win.signalpanel, "curve_formats/*.*")

        # === Testing Image I/O ----------------------------------------------------
        __test_io_features(
            "Images",
            win.imagepanel,
            "image_formats/*.*",
            getreadfunc=iohandler.get_readfunc,
            getwritefunc=iohandler.get_writefunc,
        )


if __name__ == "__main__":
    test()
