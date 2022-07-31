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


def test():
    """Run image tools test scenario"""
    with temporary_directory() as tmpdir:
        # os.startfile(tmpdir)
        with codraft_app_context() as win:
            execenv.print("I/O application test:")

            # === Testing Signal I/O ---------------------------------------------------
            execenv.print("  Signals:")
            panel = win.signalpanel
            fnames = get_test_fnames("curve_formats/*.*")
            execenv.print("    Opening:")
            for fname in fnames:
                execenv.print(f"      {fname}")
                panel.open_object(fname)
            panel.objlist.select_all_rows()
            execenv.print(f"    Saving")
            panel.save_objects(
                [osp.join(tmpdir, osp.basename(name)) for name in fnames]
            )

            # === Testing Image I/O ----------------------------------------------------
            execenv.print("  Images:")
            panel = win.imagepanel
            fnames = get_test_fnames("image_formats/*.*")
            execenv.print("    Opening:")
            for fname in fnames:
                try:
                    iohandler.get_readfunc(osp.splitext(fname)[1])
                except RuntimeError:
                    execenv.print(f"      Skipping {fname}")
                    continue
                execenv.print(f"      {fname}")
                panel.open_object(fname)
            execenv.print(f"    Saving:")
            for row, fname in enumerate(fnames):
                panel.objlist.set_current_row(row)
                try:
                    iohandler.get_writefunc(osp.splitext(fname)[1])
                except RuntimeError:
                    execenv.print(f"      Skipping {fname}")
                    continue
                execenv.print(f"      {fname}")
                panel.save_objects([osp.join(tmpdir, osp.basename(fname))])


if __name__ == "__main__":
    test()
