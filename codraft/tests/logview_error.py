# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Log viewer test: raise an exception and create a seg fault in CodraFT
"""


from codraft.core.gui.main import CodraFTMainWindow
from codraft.env import execenv
from codraft.utils.qthelpers import qt_app_context

SHOW = False  # Do not show test in GUI-based test launcher


def error():
    """Raise an exception and create a seg fault in CodraFT"""
    execenv.unattended = True
    with qt_app_context(exec_loop=True):
        win = CodraFTMainWindow()
        win.test_segfault_error()


if __name__ == "__main__":
    error()
