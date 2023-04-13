# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Log viewer test: raise an exception and create a seg fault in CobraDataLab
"""


from cdl.core.gui.main import CDLMainWindow
from cdl.env import execenv
from cdl.utils.qthelpers import qt_app_context

SHOW = False  # Do not show test in GUI-based test launcher


def error():
    """Raise an exception and create a seg fault in CobraDataLab"""
    execenv.unattended = True
    with qt_app_context(exec_loop=True):
        win = CDLMainWindow()
        win.test_segfault_error()


if __name__ == "__main__":
    error()
