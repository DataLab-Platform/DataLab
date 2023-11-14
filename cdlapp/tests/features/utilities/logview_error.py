# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Log viewer test: raise an exception and create a seg fault in DataLab
"""

# guitest: skip

from guidata.qthelpers import qt_app_context

from cdlapp.core.gui.main import CDLMainWindow
from cdlapp.env import execenv


def error():
    """Raise an exception and create a seg fault in DataLab"""
    execenv.unattended = True
    with qt_app_context(exec_loop=True):
        win = CDLMainWindow()
        win.test_segfault_error()


if __name__ == "__main__":
    error()
