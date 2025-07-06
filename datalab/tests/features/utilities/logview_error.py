# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Log viewer test: raise an exception and create a seg fault in DataLab
"""

# guitest: skip

from guidata.qthelpers import qt_app_context

from datalab.env import execenv
from datalab.gui.main import DLMainWindow


def error():
    """Raise an exception and create a seg fault in DataLab"""
    with execenv.context(unattended=True):
        with qt_app_context(exec_loop=True):
            win = DLMainWindow()
            win.test_segfault_error()


if __name__ == "__main__":
    error()
