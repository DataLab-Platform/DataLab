# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Basic application launcher test 1

Running application a few times in a row with different entry parameters.
"""

# guitest: show

from datalab import app
from datalab.env import execenv
from datalab.utils.qthelpers import datalab_app_context


def test_launcher1(screenshots: bool = False):
    """Testing DataLab app launcher"""
    with datalab_app_context(exec_loop=not screenshots):
        execenv.print("Opening DataLab with no argument")
        win = app.create()
        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_app_at_startup")
            win.close()


if __name__ == "__main__":
    test_launcher1()
