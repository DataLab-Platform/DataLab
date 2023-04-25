# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

"""
Basic application launcher test 1

Running application a few times in a row with different entry parameters.
"""

from cdl import app
from cdl.env import execenv
from cdl.utils.qthelpers import qt_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Testing DataLab app launcher"""
    with qt_app_context(exec_loop=True):
        execenv.print("Opening DataLab with no argument")
        app.create()


if __name__ == "__main__":
    test()
