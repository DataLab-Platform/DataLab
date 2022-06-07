# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Basic application launcher test 1

Running application a few times in a row with different entry parameters.
"""

from codraft import app
from codraft.utils.env import execenv
from codraft.utils.qthelpers import qt_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Testing CodraFT app launcher"""
    with qt_app_context(exec_loop=True):
        execenv.print("Opening CodraFT with no argument")
        app.create()


if __name__ == "__main__":
    test()
