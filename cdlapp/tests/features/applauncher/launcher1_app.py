# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Basic application launcher test 1

Running application a few times in a row with different entry parameters.
"""

# guitest: show

from cdlapp import app
from cdlapp.env import execenv
from cdlapp.utils.qthelpers import cdl_app_context


def test():
    """Testing DataLab app launcher"""
    with cdl_app_context(exec_loop=True):
        execenv.print("Opening DataLab with no argument")
        app.create()


if __name__ == "__main__":
    test()
