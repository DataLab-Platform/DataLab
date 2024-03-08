# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Tour test
"""

# guitest: show

from cdl.core.gui.tour import start
from cdl.tests import cdltest_app_context


def test_tour() -> None:
    """
    Test the tour of DataLab features.
    """
    with cdltest_app_context() as win:
        start(win)


if __name__ == "__main__":
    test_tour()
