# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Tour test
"""

# guitest: show

from datalab.gui.tour import start
from datalab.tests import cdltest_app_context


def test_tour() -> None:
    """
    Test the tour of DataLab features.
    """
    with cdltest_app_context() as win:
        start(win)


if __name__ == "__main__":
    test_tour()
