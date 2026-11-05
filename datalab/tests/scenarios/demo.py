# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Demo

Plays the demo of the application (:mod:`datalab.gui.demo`) from the test launcher.
"""

# guitest: show,skip

from __future__ import annotations

from datalab.config import reset
from datalab.gui.demo import play_demo
from datalab.tests import datalab_test_app_context


def run() -> None:
    """Run demo"""
    reset()  # Reset configuration (remove configuration file and initialize it)
    with datalab_test_app_context(console=False) as win:
        play_demo(win)


if __name__ == "__main__":
    run()
