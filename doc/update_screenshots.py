# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Module for taking CodraFT screenshots
"""

from codraft import config
from codraft.app import create
from codraft.tests.data import create_test_image1, create_test_signal1
from codraft.utils.qthelpers import qt_app_context


def take_menu_screenshots():
    """Run the CodraFT application and take screenshots"""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    sig1 = create_test_signal1()
    ima1 = create_test_image1()
    objects = (sig1, ima1)
    with qt_app_context():
        window = create(splash=False, objects=objects, size=(800, 300))
        window.take_menu_screenshots()
        window.take_screenshot("i_simple_example")
        window.switch_to_signal_panel()
        window.take_screenshot("s_simple_example")
        window.set_modified(False)
        window.close()


if __name__ == "__main__":
    take_menu_screenshots()
