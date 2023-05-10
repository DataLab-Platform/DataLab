# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Module for taking DataLab screenshots
"""

from cdl import config
from cdl.app import create
from cdl.tests.data import create_test_image1, create_test_signal1
from cdl.utils.qthelpers import qt_app_context


def take_menu_screenshots():
    """Run the DataLab application and take screenshots"""
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
