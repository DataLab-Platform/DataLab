# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Module for taking DataLab screenshots
"""

from datalab import config
from datalab.tests.features.applauncher import launcher1_app_test
from datalab.tests.features.utilities import settings_unit_test
from datalab.tests.scenarios import beautiful_app

if __name__ == "__main__":
    print("Updating screenshots...", end=" ")
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    launcher1_app_test.test_launcher1(screenshots=True)
    config.reset()
    beautiful_app.run_beautiful_scenario(screenshots=True)
    settings_unit_test.capture_settings_screenshots()
    print("done.")
