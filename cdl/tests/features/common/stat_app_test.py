# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Statistics test

Testing the following:
  - Create a signal
  - Compute statistics on signal and show results
  - Create an image
  - Compute statistics on image and show results
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.tests import cdltest_app_context, take_plotwidget_screenshot
from cdl.tests.features.common.stat_unit_test import (
    create_reference_image,
    create_reference_signal,
)


def test_stat_app():
    """Run statistics application test scenario"""
    with cdltest_app_context() as win:
        # === Signal statistics test ===
        panel = win.signalpanel
        panel.add_object(create_reference_signal())
        take_plotwidget_screenshot(panel, "stat_test")
        panel.processor.run_feature("stats")
        # === Image statistics test ===
        panel = win.imagepanel
        panel.add_object(create_reference_image())
        take_plotwidget_screenshot(panel, "stat_test")
        panel.processor.run_feature("stats")


if __name__ == "__main__":
    test_stat_app()
