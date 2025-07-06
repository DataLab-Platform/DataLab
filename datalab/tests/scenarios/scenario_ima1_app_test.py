# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image processing test scenario
------------------------------

Testing all the image processing features, without process isolation.
"""

# pylint: disable=duplicate-code
# guitest: show

from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import cdltest_app_context
from datalab.tests.scenarios import common


def test_scenario_image() -> None:
    """Run image unit test scenario 1 (no process isolation)"""
    assert Conf.main.process_isolation_enabled.get(), (
        "Process isolation must be enabled"
    )
    with cdltest_app_context(save=True) as win:
        with win.context_no_refresh():
            execenv.print("Testing image features without process isolation...")
            win.set_process_isolation_enabled(False)
            common.run_image_computations(win)
        win.imagepanel.remove_all_objects()
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_image()
