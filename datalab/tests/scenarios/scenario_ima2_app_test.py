# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Image processing test scenario
------------------------------

Testing all the image processing features, with process isolation.
"""

# pylint: disable=duplicate-code
# guitest: show

from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import datalab_test_app_context
from datalab.tests.scenarios import common


def test_scenario_image2() -> None:
    """Run image unit test scenario 2 (process isolation)"""
    assert Conf.main.process_isolation_enabled.get(), (
        "Process isolation must be enabled"
    )

    with datalab_test_app_context(save=True) as win:
        execenv.print("Testing image features *with* process isolation...")
        common.run_image_computations(win, all_types=False)
        oids = win.imagepanel.objmodel.get_object_ids()
        win.imagepanel.open_separate_view(oids[:4])
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_image2()
