# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal processing test scenario
-------------------------------

Testing all the signal processing features, without process isolation.
"""

# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import datalab_test_app_context
from datalab.tests.scenarios import common


def test_scenario_signal1() -> None:
    """Run signal unit test scenario 1 (no process isolation)"""
    assert Conf.main.process_isolation_enabled.get(), (
        "Process isolation must be enabled"
    )
    with datalab_test_app_context(save=True) as win:
        with win.context_no_refresh():
            execenv.print("Testing signal features (process isolation: off)...")
            win.set_process_isolation_enabled(False)
            common.run_signal_computations(win, all_types=True)
        win.signalpanel.remove_all_objects()
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_signal1()
