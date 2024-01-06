# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal processing test scenario
-------------------------------

Testing all the signal processing features, without process isolation.
"""

# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from cdl.config import Conf
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.scenarios import common


def test_scenario_signal1() -> None:
    """Run signal unit test scenario 1 (no process isolation)"""
    assert (
        Conf.main.process_isolation_enabled.get()
    ), "Process isolation must be enabled"
    with cdltest_app_context(save=True) as win:
        win.toggle_auto_refresh(False)
        execenv.print("Testing signal features (process isolation: off)...")
        win.set_process_isolation_enabled(False)
        common.run_signal_computations(win, all_types=True)
        win.toggle_auto_refresh(True)  # Just to test the auto-refresh feature
        win.signalpanel.remove_all_objects()
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_signal1()
