# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal processing test scenario
-------------------------------

Testing all the signal processing features, with process isolation.
"""

# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from cdl.config import Conf
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from cdl.tests.scenarios import common


def test_scenario_signal2() -> None:
    """Run signal unit test scenario 2 (process isolation)"""
    assert (
        Conf.main.process_isolation_enabled.get()
    ), "Process isolation must be enabled"
    with cdltest_app_context(save=True) as win:
        execenv.print("Testing signal features (process isolation: on)...")
        common.run_signal_computations(win, all_types=False)
        oids = win.signalpanel.objmodel.get_object_ids()
        win.signalpanel.open_separate_view(oids[:3])
        execenv.print("==> OK")


if __name__ == "__main__":
    test_scenario_signal2()
