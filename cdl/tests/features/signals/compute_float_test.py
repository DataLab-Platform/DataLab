# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for some compute functions returning float values (enob and bandwidth).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from typing import Callable

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context

from cdl.algorithms.signal import bandwidth, enob
from cdl.env import execenv
from cdl.obj import SignalTypes, create_signal_from_param, new_signal_param


@pytest.mark.parametrize("func", (bandwidth, enob))
def test_float_result(func: Callable[[np.ndarray, np.ndarray], float]):
    """Signal bandwidth test"""
    with qt_app_context():
        newparam = new_signal_param(stype=SignalTypes.COSINUS, size=10000)
        s1 = create_signal_from_param(newparam)
        x, y = s1.xydata
        res = func(x, y)
        assert isinstance(res, float)
        execenv.print(f"{func.__name__}={res}", end=" ")
        execenv.print("OK")


if __name__ == "__main__":
    test_float_result(bandwidth)
    test_float_result(enob)
