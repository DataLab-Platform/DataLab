# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for signal computing functions
-----------------------------------------

Features from the "Computing" menu are covered by this test.
The "Computing" menu contains functions to compute signal properties like
bandwidth, ENOB, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import cdl.algorithms.signal as alg
import cdl.core.computation.signal as cps
import cdl.param
from cdl.env import execenv
from cdl.obj import SignalTypes, create_signal_from_param, new_signal_param
from cdl.tests.data import check_scalar_result, get_test_fnames


@pytest.mark.parametrize(
    "func",
    (
        alg.bandwidth,
        alg.enob,
        alg.sinad,
        alg.thd,
        alg.sfdr,
        alg.snr,
        alg.sinus_frequency,
    ),
)
def test_func_for_errors(func: Callable[[np.ndarray, np.ndarray], float]) -> None:
    """Generic test for functions returning a float result.
    This test only checks if the function runs without errors.
    The result is not checked."""
    newparam = new_signal_param(stype=SignalTypes.COSINUS, size=200)
    s1 = create_signal_from_param(newparam)
    x, y = s1.xydata
    res = func(x, y)
    assert isinstance(res, float)
    execenv.print(f"{func.__name__}={res}", end=" ")
    execenv.print("OK")


@pytest.mark.validation
def test_dynamic_parameters() -> None:
    """Validation test for dynamic parameters computation."""
    obj = cdl.obj.read_signal(get_test_fnames("dynamic_parameters.txt")[0])
    param = cdl.param.DynamicParam.create(full_scale=1.0)
    df = cps.compute_dynamic_parameters(obj, param).to_dataframe()
    check_scalar_result("ENOB", df.ENOB[0], 5.1, rtol=0.001)
    check_scalar_result("SINAD", df.SINAD[0], 32.49, rtol=0.001)
    check_scalar_result("THD", df.THD[0], -30.18, rtol=0.001)
    check_scalar_result("SFDR", df.SFDR[0], 34.03, rtol=0.001)
    check_scalar_result("f", df.f[0], 49998377.464, rtol=0.001)
    check_scalar_result("SNR", df.SNR[0], 101.52, rtol=0.001)


if __name__ == "__main__":
    test_dynamic_parameters()
    test_func_for_errors(alg.bandwidth)
    test_func_for_errors(alg.enob)
