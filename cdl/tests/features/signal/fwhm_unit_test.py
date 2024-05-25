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

import pytest
from guidata.qthelpers import qt_app_context
from plotpy.builder import make

import cdl.core.computation.signal as cps
import cdl.obj
import cdl.param
from cdl.env import execenv
from cdl.tests.data import (
    check_scalar_result,
    create_paracetamol_signal,
    get_test_signal,
)
from cdl.utils.vistools import view_curve_items


def __test_fwhm_interactive(obj: cdl.obj.SignalObj, method: str) -> None:
    """Interactive test for the full width at half maximum computation."""
    param = cdl.param.FWHMParam.create(method=method)
    df = cps.compute_fwhm(obj, param).to_dataframe()
    view_curve_items(
        [
            obj.make_item(),
            make.annotated_segment(df.x0[0], df.y0[0], df.x1[0], df.y1[0]),
        ],
        title=f"FWHM [{method}]",
    )


def test_signal_fwhm_interactive() -> None:
    """FWHM interactive test."""
    with qt_app_context():
        execenv.print("Computing FWHM of a multi-peak signal...", end=" ")
        obj = create_paracetamol_signal()
        __test_fwhm_interactive(obj, "zero-crossing")
        execenv.print("OK")


@pytest.mark.validation
def test_signal_fwhm() -> None:
    """Validation test for the full width at half maximum computation."""
    obj = get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    for method, exp in (
        ("gauss", 2.40323),
        ("lorentz", 2.78072),
        ("voigt", 2.56591),
        ("zero-crossing", real_fwhm),
    ):
        param = cdl.param.FWHMParam.create(method=method)
        df = cps.compute_fwhm(obj, param).to_dataframe()
        check_scalar_result(f"FWHM[{method}]", df.L[0], exp, rtol=0.05)


@pytest.mark.validation
def test_signal_fw1e2() -> None:
    """Validation test for the full width at 1/e^2 maximum computation."""
    obj = get_test_signal("fw1e2.txt")
    exp = 4.06  # Manual validation
    df = cps.compute_fw1e2(obj).to_dataframe()
    check_scalar_result("FW1E2", df.L[0], exp, rtol=0.005)


if __name__ == "__main__":
    test_signal_fwhm_interactive()
    test_signal_fwhm()
    test_signal_fw1e2()
