# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests for full width computing features
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import pytest
from guidata.qthelpers import qt_app_context
from plotpy.builder import make

import cdl.tests.data as cdltd
import sigima_.computation.signal as sigima_signal
import sigima_.obj
import sigima_.param
import sigima_.tests.helpers
from cdl.adapters_plotpy.factories import create_adapter_from_object
from cdl.env import execenv
from cdl.utils.vistools import view_curve_items


def __test_fwhm_interactive(obj: sigima_.obj.SignalObj, method: str) -> None:
    """Interactive test for the full width at half maximum computation."""
    param = sigima_.param.FWHMParam.create(method=method)
    df = sigima_signal.fwhm(obj, param).to_dataframe()
    view_curve_items(
        [
            create_adapter_from_object(obj).make_item(),
            make.annotated_segment(df.x0[0], df.y0[0], df.x1[0], df.y1[0]),
        ],
        title=f"FWHM [{method}]",
    )


def test_signal_fwhm_interactive() -> None:
    """FWHM interactive test."""
    with qt_app_context():
        execenv.print("Computing FWHM of a multi-peak signal:")
        obj1 = cdltd.create_paracetamol_signal()
        obj2 = cdltd.create_noisy_signal(cdltd.GaussianNoiseParam.create(sigma=0.05))
        for method, _mname in sigima_.param.FWHMParam.methods:
            execenv.print(f"  Method: {method}")
            for obj in (obj1, obj2):
                if method == "zero-crossing":
                    # Check that a warning is raised when using the zero-crossing method
                    with pytest.warns(UserWarning):
                        __test_fwhm_interactive(obj, method)
                else:
                    __test_fwhm_interactive(obj, method)


@pytest.mark.validation
def test_signal_fwhm() -> None:
    """Validation test for the full width at half maximum computation."""
    obj = cdltd.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    for method, exp in (
        ("gauss", 2.40323),
        ("lorentz", 2.78072),
        ("voigt", 2.56591),
        ("zero-crossing", real_fwhm),
    ):
        param = sigima_.param.FWHMParam.create(method=method)
        df = sigima_signal.fwhm(obj, param).to_dataframe()
        sigima_.tests.helpers.check_scalar_result(
            f"FWHM[{method}]", df.L[0], exp, rtol=0.05
        )
    obj = cdltd.create_paracetamol_signal()
    with pytest.warns(UserWarning):
        sigima_signal.fwhm(obj, sigima_.param.FWHMParam.create(method="zero-crossing"))


@pytest.mark.validation
def test_signal_fw1e2() -> None:
    """Validation test for the full width at 1/e^2 maximum computation."""
    obj = cdltd.get_test_signal("fw1e2.txt")
    exp = 4.06  # Manual validation
    df = sigima_signal.fw1e2(obj).to_dataframe()
    sigima_.tests.helpers.check_scalar_result("FW1E2", df.L[0], exp, rtol=0.005)


@pytest.mark.validation
def test_signal_full_width_at_y() -> None:
    """Validation test for the full width at y computation."""
    obj = cdltd.get_test_signal("fwhm.txt")
    real_fwhm = 2.675  # Manual validation
    param = sigima_.param.OrdinateParam.create(y=0.5)
    df = sigima_signal.full_width_at_y(obj, param).to_dataframe()
    sigima_.tests.helpers.check_scalar_result("∆X", df.L[0], real_fwhm, rtol=0.05)


if __name__ == "__main__":
    test_signal_fwhm_interactive()
    test_signal_fwhm()
    test_signal_fw1e2()
    test_signal_full_width_at_y()
