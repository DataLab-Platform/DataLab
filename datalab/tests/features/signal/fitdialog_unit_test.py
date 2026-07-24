# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog test

Testing fit dialogs: Gaussian, Lorentzian, Voigt, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from guidata.qthelpers import qt_app_context
from sigima.objects import NormalDistribution1DParam
from sigima.tests.data import create_noisy_signal, get_test_signal
from sigima.tools.signal import pulse
from sigima.tools.signal.peakdetection import peak_indices

from datalab.env import execenv
from datalab.tests import helpers
from datalab.widgets import fitdialog as fdlg


def test_fit_dialog():
    """Test function"""
    with qt_app_context():
        # Multi-gaussian curve fitting test
        s1 = get_test_signal("paracetamol.txt")
        peakidx = peak_indices(s1.y)
        s2 = create_noisy_signal(NormalDistribution1DParam.create(sigma=5.0))
        s3 = get_test_signal("gaussian_fit.txt")
        s4 = get_test_signal("piecewiseexponential_fit.txt")

        ep = execenv.print
        tn = helpers.get_default_test_name

        ep(fdlg.polynomial_fit(s2.x, s2.y, 4, name=tn("00")))
        ep(fdlg.linear_fit(s2.x, s2.y, name=tn("01")))
        ep(fdlg.gaussian_fit(s3.x, s3.y, name=tn("02")))
        ep(fdlg.lorentzian_fit(s3.x, s3.y, name=tn("03")))
        ep(fdlg.multigaussian_fit(s1.x, s1.y, peakidx, name=tn("04")))
        ep(fdlg.multilorentzian_fit(s1.x, s1.y, peakidx, name=tn("05")))
        ep(fdlg.voigt_fit(s3.x, s3.y, name=tn("06")))
        ep(fdlg.exponential_fit(s2.x, s2.y, name=tn("07")))
        ep(fdlg.sinusoidal_fit(s2.x, s2.y, name=tn("08")))
        ep(fdlg.cdf_fit(s2.x, s2.y, name=tn("09")))
        ep(fdlg.planckian_fit(s3.x, s3.y, name=tn("10")))
        ep(fdlg.twohalfgaussian_fit(s3.x, s3.y, name=tn("11")))
        ep(fdlg.piecewiseexponential_fit(s4.x, s4.y, name=tn("12")))


def test_evaluate_fit_matches_models():
    """Test the GUI-free deterministic fit evaluator (no dialog)."""
    x = np.linspace(-5, 5, 50)

    # Polynomial (degree 2)
    poly_values = [2.0, -1.0, 3.0]
    assert np.allclose(
        fdlg.evaluate_fit("polynomial", x, poly_values),
        np.polyval(poly_values, x),
    )

    # Gaussian: [amp, sigma, x0, y0]
    gauss_values = [5.0, 1.5, 0.3, 0.2]
    assert np.allclose(
        fdlg.evaluate_fit("gaussian", x, gauss_values),
        pulse.GaussianModel.func(x, *gauss_values),
    )

    # Multi-Gaussian: [A1, σ1, A2, σ2, y0] + fixed peak abscissas
    multi_values = [1.0, 0.5, 2.0, 0.4, 0.1]
    a_x0 = [-1.0, 1.5]
    assert np.allclose(
        fdlg.evaluate_fit("multigaussian", x, multi_values, extra={"a_x0": a_x0}),
        fdlg.multigaussian(x, *multi_values, a_x0=np.array(a_x0)),
    )

    # Canonical mapping helper
    assert fdlg.fit_type_from_dlgfunc_name("gaussian_fit") == "gaussian"
    assert fdlg.fit_type_from_dlgfunc_name("unknown") is None

    # Unknown fit type raises
    try:
        fdlg.evaluate_fit("nope", x, [])
    except ValueError:
        pass
    else:
        raise AssertionError("evaluate_fit should raise ValueError for unknown type")


if __name__ == "__main__":
    test_fit_dialog()
    test_evaluate_fit_matches_models()
