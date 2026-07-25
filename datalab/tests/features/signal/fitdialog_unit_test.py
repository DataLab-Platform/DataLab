# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog test

Testing fit dialogs: Gaussian, Lorentzian, Voigt, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context
from sigima.objects import NormalDistribution1DParam
from sigima.tests.data import create_noisy_signal, get_test_signal
from sigima.tools.signal import pulse
from sigima.tools.signal.peakdetection import peak_indices

from datalab.env import execenv
from datalab.tests import helpers
from datalab.widgets import fitdialog as fdlg


def check_peak_fit_output(output):
    """Check versioned interactive peak-fit metadata."""
    assert output is not None
    _y_fitted, _params, fit_params = output
    assert fit_params["fit_params_version"] == 2
    assert fit_params["peak_parameterization"] == "height"
    assert fit_params["interactive"] is True


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


def test_peak_fit_metadata(monkeypatch):
    """Peak fit dialogs return canonical metadata when accepted."""

    def accept_initial_values(_x, _y, _fitfunc, fitparams, **_kwargs):
        return [param.value for param in fitparams]

    monkeypatch.setattr(fdlg, "guifit", accept_initial_values)
    single_peak = get_test_signal("gaussian_fit.txt")
    multi_peak = get_test_signal("paracetamol.txt")
    peakidx = peak_indices(multi_peak.y)

    outputs = (
        fdlg.gaussian_fit(single_peak.x, single_peak.y),
        fdlg.lorentzian_fit(single_peak.x, single_peak.y),
        fdlg.voigt_fit(single_peak.x, single_peak.y),
        fdlg.multigaussian_fit(multi_peak.x, multi_peak.y, peakidx),
        fdlg.multilorentzian_fit(multi_peak.x, multi_peak.y, peakidx),
    )
    for output in outputs:
        check_peak_fit_output(output)


@pytest.mark.parametrize(
    ("dialog", "model"),
    [
        (fdlg.gaussian_fit, pulse.GaussianModel),
        (fdlg.lorentzian_fit, pulse.LorentzianModel),
        (fdlg.voigt_fit, pulse.VoigtModel),
    ],
)
def test_peak_fit_dialog_supports_negative_amplitude(monkeypatch, dialog, model):
    """Interactive peak controls expose and preserve signed amplitudes."""
    captured_amplitudes = []

    def accept_initial_values(_x, _y, _fitfunc, fitparams, **_kwargs):
        captured_amplitudes.append(fitparams[0])
        return [param.value for param in fitparams]

    monkeypatch.setattr(fdlg, "guifit", accept_initial_values)
    x = np.linspace(-10.0, 10.0, 400)
    y = model.evaluate(x, -3.0, 1.5, 0.75, 2.0)

    output = dialog(x, y)

    assert output is not None
    _y_fitted, _params, fit_params = output
    assert fit_params["amplitude"] < 0.0
    amplitude_param = captured_amplitudes[0]
    assert amplitude_param.min < 0.0 < amplitude_param.max


if __name__ == "__main__":
    test_fit_dialog()
