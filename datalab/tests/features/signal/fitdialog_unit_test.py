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
from sigima.tools.signal import fitting, pulse
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


NON_PEAK_FIT_CASES = (
    ("linear", "noisy", fdlg.linear_fit),
    ("polynomial", "noisy", lambda x, y: fdlg.polynomial_fit(x, y, 4)),
    ("exponential", "noisy", fdlg.exponential_fit),
    ("sinusoidal", "noisy", fdlg.sinusoidal_fit),
    ("cdf", "noisy", fdlg.cdf_fit),
    ("planckian", "gaussian_fit.txt", fdlg.planckian_fit),
    ("twohalfgaussian", "gaussian_fit.txt", fdlg.twohalfgaussian_fit),
    (
        "doubleexponential",
        "piecewiseexponential_fit.txt",
        fdlg.piecewiseexponential_fit,
    ),
)


@pytest.mark.parametrize(("fit_type", "data", "call_dialog"), NON_PEAK_FIT_CASES)
def test_non_peak_fit_metadata(monkeypatch, fit_type, data, call_dialog):
    """Non-peak fit dialogs return evaluable canonical metadata.

    The decisive check is the round-trip: re-evaluating the stored parameters
    with Sigima must reproduce the curve computed by the dialog. It catches any
    parameter name, ordering or unit mismatch between the two layers.
    """

    def accept_initial_values(_x, _y, _fitfunc, fitparams, **_kwargs):
        return [param.value for param in fitparams]

    monkeypatch.setattr(fdlg, "guifit", accept_initial_values)
    if data == "noisy":
        signal = create_noisy_signal(NormalDistribution1DParam.create(sigma=5.0))
    else:
        signal = get_test_signal(data)

    output = call_dialog(signal.x, signal.y)

    assert output is not None
    y_fitted, _params, fit_params = output
    assert fit_params["fit_type"] == fit_type
    assert fit_params["interactive"] is True
    fitting.validate_fit_params(fit_params)
    np.testing.assert_allclose(
        fitting.evaluate_fit(signal.x, **fit_params), y_fitted, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize(
    ("dialog", "fit_type"),
    [
        (fdlg.multigaussian_fit, "multigaussian"),
        (fdlg.multilorentzian_fit, "multilorentzian"),
    ],
)
def test_multi_peak_fit_metadata_preserves_fixed_centers(monkeypatch, dialog, fit_type):
    """Multi-peak metadata preserves centers without adding fit controls."""

    def accept_initial_values(_x, _y, _fitfunc, fitparams, **_kwargs):
        values = [param.value for param in fitparams]
        values[1] = -abs(values[1])
        return values

    monkeypatch.setattr(fdlg, "guifit", accept_initial_values)
    signal = get_test_signal("paracetamol.txt")
    peakidx = peak_indices(signal.y)

    output = dialog(signal.x, signal.y, peakidx)

    assert output is not None
    y_fitted, params, fit_params = output
    assert len(params) == 2 * len(peakidx) + 1
    assert fit_params["fit_type"] == fit_type
    for index, peak_index in enumerate(peakidx, start=1):
        assert fit_params[f"x0_{index}"] == pytest.approx(signal.x[peak_index])
        assert fit_params[f"sigma_{index}"] > 0.0
    np.testing.assert_allclose(fitting.evaluate_fit(signal.x, **fit_params), y_fitted)


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


def __capture_fit_params(monkeypatch) -> list:
    """Patch `guifit` so it accepts the initial values and records the controls."""
    captured: list = []

    def accept_initial_values(_x, _y, _fitfunc, fitparams, **_kwargs):
        captured.extend(fitparams)
        return [param.value for param in fitparams]

    monkeypatch.setattr(fdlg, "guifit", accept_initial_values)
    return captured


@pytest.mark.parametrize(
    ("dialog", "make_data", "true_values"),
    [
        # A decaying exponential: the B slider used to be restricted to
        # positive values, so this optimum was unreachable.
        # Parameter order: (a, b, y0)
        (
            fdlg.exponential_fit,
            lambda x: 3.0 * np.exp(-0.8 * x) + 1.0,
            {1: -0.8},
        ),
        # A descending transition: the amplitude slider used to start at 0.
        # Parameter order: (amplitude, mu, sigma, baseline)
        (
            fdlg.cdf_fit,
            lambda x: (
                -2.0 * fitting.CDFFitComputer.evaluate(x, 1.0, 5.0, 1.0, 0.0) + 4.0
            ),
            {0: -2.0},
        ),
        # A decay-then-rise shape: the rate sliders used to hard-code the
        # opposite (rise-then-decay) sign convention.
        # Parameter order: (x_center, a_left, b_left, a_right, b_right, y0)
        (
            fdlg.piecewiseexponential_fit,
            lambda x: np.where(x < 5.0, np.exp(-(x - 5.0)), np.exp(x - 5.0)) + 0.5,
            {2: -1.0, 4: 1.0},
        ),
    ],
)
def test_fit_dialog_bounds_contain_the_optimum(
    monkeypatch, dialog, make_data, true_values
):
    """Interactive fit sliders must be able to reach the true parameters.

    Several dialogs used one-sided bounds that excluded a whole family of
    shapes, or bounds derived from the magnitude of the initial guess, which
    could invert into an empty interval.
    """
    captured = __capture_fit_params(monkeypatch)
    x = np.linspace(0.0, 10.0, 400)

    assert dialog(x, make_data(x)) is not None

    for param in captured:
        assert param.min < param.max, f"{param.name}: inverted bounds"
        assert param.min <= param.value <= param.max, (
            f"{param.name}: initial value outside its bounds"
        )

    for index, true_value in true_values.items():
        param = captured[index]
        assert param.min <= true_value <= param.max, (
            f"{param.name}: true value {true_value} is outside the slider range "
            f"[{param.min}, {param.max}]"
        )


if __name__ == "__main__":
    test_fit_dialog()
