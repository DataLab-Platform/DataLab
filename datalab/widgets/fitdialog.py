# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog widgets"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog
from plotpy.plot import PlotOptions
from plotpy.widgets.fit import FitDialog, FitParam
from scipy.special import erf  # pylint: disable=no-name-in-module
from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal import fitting, fourier, pulse

from datalab.config import Conf, _

DEFAULT_FORMAT = "%g"


def create_interactive_fit_params(fit_type, values, y, y_fitted):
    """Create canonical metadata for a fit committed from a dialog."""
    residual_rms = np.sqrt(np.mean((y - y_fitted) ** 2))
    return fitting.create_fit_params(
        fit_type, values, residual_rms=residual_rms, interactive=True
    )


def guifit(
    x,
    y,
    fitfunc,
    fitparams,
    fitargs=None,
    fitkwargs=None,
    wintitle=None,
    title=None,
    xlabel=None,
    ylabel=None,
    param_cols=1,
    auto_fit=True,
    winsize=None,
    winpos=None,
    parent=None,
    name=None,
):  # pylint: disable=too-many-positional-arguments
    """GUI-based curve fitting tool"""
    win = FitDialog(
        edit=True,
        title=wintitle,
        icon=None,
        toolbar=True,
        options=PlotOptions(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            curve_antialiasing=True,
            show_axes_tab=False,
            autoscale_margin_percent=Conf.view.sig_autoscale_margin_percent.get(),
        ),
        parent=parent,
        param_cols=param_cols,
        auto_fit=auto_fit,
    )
    win.setObjectName(name)
    win.set_data(x, y, fitfunc, fitparams, fitargs, fitkwargs)
    try:
        win.autofit()  # TODO: [P3] make this optional
    except ValueError:
        pass
    if parent is None:
        win.setWindowIcon(get_icon("DataLab.svg"))
    if winsize is not None:
        win.resize(*winsize)
    if winpos is not None:
        win.move(*winpos)
    win.get_plot().do_autoscale()
    if exec_dialog(win):
        return win.get_values()
    return None


# --- Polynomial fitting curve -------------------------------------------------
def polynomial_fit(x, y, degree, parent=None, name=None, fit_type="polynomial"):
    """Compute polynomial fit

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    computer = fitting.PolynomialFitComputer(x, y, degree)
    ivals = np.polyfit(x, y, degree)

    params = []
    for index in range(degree + 1):
        val = ivals[index]
        vmax = max(1.0, np.abs(val))
        param = FitParam(
            f"c{(len(ivals) - index - 1):d}",
            val,
            -2 * vmax,
            2 * vmax,
            format=DEFAULT_FORMAT,
        )
        params.append(param)

    def fitfunc(x, params):
        return np.polyval(params, x)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Polymomial fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        # Both `np.polyfit` and `PolynomialFitComputer` order coefficients from
        # the highest degree to the lowest, so a plain zip is correct here.
        fit_values = dict(zip(computer.get_params_names(), values))
        fit_params = create_interactive_fit_params(fit_type, fit_values, y, y_fitted)
        return y_fitted, params, fit_params


def linear_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute linear fit using polynomialfit.

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary
    """
    # A first-degree polynomial and a linear fit share the same `(a, b)` parameter
    # names, so only the stored fit type has to be overridden.
    return polynomial_fit(x, y, 1, parent=parent, name=name, fit_type="linear")


# --- Gaussian fitting curve ---------------------------------------------------
def gaussian_fit(x, y, parent=None, name=None):
    """Compute Gaussian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    # Get initial parameter estimates from Sigima GaussianFitComputer
    computer = fitting.GaussianFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    amplitude_guess = initial_params["amplitude"]
    sigma_guess = initial_params["sigma"]
    mu_guess = initial_params["x0"]
    b_guess = initial_params["y0"]

    dy = np.max(y) - np.min(y)
    max_amplitude = max(2.0 * dy, 2.0 * abs(amplitude_guess))
    amplitude = FitParam(
        _("Amplitude"),
        amplitude_guess,
        -max_amplitude,
        max_amplitude,
        format=DEFAULT_FORMAT,
    )
    b = FitParam(
        _("Base line"), b_guess, np.min(y) - 0.1 * dy, np.max(y), format=DEFAULT_FORMAT
    )
    sigma = FitParam(
        _("Std-dev") + " (σ)",
        sigma_guess,
        sigma_guess * 0.1,
        sigma_guess * 10,
        format=DEFAULT_FORMAT,
    )
    mu = FitParam(
        _("Mean") + " (μ)", mu_guess, np.min(x), np.max(x), format=DEFAULT_FORMAT
    )

    params = [amplitude, sigma, mu, b]

    def fitfunc(x, params):
        return pulse.GaussianModel.evaluate(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Gaussian fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "gaussian", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Lorentzian fitting curve -------------------------------------------------
def lorentzian_fit(x, y, parent=None, name=None):
    """Compute Lorentzian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    # Get initial parameter estimates from Sigima LorentzianFitComputer
    computer = fitting.LorentzianFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    amplitude_guess = initial_params["amplitude"]
    sigma_guess = initial_params["sigma"]
    mu_guess = initial_params["x0"]
    b_guess = initial_params["y0"]

    # Create parameter bounds
    dy = np.max(y) - np.min(y)

    max_amplitude = max(2.0 * dy, 2.0 * abs(amplitude_guess))
    amplitude = FitParam(
        _("Amplitude"),
        amplitude_guess,
        -max_amplitude,
        max_amplitude,
        format=DEFAULT_FORMAT,
    )
    b = FitParam(
        _("Base line"), b_guess, np.min(y) - 0.1 * dy, np.max(y), format=DEFAULT_FORMAT
    )
    sigma = FitParam(
        _("Std-dev") + " (σ)",
        sigma_guess,
        sigma_guess * 0.1,
        sigma_guess * 10,
        format=DEFAULT_FORMAT,
    )
    mu = FitParam(
        _("Mean") + " (μ)", mu_guess, np.min(x), np.max(x), format=DEFAULT_FORMAT
    )

    params = [amplitude, sigma, mu, b]

    def fitfunc(x, params):
        return pulse.LorentzianModel.evaluate(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Lorentzian fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "lorentzian", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Voigt fitting curve ------------------------------------------------------
def voigt_fit(x, y, parent=None, name=None):
    """Compute Voigt fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    # Get initial parameter estimates from Sigima VoigtFitComputer
    computer = fitting.VoigtFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    amplitude_guess = initial_params["amplitude"]
    sigma_guess = initial_params["sigma"]
    mu_guess = initial_params["x0"]
    b_guess = initial_params["y0"]

    # Create parameter bounds
    dy = np.max(y) - np.min(y)

    max_amplitude = max(2.0 * dy, 2.0 * abs(amplitude_guess))
    amplitude = FitParam(
        _("Amplitude"),
        amplitude_guess,
        -max_amplitude,
        max_amplitude,
        format=DEFAULT_FORMAT,
    )
    b = FitParam(
        _("Base line"), b_guess, np.min(y) - 0.1 * dy, np.max(y), format=DEFAULT_FORMAT
    )
    sigma = FitParam(
        _("Std-dev") + " (σ)",
        sigma_guess,
        sigma_guess * 0.1,
        sigma_guess * 10,
        format=DEFAULT_FORMAT,
    )
    mu = FitParam(
        _("Mean") + " (μ)", mu_guess, np.min(x), np.max(x), format=DEFAULT_FORMAT
    )

    params = [amplitude, sigma, mu, b]

    def fitfunc(x, params):
        return pulse.VoigtModel.evaluate(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Voigt fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "voigt", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Multi-Gaussian fitting curve ---------------------------------------------
def multigaussian(x, *values, **kwargs):
    """Return a 1-dimensional multi-Gaussian function."""
    amplitudes = values[0::2]
    a_sigma = values[1::2]
    y0 = values[-1]
    a_x0 = kwargs["a_x0"]
    y = np.zeros_like(x) + y0
    for amplitude, sigma, x0 in zip(amplitudes, a_sigma, a_x0):
        y += pulse.GaussianModel.evaluate(x, amplitude, sigma, x0, 0.0)
    return y


def multigaussian_fit(x, y, peak_indices, parent=None, name=None):
    """Compute Multi-Gaussian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    # Get initial parameter estimates from Sigima MultiGaussianFitComputer
    computer = fitting.MultiGaussianFitComputer(x, y, peak_indices)
    initial_params = computer.compute_initial_params()
    # Use Sigima parameters to populate DataLab params
    params = []
    for index, i0 in enumerate(peak_indices):
        stri = f"{index + 1:02d}"
        amplitude_key = f"amplitude_{index + 1}"
        sigma_key = f"sigma_{index + 1}"
        amplitude_value = initial_params.get(amplitude_key, y[i0] - np.min(y))
        sigma_val = (
            initial_params[sigma_key]
            if sigma_key in initial_params
            else (x.max() - x.min()) / 100
        )

        # Calculate bounds based on local data
        istart = 0
        iend = len(x) - 1
        if index > 0:
            istart = (peak_indices[index - 1] + i0) // 2
        if index < len(peak_indices) - 1:
            iend = (peak_indices[index + 1] + i0) // 2
        dx = 0.5 * (x[iend] - x[istart])
        dy = np.max(y[istart:iend]) - np.min(y[istart:iend])
        amplitude_range = max(dy * 2, abs(amplitude_value) * 2)

        params += [
            FitParam(
                ("A") + stri,
                amplitude_value,
                -amplitude_range,
                amplitude_range,
                format=DEFAULT_FORMAT,
            ),
            FitParam("σ" + stri, sigma_val, dx / 100, dx, format=DEFAULT_FORMAT),
        ]

    y0_val = initial_params.get("y0", np.min(y))
    params.append(
        FitParam(
            _("Y0"),
            y0_val,
            np.min(y) - 0.1 * (np.max(y) - np.min(y)),
            np.max(y),
            format=DEFAULT_FORMAT,
        )
    )

    kwargs = {"a_x0": x[peak_indices]}

    def fitfunc(xi, params):
        return multigaussian(xi, *params, **kwargs)

    param_cols = 1
    if len(params) > 8:
        param_cols = 4
    values = guifit(
        x,
        y,
        fitfunc,
        params,
        param_cols=param_cols,
        winsize=(900, 600),
        parent=parent,
        name=name,
        wintitle=_("Multi-Gaussian fit"),
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_values = {"y0": values[-1]}
        for index, (amplitude, sigma, x0) in enumerate(
            zip(values[0::2], values[1::2], kwargs["a_x0"]), start=1
        ):
            fit_values[f"amplitude_{index}"] = amplitude
            fit_values[f"sigma_{index}"] = abs(sigma)
            fit_values[f"x0_{index}"] = x0
        fit_params = create_interactive_fit_params(
            "multigaussian", fit_values, y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Multi-Lorentzian fitting curve -------------------------------------------
def multilorentzian(x, *values, **kwargs):
    """Return a 1-dimensional multi-Lorentzian function."""
    amplitudes = values[0::2]
    a_sigma = values[1::2]
    y0 = values[-1]
    a_x0 = kwargs["a_x0"]
    y = np.zeros_like(x) + y0
    for amplitude, sigma, x0 in zip(amplitudes, a_sigma, a_x0):
        y += pulse.LorentzianModel.evaluate(x, amplitude, sigma, x0, 0.0)
    return y


def multilorentzian_fit(
    x: np.ndarray, y: np.ndarray, peak_indices, parent=None, name=None
):
    """Compute Multi-Lorentzian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    # Get initial parameter estimates from Sigima MultiLorentzianFitComputer
    computer = fitting.MultiLorentzianFitComputer(x, y, peak_indices)
    initial_params = computer.compute_initial_params()
    # Use Sigima parameters to populate DataLab params
    params = []
    dy = np.max(y) - np.min(y)
    for index, i0 in enumerate(peak_indices):
        stri = f"{index + 1:02d}"
        amplitude_key = f"amplitude_{index + 1}"
        sigma_key = f"sigma_{index + 1}"
        amplitude_value = initial_params.get(amplitude_key, y[i0] - np.min(y))
        sigma_val = (
            initial_params[sigma_key]
            if sigma_key in initial_params
            else (x.max() - x.min()) / 100
        )

        params += [
            FitParam(
                ("A") + stri,
                amplitude_value,
                -max(abs(amplitude_value) * 2, dy * 2),
                max(abs(amplitude_value) * 2, dy * 2),
                format=DEFAULT_FORMAT,
            ),
            FitParam(
                "σ" + stri,
                sigma_val,
                sigma_val * 0.2,
                sigma_val * 10,
                format=DEFAULT_FORMAT,
            ),
        ]

    y0_val = initial_params.get("y0", np.min(y))
    params.append(
        FitParam(
            _("Y0"),
            y0_val,
            np.min(y) - 0.1 * (np.max(y) - np.min(y)),
            np.max(y),
            format=DEFAULT_FORMAT,
        )
    )

    kwargs = {"a_x0": x[peak_indices]}

    def fitfunc(xi, params):
        return multilorentzian(xi, *params, **kwargs)

    param_cols = 1
    if len(params) > 8:
        param_cols = 4
    values = guifit(
        x,
        y,
        fitfunc,
        params,
        param_cols=param_cols,
        winsize=(900, 600),
        parent=parent,
        name=name,
        wintitle=_("Multi-Lorentzian fit"),
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_values = {"y0": values[-1]}
        for index, (amplitude, sigma, x0) in enumerate(
            zip(values[0::2], values[1::2], kwargs["a_x0"]), start=1
        ):
            fit_values[f"amplitude_{index}"] = amplitude
            fit_values[f"sigma_{index}"] = abs(sigma)
            fit_values[f"x0_{index}"] = x0
        fit_params = create_interactive_fit_params(
            "multilorentzian", fit_values, y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Exponential fitting curve ------------------------------------------------


def exponential_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute exponential fit

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima ExponentialFitComputer
    computer = fitting.ExponentialFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    oa = initial_params["a"]
    ob = initial_params["b"]
    oc = initial_params["y0"]

    # Create parameter bounds
    moa, mob, moc = np.maximum(1, [abs(oa), abs(ob), abs(oc)])
    a_p = FitParam(
        _("A coefficient"), oa, -2 * moa, 2 * moa, logscale=True, format=DEFAULT_FORMAT
    )
    # B must be free to change sign: a positive-only range makes every decaying
    # exponential unreachable. Sigima uses (-10, 10) for the same parameter.
    mob = max(10.0, 2 * mob)
    b_p = FitParam(_("B coefficient"), ob, -mob, mob, format=DEFAULT_FORMAT)
    c_p = FitParam(_("y0 constant"), oc, -2 * moc, 2 * moc, format=DEFAULT_FORMAT)

    params = [a_p, b_p, c_p]

    def modelfunc(x, a, b, c):
        return a * np.exp(b * x) + c

    def fitfunc(x, params):
        return modelfunc(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Exponential fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "exponential", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Sinusoidal fitting curve ------------------------------------------------


@check_1d_arrays(x_evenly_spaced=True)
def dominant_frequency(x: np.ndarray, y: np.ndarray) -> np.floating:
    """Find the dominant frequency.

    Args:
        x: 1-D x values.
        y: 1-D y values.

    Returns:
        Dominant frequency.
    """
    f, spectrum = fourier.magnitude_spectrum(x, y)
    return np.abs(f[np.argmax(spectrum)])


def sinusoidal_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute sinusoidal fit

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima SinusoidalFitComputer
    computer = fitting.SinusoidalFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    guess_a = initial_params["amplitude"]
    guess_f = initial_params["frequency"]
    guess_ph = np.rad2deg(initial_params["phase"])  # Convert to degrees
    guess_c = initial_params["offset"]

    # Create parameter bounds
    abs_values = [abs(guess_a), abs(guess_f), abs(guess_ph), abs(guess_c)]
    moa, mof, _mop, moc = np.maximum(1, abs_values)
    a_p = FitParam(_("Amplitude"), guess_a, -2 * moa, 2 * moa, format=DEFAULT_FORMAT)
    f_p = FitParam(_("Frequency"), guess_f, 0, 2 * mof, format=DEFAULT_FORMAT)
    p_p = FitParam(_("Phase"), guess_ph, -360, 360, format=DEFAULT_FORMAT)
    c_p = FitParam(
        _("Continuous component"), guess_c, -2 * moc, 2 * moc, format=DEFAULT_FORMAT
    )

    params = [a_p, f_p, p_p, c_p]

    def modelfunc(x, a, f, p, c):
        return a * np.sin(2 * np.pi * f * x + np.deg2rad(p)) + c

    def fitfunc(x, params):
        return modelfunc(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Sinusoidal fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        # The phase is edited in degrees but stored in radians, as expected by
        # Sigima's sinusoidal model.
        amplitude, frequency, phase, offset = values
        fit_values = dict(
            zip(
                computer.get_params_names(),
                (amplitude, frequency, np.deg2rad(phase), offset),
            )
        )
        fit_params = create_interactive_fit_params(
            "sinusoidal", fit_values, y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Cumulative distribution function fitting curve -----------------------------------


def cdf_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute Cumulative Distribution Function (CDF) fit

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima CDFFitComputer
    computer = fitting.CDFFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    a_guess = initial_params["amplitude"]
    mu_guess = initial_params["mu"]
    sigma_guess = initial_params["sigma"]
    b_guess = initial_params["baseline"]

    # Create parameter bounds
    dy = np.max(y) - np.min(y)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    dx = x_max - x_min
    iamp = max(1.0, abs(a_guess))
    # Amplitude must be free to change sign, otherwise a descending transition
    # cannot be fitted at all.
    a = FitParam(
        _("Amplitude"), a_guess, -iamp * 2.0, iamp * 2.0, format=DEFAULT_FORMAT
    )
    b = FitParam(
        _("Base line"), b_guess, np.min(y) - 0.1 * dy, np.max(y), format=DEFAULT_FORMAT
    )
    # Bound sigma and mu to the abscissa range rather than to the magnitude of
    # the initial guess, which excluded negative and near-zero means.
    sigma = FitParam(
        _("Std-dev") + " (σ)",
        sigma_guess,
        dx * 0.001,
        dx,
        format=DEFAULT_FORMAT,
    )
    mu = FitParam(_("Mean") + " (μ)", mu_guess, x_min, x_max, format=DEFAULT_FORMAT)

    params = [a, mu, sigma, b]

    def modelfunc(x, a, mu, sigma, b):
        return a * erf((x - mu) / (sigma * np.sqrt(2))) + b

    def fitfunc(x, params):
        return modelfunc(x, *params)

    values = guifit(
        x,
        y,
        fitfunc,
        params,
        parent=parent,
        wintitle=_("CDF fit"),
        name=name,
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "cdf", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Planckian fitting curve --------------------------------------------------
def planckian_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute Planckian (blackbody radiation) fit

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima PlanckianFitComputer
    computer = fitting.PlanckianFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    amp_guess = initial_params["amp"]
    x0_guess = initial_params["x0"]
    sigma_guess = initial_params["sigma"]
    y0_guess = initial_params["y0"]

    # Create parameter bounds
    dy = np.max(y) - np.min(y)

    # Parameter bounds with appropriate ranges for Planckian fitting
    amp = FitParam(
        _("Amplitude"),
        amp_guess,
        amp_guess * 0.01,
        amp_guess * 100,
        format=DEFAULT_FORMAT,
    )
    x0 = FitParam(
        _("Scale factor"), x0_guess, np.min(x), np.max(x), format=DEFAULT_FORMAT
    )
    sigma = FitParam(_("Width factor"), sigma_guess, 0.1, 5.0, format=DEFAULT_FORMAT)
    y0 = FitParam(
        _("Base line"),
        y0_guess,
        y0_guess - 0.2 * dy,
        y0_guess + 0.2 * dy,
        format=DEFAULT_FORMAT,
    )

    params = [amp, x0, sigma, y0]

    def fitfunc(x, params: list[float]) -> np.ndarray:
        """Evaluate Planckian function with given parameters."""
        return fitting.PlanckianFitComputer.evaluate(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Planckian fit"), name=name
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "planckian", dict(zip(computer.get_params_names(), values)), y, y_fitted
        )
        return y_fitted, params, fit_params


# --- Two half-Gaussian fitting curve ------------------------------------------
def twohalfgaussian_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute two half-Gaussian fit for asymmetric peaks

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima TwoHalfGaussianFitComputer
    computer = fitting.TwoHalfGaussianFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    amp_left_guess = initial_params["amp_left"]
    amp_right_guess = initial_params["amp_right"]
    sigma_left_guess = initial_params["sigma_left"]
    sigma_right_guess = initial_params["sigma_right"]
    x0_guess = initial_params["x0"]
    y0_left_guess = initial_params["y0_left"]
    y0_right_guess = initial_params["y0_right"]

    # Create parameter bounds
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)

    # Parameter bounds with better ranges
    # New model signature: func(x, amp_left, amp_right, sigma_left,
    #                            sigma_right, x0, y0_left, y0_right)
    amp_left = FitParam(
        _("Left amplitude"), amp_left_guess, dy * 0.1, dy * 3, format=DEFAULT_FORMAT
    )
    amp_right = FitParam(
        _("Right amplitude"), amp_right_guess, dy * 0.1, dy * 3, format=DEFAULT_FORMAT
    )
    sigma_left = FitParam(
        _("Left width") + " (σL)",
        sigma_left_guess,
        dx * 0.001,  # Very small minimum
        dx * 0.5,  # Reasonable maximum
        format=DEFAULT_FORMAT,
    )
    sigma_right = FitParam(
        _("Right width") + " (σR)",
        sigma_right_guess,
        dx * 0.001,  # Very small minimum
        dx * 0.5,  # Reasonable maximum
        format=DEFAULT_FORMAT,
    )
    x0 = FitParam(
        _("Center") + " (x₀)", x0_guess, np.min(x), np.max(x), format=DEFAULT_FORMAT
    )
    y0_left = FitParam(
        _("Left baseline"),
        y0_left_guess,
        y0_left_guess - 0.2 * dy,
        y0_left_guess + 0.2 * dy,
        format=DEFAULT_FORMAT,
    )
    y0_right = FitParam(
        _("Right baseline"),
        y0_right_guess,
        y0_right_guess - 0.2 * dy,
        y0_right_guess + 0.2 * dy,
        format=DEFAULT_FORMAT,
    )

    params = [amp_left, amp_right, sigma_left, sigma_right, x0, y0_left, y0_right]

    def fitfunc(x, params):
        return fitting.TwoHalfGaussianFitComputer.evaluate(x, *params)

    values = guifit(
        x,
        y,
        fitfunc,
        params,
        parent=parent,
        wintitle=_("Two half-Gaussian fit"),
        name=name,
    )
    if values:
        y_fitted = fitfunc(x, values)
        fit_params = create_interactive_fit_params(
            "twohalfgaussian",
            dict(zip(computer.get_params_names(), values)),
            y,
            y_fitted,
        )
        return y_fitted, params, fit_params


# --- Piecewise exponential (raise-decay) fitting curve ------------------------
def piecewiseexponential_fit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute piecewise exponential fit (raise-decay)

    Returns (yfit, params, fit_params), where yfit is the fitted curve, params are
    the fitting parameters and fit_params is the canonical metadata dictionary"""
    # Get initial parameter estimates from Sigima DoubleExponentialFitComputer
    computer = fitting.DoubleExponentialFitComputer(x, y)
    initial_params = computer.compute_initial_params()
    x_center_guess = initial_params["x_center"]
    a_left_guess = initial_params["a_left"]
    b_left_guess = initial_params["b_left"]
    a_right_guess = initial_params["a_right"]
    b_right_guess = initial_params["b_right"]
    y0_guess = initial_params["y0"]

    # Create parameter bounds
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    x_range = x_max - x_min

    # Parameter bounds with more realistic ranges
    # New model signature: func(x, x_center, a_left, b_left, a_right, b_right, y0)
    # Amplitudes are bounded symmetrically: a `(0, guess * 10)` range silently
    # inverts into an empty interval whenever the guess is negative.
    amp_bound = max(abs(a_left_guess), abs(a_right_guess), y_range) * 10.0
    # Rates are bounded symmetrically too: forcing b_left > 0 and b_right < 0
    # assumes a rise-then-decay shape and makes the opposite shape unreachable.
    rate_bound = 100.0 / x_range
    x_center = FitParam(
        _("Center position"), x_center_guess, x_min, x_max, format=DEFAULT_FORMAT
    )
    a_left = FitParam(
        _("Left amplitude"),
        a_left_guess,
        -amp_bound,
        amp_bound,
        format=DEFAULT_FORMAT,
    )
    b_left = FitParam(
        _("Left rate") + " (bL)",
        b_left_guess,  # Already in coefficient form
        -rate_bound,
        rate_bound,
        format=DEFAULT_FORMAT,
    )
    a_right = FitParam(
        _("Right amplitude"),
        a_right_guess,
        -amp_bound,
        amp_bound,
        format=DEFAULT_FORMAT,
    )
    b_right = FitParam(
        _("Right rate") + " (bR)",
        b_right_guess,  # Already in coefficient form
        -rate_bound,
        rate_bound,
        format=DEFAULT_FORMAT,
    )
    y0 = FitParam(
        _("Base line"),
        y0_guess,
        y0_guess - 0.2 * y_range,
        y0_guess + 0.2 * y_range,
        format=DEFAULT_FORMAT,
    )

    params = [x_center, a_left, b_left, a_right, b_right, y0]

    def fitfunc(x, params):
        return fitting.DoubleExponentialFitComputer.evaluate(x, *params)

    values = guifit(
        x,
        y,
        fitfunc,
        params,
        parent=parent,
        wintitle=_("Piecewise exponential (raise-decay) fit"),
        name=name,
    )
    if values:
        y_fitted = fitfunc(x, values)
        # Sigima registers this model under "doubleexponential": the fit type is
        # not derived from the dialog function name.
        fit_params = create_interactive_fit_params(
            "doubleexponential",
            dict(zip(computer.get_params_names(), values)),
            y,
            y_fitted,
        )
        return y_fitted, params, fit_params
