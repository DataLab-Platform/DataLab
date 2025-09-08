# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog widgets"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog
from plotpy.plot import PlotOptions
from plotpy.widgets.fit import FitDialog, FitParam
from scipy.optimize import curve_fit
from scipy.special import erf  # pylint: disable=no-name-in-module
from sigima.tests.helpers import get_default_test_name
from sigima.tools.checks import check_1d_arrays
from sigima.tools.signal import fitmodels, fourier, peakdetection

from datalab.config import _


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
            title=title, xlabel=xlabel, ylabel=ylabel, curve_antialiasing=True
        ),
        parent=parent,
        param_cols=param_cols,
        auto_fit=auto_fit,
    )
    if name is None:
        name = get_default_test_name()
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
def polynomialfit(x, y, degree, parent=None, name=None):
    """Compute polynomial fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    ivals = np.polyfit(x, y, degree)

    params = []
    for index in range(degree + 1):
        val = ivals[index]
        vmax = max(1.0, np.abs(val))
        param = FitParam(f"c{(len(ivals) - index - 1):d}", val, -2 * vmax, 2 * vmax)
        params.append(param)

    def fitfunc(x, params):
        return np.polyval(params, x)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Polymomial fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


def linearfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute linear fit using polynomialfit.

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters
    """
    return polynomialfit(x, y, 1, parent=parent, name=name)


# --- Gaussian fitting curve ---------------------------------------------------
def gaussianfit(x, y, parent=None, name=None):
    """Compute Gaussian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    sigma = dx * 0.1
    amp = fitmodels.GaussianModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", peakdetection.xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fitmodels.GaussianModel.func(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Gaussian fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Lorentzian fitting curve -------------------------------------------------
def lorentzianfit(x, y, parent=None, name=None):
    """Compute Lorentzian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    sigma = dx * 0.1
    amp = fitmodels.LorentzianModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", peakdetection.xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fitmodels.LorentzianModel.func(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Lorentzian fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Voigt fitting curve ------------------------------------------------------
def voigtfit(x, y, parent=None, name=None):
    """Compute Voigt fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    sigma = dx * 0.1
    amp = fitmodels.VoigtModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", peakdetection.xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fitmodels.VoigtModel.func(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Voigt fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Multi-Gaussian fitting curve ---------------------------------------------
def multigaussian(x, *values, **kwargs):
    """Return a 1-dimensional multi-Gaussian function."""
    a_amp = values[0::2]
    a_sigma = values[1::2]
    y0 = values[-1]
    a_x0 = kwargs["a_x0"]
    y = np.zeros_like(x) + y0
    for amp, sigma, x0 in zip(a_amp, a_sigma, a_x0):
        y += amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    return y


def multigaussianfit(x, y, peak_indices, parent=None, name=None):
    """Compute Multi-Gaussian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    params = []
    for index, i0 in enumerate(peak_indices):
        istart = 0
        iend = len(x) - 1
        if index > 0:
            istart = (peak_indices[index - 1] + i0) // 2
        if index < len(peak_indices) - 1:
            iend = (peak_indices[index + 1] + i0) // 2
        dx = 0.5 * (x[iend] - x[istart])
        dy = np.max(y[istart:iend]) - np.min(y[istart:iend])
        stri = f"{index + 1:02d}"
        params += [
            FitParam(("A") + stri, y[i0], 0.0, dy * 2),
            FitParam("σ" + stri, dx / 10, dx / 100, dx),
        ]

    params.append(
        FitParam(
            _("Y0"), np.min(y), np.min(y) - 0.1 * (np.max(y) - np.min(y)), np.max(y)
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
        return fitfunc(x, values), params


# --- Exponential fitting curve ------------------------------------------------


def exponentialfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute exponential fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""

    optp: np.ndarray

    def modelfunc(x, a, b, c):
        return a * np.exp(b * x) + c

    optp, __ = curve_fit(modelfunc, x, y)  # pylint: disable=unbalanced-tuple-unpacking
    oa, ob, oc = optp
    moa, mob, moc = np.maximum(1, optp)
    a_p = FitParam(_("A coefficient"), oa, -2 * moa, 2 * moa, logscale=True)
    b_p = FitParam(_("B coefficient"), ob, 0.5 * mob, 1.5 * mob)
    c_p = FitParam(_("y0 constant"), oc, -2 * moc, 2 * moc)

    params = [a_p, b_p, c_p]

    def fitfunc(x, params):
        return modelfunc(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Exponential fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


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


def sinusoidalfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute sinusoidal fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""

    guess_a = (np.max(y) - np.min(y)) / 2
    guess_f = dominant_frequency(x, y)
    guess_ph = 0
    guess_c = np.mean(y, dtype=float)

    moa, mof, _mop, moc = np.maximum(1, [guess_a, guess_f, guess_ph, guess_c])
    a_p = FitParam(_("Amplitude"), guess_a, -2 * moa, 2 * moa)
    f_p = FitParam(_("Frequency"), guess_f, 0, 2 * mof)
    p_p = FitParam(_("Phase"), guess_ph, -360, 360)
    c_p = FitParam(_("Continuous component"), guess_c, -2 * moc, 2 * moc)

    params = [a_p, f_p, p_p, c_p]

    def modelfunc(x, a, f, p, c):
        return a * np.sin(2 * np.pi * f * x + np.deg2rad(p)) + c

    def fitfunc(x, params):
        return modelfunc(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Sinusoidal fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Cumulative distribution function fitting curve -----------------------------------


def cdffit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute Cumulative Distribution Function (CDF) fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dy = np.max(y) - np.min(y)
    a_guess = dy
    b_guess = dy / 2
    sigma_guess = (max(x) - min(x)) / 10
    mu_guess = (max(x) - abs(min(x))) / 2

    iamp, ix0, islope, _ioff = np.maximum(1, [a_guess, mu_guess, sigma_guess, b_guess])
    a = FitParam(_("Amplitude"), a_guess, 0, iamp * 1.2)
    b = FitParam(_("Base line"), b_guess, np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma_guess, islope * 0.1, islope * 2)
    mu = FitParam(_("Mean") + " (μ)", mu_guess, ix0 * 0.2, ix0 * 2)

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
        return fitfunc(x, values), params


# --- Planckian fitting curve --------------------------------------------------
def planckianfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute Planckian (blackbody radiation) fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dy = np.max(y) - np.min(y)

    # Initial parameter estimates for Planckian fitting
    x_peak = x[np.argmax(y)]
    y_max = np.max(y)
    y_min = np.min(y)
    dy = y_max - y_min

    # For Planckian curves, use the detected peak position as the Wien
    # displacement parameter
    x0_guess = x_peak  # Peak wavelength

    # Amplitude estimation: should be reasonable for the corrected model
    amp_guess = dy  # Direct scaling with intensity range

    # Sigma estimation: start with 1.0 (canonical Planck curve)
    # sigma > 1.0 gives broader curves (cooler)
    # sigma < 1.0 gives sharper curves (hotter)
    sigma_guess = 1.0

    y0_guess = y_min

    # Parameter bounds with appropriate ranges for Planckian fitting
    amp = FitParam(_("Amplitude"), amp_guess, amp_guess * 0.01, amp_guess * 100)
    x0 = FitParam(_("Peak wavelength"), x0_guess, np.min(x), np.max(x))
    sigma = FitParam(_("Width factor"), sigma_guess, 0.1, 5.0)
    y0 = FitParam(_("Base line"), y0_guess, y0_guess - 0.2 * dy, y0_guess + 0.2 * dy)

    params = [amp, x0, sigma, y0]

    def fitfunc(x, params):
        return fitmodels.PlanckianModel.func(x, *params)

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Planckian fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Multi-Lorentzian fitting curve -------------------------------------------
def multilorentzianfit(
    x: np.ndarray, y: np.ndarray, peak_indices, parent=None, name=None
):
    """Compute Multi-Lorentzian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    params = []
    for index, i0 in enumerate(peak_indices):
        istart = 0
        iend = len(x) - 1
        if index > 0:
            istart = (peak_indices[index - 1] + i0) // 2
        if index < len(peak_indices) - 1:
            iend = (peak_indices[index + 1] + i0) // 2
        dx = 0.5 * (x[iend] - x[istart])
        dy = np.max(y[istart:iend]) - np.min(y[istart:iend])
        sigma = dx * 0.1
        amp = fitmodels.LorentzianModel.get_amp_from_amplitude(dy, sigma)

        stri = f"{index + 1:02d}"
        params += [
            FitParam(("A") + stri, amp, 0.0, amp * 1.2),
            FitParam("σ" + stri, sigma, sigma * 0.2, sigma * 10),
        ]

    params.append(
        FitParam(
            _("Y0"), np.min(y), np.min(y) - 0.1 * (np.max(y) - np.min(y)), np.max(y)
        )
    )

    kwargs = {"a_x0": x[peak_indices]}

    def nlorentzian(x, *values, **kwargs):
        """Return a 1-dimensional multi-Lorentzian function."""
        a_amp = values[0::2]
        a_sigma = values[1::2]
        y0 = values[-1]
        a_x0 = kwargs["a_x0"]
        y = np.zeros_like(x) + y0
        for amp, sigma, x0 in zip(a_amp, a_sigma, a_x0):
            y += fitmodels.LorentzianModel.func(x, amp, sigma, x0, 0)
        return y

    def fitfunc(xi, params):
        return nlorentzian(xi, *params, **kwargs)

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
        return fitfunc(x, values), params


# --- Two half-Gaussian fitting curve ------------------------------------------
def twohalfgaussianfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute two half-Gaussian fit for asymmetric peaks

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dx = np.max(x) - np.min(x)
    dy = np.max(y) - np.min(y)
    x_peak = x[np.argmax(y)]
    y_min = np.min(y)

    # Improved parameter estimation
    # For the updated model with separate left/right parameters
    amp_guess = dy  # Direct height estimation for both sides

    # Estimate asymmetry by analyzing peak shape
    half_max = y_min + dy * 0.5

    # Find points at half maximum
    left_points = np.where((x < x_peak) & (y >= half_max))[0]
    right_points = np.where((x > x_peak) & (y >= half_max))[0]

    # Estimate sigma values from half-width measurements
    if len(left_points) > 0:
        left_hw = x_peak - x[left_points[0]]
        sigma_left_guess = left_hw / np.sqrt(2 * np.log(2))  # Convert HWHM to sigma
    else:
        sigma_left_guess = dx * 0.05

    if len(right_points) > 0:
        right_hw = x[right_points[-1]] - x_peak
        sigma_right_guess = right_hw / np.sqrt(2 * np.log(2))  # Convert HWHM to sigma
    else:
        sigma_right_guess = dx * 0.05

    x0_guess = x_peak
    y0_guess = y_min

    # Parameter bounds with better ranges
    # New model signature: func(x, amp_left, amp_right, sigma_left,
    #                            sigma_right, x0, y0_left, y0_right)
    amp_left = FitParam(_("Left amplitude"), amp_guess, dy * 0.1, dy * 3)
    amp_right = FitParam(_("Right amplitude"), amp_guess, dy * 0.1, dy * 3)
    sigma_left = FitParam(
        _("Left width") + " (σL)",
        sigma_left_guess,
        dx * 0.001,  # Very small minimum
        dx * 0.5,  # Reasonable maximum
    )
    sigma_right = FitParam(
        _("Right width") + " (σR)",
        sigma_right_guess,
        dx * 0.001,  # Very small minimum
        dx * 0.5,  # Reasonable maximum
    )
    x0 = FitParam(_("Center") + " (x₀)", x0_guess, np.min(x), np.max(x))
    y0_left = FitParam(
        _("Left baseline"), y0_guess, y0_guess - 0.2 * dy, y0_guess + 0.2 * dy
    )
    y0_right = FitParam(
        _("Right baseline"), y0_guess, y0_guess - 0.2 * dy, y0_guess + 0.2 * dy
    )

    params = [amp_left, amp_right, sigma_left, sigma_right, x0, y0_left, y0_right]

    def fitfunc(x, params):
        return fitmodels.TwoHalfGaussianModel.func(x, *params)

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
        return fitfunc(x, values), params


# --- Double exponential fitting curve -----------------------------------------
def doubleexponentialfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute double exponential fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""

    # Improved parameter estimation for double exponential decay
    y_range = np.max(y) - np.min(y)
    x_range = np.max(x) - np.min(x)
    y_min = np.min(y)
    y_max = np.max(y)

    # Better initial estimates based on data analysis
    # Assume the curve starts high and decays to baseline
    initial_value = y[0] if len(y) > 0 else y_max
    final_value = y[-1] if len(y) > 0 else y_min

    # Estimate baseline from final portion of data
    if len(y) > 10:
        y0_guess = np.mean(
            y[-max(5, len(y) // 10) :]
        )  # Average of last 10% or 5 points
    else:
        y0_guess = final_value

    # Estimate amplitudes: total amplitude split between components
    total_amp = initial_value - y0_guess
    amp1_guess = total_amp * 0.7  # Fast component (typically dominant)
    amp2_guess = total_amp * 0.3  # Slow component

    # Better time constant estimation based on decay behavior
    # Look for characteristic times where signal drops to 1/e
    target_fast = y0_guess + total_amp / np.e  # 1/e point
    target_slow = y0_guess + total_amp * 0.5  # Half-life point

    # Find approximate time constants from data
    tau1_guess = x_range * 0.05  # Fast decay (5% of range)
    tau2_guess = x_range * 0.3  # Slow decay (30% of range)

    # Try to estimate from actual data if enough points
    if len(x) > 20:
        # Find where signal crosses the 1/e threshold
        try:
            fast_idx = np.where(y <= target_fast)[0]
            if len(fast_idx) > 0:
                tau1_guess = x[fast_idx[0]] * 0.7  # Adjust for double exponential
        except (IndexError, ValueError):
            pass  # Keep default estimate

        try:
            slow_idx = np.where(y <= target_slow)[0]
            if len(slow_idx) > 0:
                tau2_guess = x[slow_idx[-1]] * 0.5  # Adjust for double exponential
        except (IndexError, ValueError):
            pass  # Keep default estimate

    # Parameter bounds with more realistic ranges
    # New model signature: func(x, x_center, a_left, b_left, a_right, b_right, y0)
    x_center = FitParam(_("Center position"), np.mean(x), np.min(x), np.max(x))
    a_left = FitParam(_("Left amplitude"), amp1_guess, 0.0, total_amp * 2)
    b_left = FitParam(
        _("Left time constant") + " (bL)",
        -1.0 / tau1_guess if tau1_guess > 0 else -1.0,  # Negative for decay
        -10.0 / x_range,  # Fast decay
        -0.001 / x_range,  # Slow decay
    )
    a_right = FitParam(_("Right amplitude"), amp2_guess, 0.0, total_amp * 2)
    b_right = FitParam(
        _("Right time constant") + " (bR)",
        -1.0 / tau2_guess if tau2_guess > 0 else -1.0,  # Negative for decay
        -10.0 / x_range,  # Fast decay
        -0.001 / x_range,  # Slow decay
    )
    y0 = FitParam(
        _("Base line"), y0_guess, y0_guess - 0.1 * y_range, y0_guess + 0.1 * y_range
    )

    params = [x_center, a_left, b_left, a_right, b_right, y0]

    def fitfunc(x, params):
        return fitmodels.DoubleExponentialModel.func(x, *params)

    values = guifit(
        x,
        y,
        fitfunc,
        params,
        parent=parent,
        wintitle=_("Double exponential fit"),
        name=name,
    )
    if values:
        return fitfunc(x, values), params
