# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog widgets"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np
from guidata.configtools import get_icon
from guidata.qthelpers import exec_dialog
from plotpy.plot import PlotOptions
from plotpy.widgets.fit import FitDialog, FitParam
from scipy.optimize import curve_fit
from scipy.special import erf

from cdl.algorithms import fit
from cdl.algorithms.signal import sort_frequencies, xpeak
from cdl.config import _
from cdl.utils.tests import get_default_test_name


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
):
    """GUI-based curve fitting tool"""
    win = FitDialog(
        edit=True,
        title=wintitle,
        icon=None,
        toolbar=True,
        options=PlotOptions(title=title, xlabel=xlabel, ylabel=ylabel),
        parent=parent,
        param_cols=param_cols,
        auto_fit=auto_fit,
    )
    if name is None:
        name = get_default_test_name()
    win.setObjectName(name)
    win.set_data(x, y, fitfunc, fitparams, fitargs, fitkwargs)
    win.autofit()  # TODO: [P3] make this optional
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
    """Compute linear fit

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
    amp = fit.GaussianModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fit.GaussianModel.func(x, *params)

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
    amp = fit.LorentzianModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fit.LorentzianModel.func(x, *params)

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
    amp = fit.VoigtModel.get_amp_from_amplitude(dy, sigma)

    a = FitParam(_("Amplitude"), amp, 0.0, amp * 1.2)
    b = FitParam(_("Base line"), np.min(y), np.min(y) - 0.1 * dy, np.max(y))
    sigma = FitParam(_("Std-dev") + " (σ)", sigma, sigma * 0.2, sigma * 10)
    mu = FitParam(_("Mean") + " (μ)", xpeak(x, y), np.min(x), np.max(x))

    params = [a, sigma, mu, b]

    def fitfunc(x, params):
        return fit.VoigtModel.func(x, *params)

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


def multigaussianfit(x, y, peak_indexes, parent=None, name=None):
    """Compute Multi-Gaussian fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    params = []
    for index, i0 in enumerate(peak_indexes):
        istart = 0
        iend = len(x) - 1
        if index > 0:
            istart = (peak_indexes[index - 1] + i0) // 2
        if index < len(peak_indexes) - 1:
            iend = (peak_indexes[index + 1] + i0) // 2
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

    kwargs = {"a_x0": x[peak_indexes]}

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

    opt_params: np.ndarray
    opt_params, __ = curve_fit(lambda x, a, b, c: a * np.exp(b * x) + c, x, y)
    oa, ob, oc = opt_params
    moa, mob, moc = np.maximum(1, opt_params)
    a_p = FitParam(_("A coefficient"), oa, -2 * moa, 2 * moa, logscale=True)
    b_p = FitParam(_("B coefficient"), ob, 0.5 * mob, 1.5 * mob)
    c_p = FitParam(_("C constant"), oc, -2 * moc, 2 * moc)

    params = [a_p, b_p, c_p]

    def fitfunc(x, params):
        return params[0] * np.exp(params[1] * x) + params[2]

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Exponential fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Sinusoidal fitting curve ------------------------------------------------


def sinusoidalfit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute sinusoidal fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""

    guess_a = (np.max(y) - np.min(y)) / 2
    guess_f = sort_frequencies(x, y)[0]
    guess_p = 0
    guess_c = np.mean(y)

    opt_params: np.ndarray
    opt_params, _covar = curve_fit(
        lambda x, a, f, p, c: a * np.sin(2 * np.pi * f * x + np.deg2rad(p)) + c,
        x,
        y,
        [guess_a, guess_f, guess_p, guess_c],
    )
    oa, of, op, oc = opt_params

    moa, mof, mop, moc = np.maximum(1, opt_params)
    a_p = FitParam(_("Amplitude"), oa, -2 * moa, 2 * moa)
    f_p = FitParam(_("Frequency"), of, 0, 2 * mof)
    p_p = FitParam(_("Phase"), op, -360, 360)
    c_p = FitParam(_("Continuous component"), oc, -2 * moc, 2 * moc)

    params = [a_p, f_p, p_p, c_p]

    def fitfunc(x, params):
        return params[0] * np.sin(2 * np.pi * params[1] * x + params[2]) + params[3]

    values = guifit(
        x, y, fitfunc, params, parent=parent, wintitle=_("Sinusoidal fit"), name=name
    )
    if values:
        return fitfunc(x, values), params


# --- Error function (ERF) fitting curve ------------------------------------------------


def erffit(x: np.ndarray, y: np.ndarray, parent=None, name=None):
    """Compute ERF (Error Function) fit

    Returns (yfit, params), where yfit is the fitted curve and params are
    the fitting parameters"""
    dy = np.max(y) - np.min(y)
    x0_guess = (max(x) - abs(min(x))) / 2
    off_guess = dy / 2
    slope_guess = (max(x) - min(x)) / 10
    amp_guess = dy

    def fitfunc(x: np.ndarray, amp: float, x0: float, slope: float, offset: float):
        return amp * erf((x - x0) / (slope * np.sqrt(2))) + offset

    params, _extras = curve_fit(
        fitfunc, x, y, p0=[amp_guess, x0_guess, slope_guess, off_guess]
    )
    amp_guess, x0_guess, slope_guess, off_guess = params

    amp_p = FitParam(_("Amplitude"), amp_guess, 0, amp_guess * 1.2)
    x0_p = FitParam(_("Center"), x0_guess, x0_guess * 0.2, x0_guess * 2)
    slope_p = FitParam(_("Slope"), slope_guess, slope_guess * 0.1, slope_guess * 2)
    off_p = FitParam(_("Offset"), off_guess, np.min(y) - 0.1 * dy, np.max(y))

    params = [amp_p, x0_p, slope_p, off_p]

    def wrapped_fitfunc(x: np.ndarray, p: list[float]):
        return fitfunc(x, *p)

    values = guifit(
        x,
        y,
        wrapped_fitfunc,
        params,
        parent=parent,
        wintitle=_("ERF fit"),
        name=name,
    )
    if values:
        return wrapped_fitfunc(x, values), params
