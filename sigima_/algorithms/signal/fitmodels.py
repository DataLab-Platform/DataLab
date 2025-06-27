# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Fit Models (see parent package :mod:`cdl.algorithms`)

"""

from __future__ import annotations

import abc

import numpy as np
import scipy.special


class FitModel(abc.ABC):
    """Curve fitting model base class"""

    @classmethod
    @abc.abstractmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""

    # pylint: disable=unused-argument
    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return cls.func(0, amp, sigma, 0, 0)

    @classmethod
    @abc.abstractmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""

    @classmethod
    def half_max_segment(cls, amp, sigma, x0, y0):
        """Return segment coordinates for y=half-maximum intersection"""
        hwhm = 0.5 * cls.fwhm(amp, sigma)
        yhm = 0.5 * cls.amplitude(amp, sigma) + y0
        return x0 - hwhm, yhm, x0 + hwhm, yhm


class GaussianModel(FitModel):
    """1-dimensional Gaussian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (
            amp / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
            + y0
        )

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma * np.sqrt(2 * np.log(2))


class LorentzianModel(FitModel):
    """1-dimensional Lorentzian fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        return (amp / (sigma * np.pi)) / (1 + ((x - x0) / sigma) ** 2) + y0

    @classmethod
    def get_amp_from_amplitude(cls, amplitude, sigma):
        """Return amp from function amplitude and sigma"""
        return amplitude * (sigma * np.pi)

    @classmethod
    def amplitude(cls, amp, sigma):
        """Return function amplitude"""
        return amp / (sigma * np.pi)

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        return 2 * sigma


class VoigtModel(FitModel):
    """1-dimensional Voigt fit model"""

    @classmethod
    def func(cls, x, amp, sigma, x0, y0):
        """Return fitting function"""
        # pylint: disable=no-member
        z = (x - x0 + 1j * sigma) / (sigma * np.sqrt(2.0))
        return y0 + amp * scipy.special.wofz(z).real / (sigma * np.sqrt(2 * np.pi))

    @classmethod
    def fwhm(cls, amp, sigma):
        """Return function FWHM"""
        wg = GaussianModel.fwhm(amp, sigma)
        wl = LorentzianModel.fwhm(amp, sigma)
        return 0.5346 * wl + np.sqrt(0.2166 * wl**2 + wg**2)
