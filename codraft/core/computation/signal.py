# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
CodraFT Computation / Signal module
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import numpy as np


# ----- Filtering functions ----------------------------------------------------
def moving_average(y, n):
    """Compute moving average"""
    y_padded = np.pad(y, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(y_padded, np.ones((n,)) / n, mode="valid")


# ----- Misc. functions --------------------------------------------------------
def derivative(x, y):
    """Compute numerical derivative"""
    dy = np.zeros_like(y)
    dy[0:-1] = np.diff(y) / np.diff(x)
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dy


def normalize(yin, parameter="maximum"):
    """
    Normalize input array *yin* with respect to parameter *parameter*

    Support values for *parameter*:
        'maximum' (default), 'amplitude', 'sum', 'energy'
    """
    axis = len(yin.shape) - 1
    if parameter == "maximum":
        maximum = np.max(yin, axis)
        if axis == 1:
            maximum = maximum.reshape((len(maximum), 1))
        maxarray = np.tile(maximum, yin.shape[axis]).reshape(yin.shape)
        return yin / maxarray
    if parameter == "amplitude":
        ytemp = np.array(yin, copy=True)
        minimum = np.min(yin, axis)
        if axis == 1:
            minimum = minimum.reshape((len(minimum), 1))
        ytemp -= minimum
        return normalize(ytemp, parameter="maximum")
    if parameter == "sum":
        return yin / yin.sum()
    if parameter == "energy":
        return yin / (yin * yin.conjugate()).sum()
    raise RuntimeError(f"Unsupported parameter {parameter}")


def xy_fft(x, y):
    """Compute FFT on X,Y data"""
    y1 = np.fft.fft(y)
    x1 = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], d=x[1] - x[0]))
    return x1, y1


def xy_ifft(x, y):
    """Compute iFFT on X,Y data"""
    y1 = np.fft.ifft(y)
    x1 = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], d=x[1] - x[0]))
    return x1, y1


# ----- Peak detection functions -----------------------------------------------
def peak_indexes(y, thres=0.3, min_dist=1, thres_abs=False):
    #  Copyright (c) 2014 Lucas Hermann Negri
    #  Unmodified code snippet from PeakUtils 1.3.0
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order
    difference. By using *thres* and *min_dist* parameters, it is possible
    to reduce the number of detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    thres_abs: boolean
        If True, the thres value will be interpreted as an absolute value,
        instead of a normalized threshold.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    if not thres_abs:
        thres = thres * (np.max(y) - np.min(y)) + np.min(y)

    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels
    # (0-value)
    (zeros,) = np.where(dy == 0)

    # check if the signal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])

    if len(zeros):
        # compute first order difference of zero indexes
        zeros_diff = np.diff(zeros)
        # check when zeros are not chained together
        (zeros_diff_not_one,) = np.add(np.where(zeros_diff != 1), 1)
        # make an array of the chained zero indexes
        zero_plateaus = np.split(zeros, zeros_diff_not_one)

        # fix if leftmost value in dy is zero
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)

        # fix if rightmost value of dy is zero
        if len(zero_plateaus) > 0 and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)

        # for each chain of zero indexes
        for plateau in zero_plateaus:
            median = np.median(plateau)
            # set leftmost values to leftmost non zero values
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            # set rightmost and middle values to rightmost non zero values
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]

    # find the peaks by using the first order difference
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, thres))
    )[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def xpeak(x, y):
    """Return default peak X-position (assuming a single peak)"""
    peaks = peak_indexes(y)
    if peaks.size:
        return x[peaks[0]]
    return np.average(x, weights=y)
