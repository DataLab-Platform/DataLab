# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Stability Analysis (see parent package :mod:`sigima_.algorithms.signal`)

"""

from __future__ import annotations

import numpy as np


def allan_variance(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Allan variance for given time and measurement values at specified
    tau values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Allan variance values
    """
    if len(x) != len(y):
        raise ValueError(
            "Time array (x) and measured values array (y) must have the same length."
        )

    dt = np.mean(np.diff(x))  # Time step size
    if not np.allclose(np.diff(x), dt):
        raise ValueError("Time values (x) must be equally spaced.")

    allan_var = []
    for tau in tau_values:
        m = int(round(tau / dt))  # Number of time steps in a tau
        if m < 1:
            raise ValueError(
                f"Tau value {tau} is smaller than the sampling interval {dt}"
            )
        if m > len(y) // 2:
            # Tau too large for reliable statistics
            allan_var.append(np.nan)
            continue

        # Calculate the clusters/bins
        clusters = y[: len(y) - (len(y) % m)].reshape(-1, m)
        bin_means = clusters.mean(axis=1)

        # Calculate Allan variance using the definition
        # σ²(τ) = 1/(2(N-1)) Σ(y_(i+1) - y_i)²
        # where y_i are the bin means
        squared_diff = np.sum(np.diff(bin_means) ** 2)
        n = len(bin_means) - 1

        if n > 0:
            var = squared_diff / (2.0 * n)
            allan_var.append(var)
        else:
            allan_var.append(np.nan)

    return np.array(allan_var)


def allan_deviation(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Allan deviation for given time and measurement values at specified
    tau values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Allan deviation values
    """
    return np.sqrt(allan_variance(x, y, tau_values))


def overlapping_allan_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Overlapping Allan variance for given time and measurement values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Allan deviation time values

    Returns:
        Overlapping Allan variance values
    """
    if len(x) != len(y):
        raise ValueError(
            "Time array (x) and measured values array (y) must have the same length."
        )

    dt = np.mean(np.diff(x))  # Time step size
    if not np.allclose(np.diff(x), dt):
        raise ValueError("Time values (x) must be equally spaced.")

    overlapping_var = []
    for tau in tau_values:
        tau_bins = int(tau / dt)
        if tau_bins <= 1 or tau_bins > len(y) / 2:
            overlapping_var.append(np.nan)
            continue

        m = len(y) - tau_bins  # Number of overlapping segments
        avg_values = [np.mean(y[i : i + tau_bins]) for i in range(m)]
        diff = np.diff(avg_values)
        overlapping_var.append(0.5 * np.mean(np.array(diff) ** 2))

    return np.array(overlapping_var)


def modified_allan_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Modified Allan variance for given time and measurement values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Modified Allan deviation time values

    Returns:
        Modified Allan variance values
    """
    if len(x) != len(y):
        raise ValueError(
            "Time array (x) and measured values array (y) must have the same length."
        )

    dt = np.mean(np.diff(x))
    if not np.allclose(np.diff(x), dt):
        raise ValueError("Time values (x) must be equally spaced.")

    mod_allan_var = []
    for tau in tau_values:
        tau_bins = int(tau / dt)
        if tau_bins <= 1 or tau_bins > len(y) / 2:
            mod_allan_var.append(np.nan)
            continue

        m = int(len(y) / tau_bins)
        reshaped = y[: m * tau_bins].reshape(m, tau_bins)

        avg_values = reshaped.mean(axis=1)
        squared_diff = (np.diff(avg_values)) ** 2
        mod_allan_var.append(np.mean(squared_diff) / (2 * (tau_bins**2)))

    return np.array(mod_allan_var)


def hadamard_variance(
    x: np.ndarray, y: np.ndarray, tau_values: np.ndarray
) -> np.ndarray:
    """
    Calculate the Hadamard variance for given time and measurement values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Hadamard deviation time values

    Returns:
        Hadamard variance values
    """
    if len(x) != len(y):
        raise ValueError(
            "Time array (x) and measured values array (y) must have the same length."
        )

    dt = np.mean(np.diff(x))
    if not np.allclose(np.diff(x), dt):
        raise ValueError("Time values (x) must be equally spaced.")

    hadamard_var = []
    for tau in tau_values:
        tau_bins = int(tau / dt)
        if tau_bins <= 1 or tau_bins > len(y) / 3:
            hadamard_var.append(np.nan)
            continue

        m = len(y) - 2 * tau_bins
        avg_values = [np.mean(y[i : i + tau_bins]) for i in range(m)]
        diff = np.diff(avg_values, n=2)  # Second differences
        hadamard_var.append(np.mean(diff**2) / 6)

    return np.array(hadamard_var)


def total_variance(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Total variance for given time and measurement values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Total variance time values

    Returns:
        Total variance values
    """
    if len(x) != len(y):
        raise ValueError(
            "Time array (x) and measured values array (y) must have the same length."
        )

    dt = np.mean(np.diff(x))
    if not np.allclose(np.diff(x), dt):
        raise ValueError("Time values (x) must be equally spaced.")

    total_var = []
    for tau in tau_values:
        tau_bins = int(tau / dt)
        if tau_bins <= 1 or tau_bins > len(y) / 2:
            total_var.append(np.nan)
            continue

        m = int(len(y) / tau_bins)
        reshaped = y[: m * tau_bins].reshape(m, tau_bins)

        avg_values = reshaped.mean(axis=1)
        squared_diff = np.diff(avg_values) ** 2
        total_var.append(np.mean(squared_diff))

    return np.array(total_var)


def time_deviation(x: np.ndarray, y: np.ndarray, tau_values: np.ndarray) -> np.ndarray:
    """
    Calculate the Time Deviation (TDEV) for given time and measurement values.

    Args:
        x: Time array
        y: Measured values array
        tau_values: Time deviation time values

    Returns:
        Time deviation values
    """
    allan_var = allan_variance(x, y, tau_values)
    return np.sqrt(allan_var) * tau_values
