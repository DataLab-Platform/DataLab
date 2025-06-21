# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal stability analysis unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

import numpy as np
import pytest

import sigima_.computation.signal as sigima_signal
import sigima_.obj
import sigima_.param
from cdl.utils.tests import check_array_result


def generate_white_noise(n_points, sigma=1.0):
    """Generate white noise with known characteristics."""
    return np.random.normal(0, sigma, n_points)


def theoretical_allan_variance_white_noise(tau, sigma):
    """
    Calculate theoretical Allan variance for white noise.
    For white noise: AVAR(τ) = σ²/(2τ)
    But the Allan variance is computed as AVAR(τ) = σ²τ/τ = σ²τ because of the
    overlapping nature of the samples.
    """
    return sigma**2 / tau


def generate_drift_signal(n_points, slope, intercept=0):
    """Generate a linear drift signal."""
    time = np.arange(n_points)
    values = slope * time + intercept
    return time, values


def theoretical_allan_variance_drift(tau, slope):
    """
    Theoretical Allan variance for a drift signal.
    """
    return (slope**2 * tau**2) / 2


@pytest.mark.validation
def test_signal_allan_variance():
    """Test Allan variance computation against theoretical values."""
    n_points = 10000
    sigma = 1.0
    tau_values = np.array([1, 2, 5, 10, 20, 50])

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima_.obj.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima_.param.AllanVarianceParam()
    param.max_tau = max(tau_values)

    # Compute Allan variance using the high-level function
    res1 = sigima_signal.allan_variance(sig1, param)
    th_av_white = theoretical_allan_variance_white_noise(res1.x, sigma)

    check_array_result("White noise Allan variance", res1.y, th_av_white, atol=0.05)

    # Generate and test drift signal
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima_.obj.create_signal("Drift Test", time, values)

    # Compute Allan variance using the high-level function
    res2 = sigima_signal.allan_variance(sig2, param)
    th_av_drift = theoretical_allan_variance_drift(res2.x, slope)

    check_array_result("Drift Allan variance", res2.y, th_av_drift, atol=0.01)


@pytest.mark.validation
def test_signal_allan_deviation():
    """Test Allan deviation computation against theoretical values."""
    n_points = 10000
    sigma = 1.0
    tau_values = np.array([1, 2, 5, 10, 20, 50])

    # Generate and test white noise signal
    time_white = np.arange(n_points)
    values_white = generate_white_noise(n_points, sigma)
    sig1 = sigima_.obj.create_signal("White Noise Test", time_white, values_white)

    # Define Allan variance parameters
    param = sigima_.param.AllanVarianceParam()
    param.max_tau = max(tau_values)

    # Compute Allan deviation using the high-level function
    res1 = sigima_signal.allan_deviation(sig1, param)
    th_av_white = theoretical_allan_variance_white_noise(res1.x, sigma)

    check_array_result(
        "White noise Allan deviation", res1.y, np.sqrt(th_av_white), atol=0.05
    )

    # Generate and test drift signal
    slope = 0.01
    time, values = generate_drift_signal(n_points, slope)
    sig2 = sigima_.obj.create_signal("Drift Test", time, values)

    # Compute Allan deviation using the high-level function
    res2 = sigima_signal.allan_deviation(sig2, param)
    th_av_drift = theoretical_allan_variance_drift(res2.x, slope)

    check_array_result("Drift Allan deviation", res2.y, np.sqrt(th_av_drift), atol=0.01)


if __name__ == "__main__":
    test_signal_allan_variance()
    test_signal_allan_deviation()
