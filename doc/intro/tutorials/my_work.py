# -*- coding: utf-8 -*-

"""
Tutorial 'Working with Spyder'
==============================

Generate 1D data that can be used in the tutorials to represent some results.
"""

from __future__ import annotations

import numpy as np
from cdlclient import SimpleRemoteProxy


def generate_1d_data(
    amplitude: float,
    peak_position: float,
    peak_width: float,
    relaxation_oscillations: bool = True,
    noise: bool = True,
    noise_std: float = 0.1,
    x_min: float = -1,
    x_max: float = 1,
    num_points: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 1D data that can be used in the tutorials to represent some results.

    Args:
        amplitude: Amplitude of the peak.
        peak_position: Position of the peak.
        peak_width: Width of the peak.
        relaxation_oscillations: Whether to add relaxation oscillations to the data,
         by default True.
        noise: Whether to add noise to the data, by default True.
        noise_std: Standard deviation of the noise, by default 0.1.
        x_min: Minimum value of the x axis, by default -1.
        x_max: Maximum value of the x axis, by default 1.
        num_points: Number of points in the generated data, by default 100.

    Returns:
        Generated data (x, y; where y is a 1D array and x is a 1D array).
    """
    x = np.linspace(x_min, x_max, num_points)
    y = amplitude * np.exp(-(((x - peak_position) / peak_width) ** 2))
    if relaxation_oscillations:
        y += 0.1 * np.sin(10 * x)
    if noise:
        y += np.random.normal(scale=noise_std, size=num_points)
    return x, y


def generate_2d_data(
    num_lines: int = 10,
    noise_std: float = 0.1,
    x_min: float = -1,
    x_max: float = 1,
    num_points: int = 100,
    debug_with_datalab: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D data that can be used in the tutorials to represent some results.

    Args:
        num_lines: Number of lines in the generated data, by default 10.
        noise_std: Standard deviation of the noise, by default 0.1.
        x_min: Minimum value of the x axis, by default -1.
        x_max: Maximum value of the x axis, by default 1.
        num_points: Number of points in the generated data, by default 100.
        debug_with_datalab: Whether to use the DataLab to debug the function,
         by default False.

    Returns:
        Generated data (x, y; where y is a 2D array and x is a 1D array).
    """
    proxy = None
    if debug_with_datalab:
        proxy = SimpleRemoteProxy()
        proxy.connect()
    z = np.zeros((num_lines, num_points))
    for i in range(num_lines):
        amplitude = 0.1 * i**2
        peak_position = 0.5 * i**2
        peak_width = 0.1 * i**2
        x, y = generate_1d_data(
            amplitude,
            peak_position,
            peak_width,
            relaxation_oscillations=True,
            noise=True,
            noise_std=noise_std,
            x_min=x_min,
            x_max=x_max,
            num_points=num_points,
        )
        z[i] = y
        if proxy is not None:
            proxy.add_signal(f"Line {i}", x, y)
    return x, z


def test_my_1d_algorithm() -> np.ndarray:
    """Generate 1D data that can be used in the tutorials to represent some results."""
    # Generate 1D data using the function generate_1d_data:
    data = generate_1d_data(1, 0, 0.1, relaxation_oscillations=True, noise=True)
    return data


def test_my_2d_algorithm(debug_with_datalab: bool = False) -> np.ndarray:
    """Generate 2D data that can be used in the tutorials to represent some results."""
    # Generate 2D data using the function generate_2d_data:
    data = generate_2d_data(
        num_lines=10, noise_std=0.1, debug_with_datalab=debug_with_datalab
    )
    return data
