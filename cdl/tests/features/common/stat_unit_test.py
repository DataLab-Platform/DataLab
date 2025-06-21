# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Statistics unit test

Testing the following:
  - Create a signal
  - Compute statistics on signal and compare with expected results
  - Create an image
  - Compute statistics on image and compare with expected results
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
import scipy.integrate as spt

import sigima_.image.measurement as si
import sigima_.obj as so
import sigima_.signal as ss


def get_analytical_stats(data: np.ndarray) -> dict[str, float]:
    """Compute analytical statistics for data

    Args:
        data: Array of data

    Returns:
        Dictionary with analytical statistics
    """
    results = {}
    if data.shape[0] == 2:
        # This is a signal data (row 0: x, row 1: y)
        results["trapz"] = spt.trapezoid(data[1], data[0])
        data = data[1]
    results.update(
        {
            "min": np.min(data),
            "max": np.max(data),
            "mean": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "mean/std": np.mean(data) / np.std(data),
            "ptp": np.ptp(data),
            "sum": np.sum(data),
        }
    )
    return results


def create_reference_signal() -> so.SignalObj:
    """Create reference signal"""
    snew = so.NewSignalParam.create(title="Gaussian", stype=so.SignalTypes.GAUSS)
    extra_param = so.GaussLorentzVoigtParam()
    sig = so.create_signal_from_param(snew, extra_param=extra_param)
    sig.roi = so.create_signal_roi([len(sig.x) // 2, len(sig.x) - 1], indices=True)
    return sig


def create_reference_image() -> so.ImageObj:
    """Create reference image"""
    inew = so.NewImageParam.create(title="2D-Gaussian", itype=so.ImageTypes.GAUSS)
    extra_param = so.Gauss2DParam()
    ima = so.create_image_from_param(inew, extra_param=extra_param)
    dy, dx = ima.data.shape
    ima.roi = so.create_image_roi(
        "rectangle",
        [
            [dx // 2, 0, dx, dy],
            [0, 0, dx // 3, dy // 3],
            [dx // 2, dy // 2, dx, dy],
        ],
    )
    return ima


@pytest.mark.validation
def test_signal_stats_unit() -> None:
    """Validate computed statistics for signals"""
    obj = create_reference_signal()
    res = ss.stats(obj)
    df = res.to_dataframe()
    ref = get_analytical_stats(so.xydata)
    name_map = {
        "min": "min(y)",
        "max": "max(y)",
        "mean": "<y>",
        "median": "median(y)",
        "std": "σ(y)",
        "mean/std": "<y>/σ(y)",
        "ptp": "peak-to-peak(y)",
        "sum": "Σ(y)",
        "trapz": "∫ydx",
    }
    for key, val in ref.items():
        colname = name_map[key]
        assert colname in df
        assert np.isclose(df[colname][0], val), f"Incorrect value for {colname}"

    # Given the fact that signal ROI is set to [len(sig.x) // 2, len(sig.x) - 1],
    # we may check the relationship between the results on the whole signal and the ROI:
    for key, val in ref.items():
        colname = name_map[key]
        if key in ("trapz", "sum"):
            assert np.isclose(df[colname][1], val / 2, rtol=0.02)
        elif key == "median":
            continue
        else:
            assert np.isclose(df[colname][1], val, rtol=0.01)


@pytest.mark.validation
def test_image_stats_unit() -> None:
    """Validate computed statistics for images"""
    obj = create_reference_image()

    # Ignore "RuntimeWarning: invalid value encountered in scalar divide" in the test
    # (this warning is due to the fact that the 2nd ROI has zero sum of pixel values,
    # hence the mean/std is NaN)
    with np.errstate(invalid="ignore"):
        res = si.stats(obj)

    df = res.to_dataframe()
    ref = get_analytical_stats(so.data)
    name_map = {
        "min": "min(z)",
        "max": "max(z)",
        "mean": "<z>",
        "median": "median(z)",
        "std": "σ(z)",
        "mean/std": "<z>/σ(z)",
        "ptp": "peak-to-peak(z)",
        "sum": "Σ(z)",
    }
    for key, val in ref.items():
        colname = name_map[key]
        assert colname in df
        assert np.isclose(df[colname][0], val, rtol=1e-4, atol=1e-5), (
            f"Incorrect value for {colname}"
        )

    # Given the fact that image ROI is set to
    # [[dx // 2, 0, dx, dy], [0, 0, dx // 3, dy // 3], [dx // 2, dy // 2, dx, dy]],
    # we may check the relationship between the results on the whole image and the ROIs:
    for key, val in ref.items():
        colname = name_map[key]
        if key == "sum":
            assert np.isclose(df[colname][1], val / 2, rtol=0.02)
            assert np.isclose(df[colname][3], val / 4, rtol=0.02)
        elif key == "median":
            continue
        else:
            assert np.isclose(df[colname][1], val, rtol=0.01)
            assert np.isclose(df[colname][3], val, rtol=0.01)
            if key != "mean/std":
                assert np.isclose(df[colname][2], 0.0)


if __name__ == "__main__":
    test_signal_stats_unit()
    test_image_stats_unit()
