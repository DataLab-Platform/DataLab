# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for multiple geometry results with multiple ROIs
======================================================

This test verifies that multiple geometry results (e.g., FWHM and FW1e2) can coexist
when computed on signals with multiple ROIs (e.g., multiple pulse peaks).
"""

from __future__ import annotations

import numpy as np
import sigima.objects as sio
import sigima.params
import sigima.proc.signal as sig_proc

from datalab.adapters_metadata import GeometryAdapter


def test_multiple_results_with_multiple_rois():
    """Test that FWHM and FW1e2 work correctly with multiple ROIs."""
    # Create a signal with three Gaussian pulses
    x = np.linspace(0, 30, 3000)

    # Three pulses at different positions
    pulse1 = 1.0 * np.exp(-((x - 5) ** 2) / 2)  # Peak at x=5, sigma=1
    pulse2 = 0.8 * np.exp(-((x - 15) ** 2) / 4)  # Peak at x=15, sigma=2
    pulse3 = 1.2 * np.exp(-((x - 25) ** 2) / 1)  # Peak at x=25, sigma=0.707

    y = pulse1 + pulse2 + pulse3
    signal = sio.create_signal("Triple Pulse Signal", x, y)

    # Create three ROIs around each pulse (pass all coordinates at once)
    signal.roi = sio.create_signal_roi(
        [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]]  # Three ROI ranges
    )

    roi_count = len(list(signal.iterate_roi_indices()))
    print(f"Signal has {roi_count} ROIs")
    fwhm_param = sigima.params.FWHMParam()
    fwhm_param.method = "zero-crossing"
    fwhm_result = sig_proc.fwhm(signal, fwhm_param)

    assert fwhm_result is not None, "FWHM computation failed"
    print(f"FWHM result has {len(fwhm_result.coords)} rows")
    assert len(fwhm_result.coords) == 3, "Expected 3 FWHM measurements (one per ROI)"

    GeometryAdapter(fwhm_result).add_to(signal)

    # Compute FW1e2 (should also get one result per ROI)
    fw1e2_result = sig_proc.fw1e2(signal)

    assert fw1e2_result is not None, "FW1e2 computation failed"
    print(f"FW1e² result has {len(fw1e2_result.coords)} rows")
    assert len(fw1e2_result.coords) == 3, "Expected 3 FW1e2 measurements (one per ROI)"

    GeometryAdapter(fw1e2_result).add_to(signal)

    # Verify both results are stored separately
    results = list(GeometryAdapter.iterate_from_obj(signal))
    assert len(results) == 2, f"Expected 2 distinct results, got {len(results)}"

    # Verify the titles
    titles = {adapter.title for adapter in results}
    assert "fwhm" in titles, "FWHM result not found"
    assert "fw1e2" in titles, "FW1e2 result not found"

    # Verify each result has measurements for all 3 ROIs
    for adapter in results:
        assert len(adapter.result.coords) == 3, (
            f"{adapter.title} should have 3 measurements"
        )

        # Verify ROI indices are correct
        roi_indices = adapter.result.roi_indices
        assert roi_indices is not None, f"{adapter.title} should have ROI indices"
        assert len(roi_indices) == 3, f"{adapter.title} should have 3 ROI indices"
        assert set(roi_indices) == {0, 1, 2}, (
            f"{adapter.title} ROI indices should be [0, 1, 2]"
        )

        print(f"\n{adapter.title.upper()} measurements:")
        for i, (coords, roi_idx) in enumerate(zip(adapter.result.coords, roi_indices)):
            width = abs(coords[2] - coords[0])
            print(f"  ROI {roi_idx}: width = {width:.4f}")

    print("\n✓ Multiple results with multiple ROIs work correctly")


def test_roi_filtering():
    """Test that we can filter results by ROI index."""
    # Create a signal with two pulses
    x = np.linspace(0, 20, 2000)
    pulse1 = np.exp(-((x - 5) ** 2) / 2)
    pulse2 = np.exp(-((x - 15) ** 2) / 2)
    y = pulse1 + pulse2
    signal = sio.create_signal("Dual Pulse Signal", x, y)

    # Create two ROIs (pass all coordinates at once)
    signal.roi = sio.create_signal_roi([[0.0, 10.0], [10.0, 20.0]])

    # Compute FWHM
    fwhm_param = sigima.params.FWHMParam()
    fwhm_param.method = "zero-crossing"
    fwhm_result = sig_proc.fwhm(signal, fwhm_param)

    assert fwhm_result is not None
    assert len(fwhm_result.coords) == 2

    GeometryAdapter(fwhm_result).add_to(signal)

    # Retrieve and filter by ROI
    adapters = list(GeometryAdapter.iterate_from_obj(signal))
    assert len(adapters) == 1

    adapter = adapters[0]

    # Get data for specific ROI
    roi0_df = adapter.get_roi_data(0)
    roi1_df = adapter.get_roi_data(1)

    assert len(roi0_df) == 1, "ROI 0 should have 1 measurement"
    assert len(roi1_df) == 1, "ROI 1 should have 1 measurement"

    print("\n✓ ROI filtering works correctly")


if __name__ == "__main__":
    test_multiple_results_with_multiple_rois()
    test_roi_filtering()
    print("\n✅ All multi-ROI tests passed!")
