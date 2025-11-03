# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plot results application test:

Testing the "Plot results" feature with different options:
  - Different plot kinds (one curve per object vs. one curve per title)
  - Different X/Y axis selections
  - Results with and without ROIs
  - Both signal and image panels
  - Multiple result types (scalar and geometry results)
"""

# guitest: show

from __future__ import annotations

from typing import Generator

import sigima.objects
import sigima.params
from sigima.tests import data as test_data

from datalab.env import execenv
from datalab.tests import datalab_test_app_context


def iterate_noisy_signals(
    count: int,
    a: float,
    sigma: float,
) -> Generator[tuple[sigima.objects.SignalObj, float], None, None]:
    """Generate noisy signals for testing."""
    noiseparam = sigima.objects.NormalDistribution1DParam.create(sigma=sigma, mu=0.0)
    param = sigima.objects.GaussParam.create(a=a)
    for i in range(count):
        param.sigma = 1.0 + (i * 0.1) ** 2
        theoretical_fwhm = param.get_expected_features().fwhm
        sig = test_data.create_noisy_signal(
            noiseparam, param, f"Signal|fwhm_th={theoretical_fwhm:.2f}"
        )
        yield sig, theoretical_fwhm


def test_plot_results_signals_one_curve_per_title():
    """Test plot results feature with signals, one curve per title.

    Create signals with single-value results (e.g., FWHM) and plot them.
    """
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        with execenv.context(unattended=True):
            x_th = []
            y_th = []
            for i, (sig, theoretical_fwhm) in enumerate(
                iterate_noisy_signals(5, a=10.0, sigma=0.01)
            ):
                x_th.append(i)
                y_th.append(theoretical_fwhm)
                panel.add_object(sig)
                # Compute FWHM using the default method (zero-crossing) which introduces
                # a systematic ~2% error compared to the theoretical value
                # (trade-off for noise robustness)
                panel.processor.run_feature("fwhm", sigima.params.FWHMParam())

            panel.objview.selectAll()
            panel.show_results()
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="Δx")

            fwhm_var_th = sigima.objects.create_signal("FWHM_Theoretical", x_th, y_th)
            panel.add_object(fwhm_var_th)
            panel.objview.select_objects((6, 7))
            # The observed offset should be around ~2% of the theoretical value


def test_plot_results_images_one_curve_per_object():
    """Test plot results feature with images, one curve per object.

    Create images with multi-value results (e.g., peak detection) and plot them.
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel

        with execenv.context(unattended=True):
            for i in range(3):
                img = test_data.create_peak_image()
                img.title = f"Peaks_{i + 1}"
                panel.add_object(img)
                param = sigima.params.Peak2DDetectionParam.create(create_rois=False)
                panel.processor.run_feature("peak_detection", param)

            panel.objview.selectAll()
            panel.show_results()
            # Use programmatic parameters: plot y vs x for each peak
            panel.plot_results(kind="one_curve_per_object", xaxis="x", yaxis="y")


def test_plot_results_images_with_rois():
    """Test plot results feature with images containing ROIs.

    Create images with ROIs and single-value results (e.g., centroid) and plot them.
    """
    with datalab_test_app_context() as win:
        panel = win.imagepanel

        with execenv.context(unattended=True):
            size = 512
            peak_param = test_data.PeakDataParam.create(size=size)
            for i in range(3):
                data, peak_coords = test_data.get_peak2d_data(peak_param)
                img = sigima.objects.create_image(f"Image_ROI_{i + 1}", data)
                # Add rectangular ROI to image
                roi_coords = []
                for xpeak, ypeak in peak_coords:
                    roi_coords.append([xpeak, ypeak, 20])
                img.roi = sigima.objects.create_image_roi("circle", roi_coords)
                panel.add_object(img)
                panel.processor.run_feature("centroid")

            panel.objview.selectAll()
            panel.show_results()
            # Plot centroid x vs indices with ROIs
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="x")


if __name__ == "__main__":
    test_plot_results_signals_one_curve_per_title()
    test_plot_results_images_one_curve_per_object()
    test_plot_results_images_with_rois()
