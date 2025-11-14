# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Plot results application test:

Testing the "Plot results" feature with different options:
  - Different plot kinds (one curve per object vs. one curve per title)
  - Different X/Y axis selections
  - Results with and without ROIs
  - Both signal and image panels
  - Multiple result types (scalar and geometry results)
  - Group selection creates a new result group
"""

# guitest: show

from __future__ import annotations

from typing import Generator

import sigima.objects
import sigima.params
from sigima.tests import data as test_data

from datalab.config import _
from datalab.env import execenv
from datalab.objectmodel import get_uuid
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
    Verify that results are created in the "Results" group.
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

            # Get number of groups before plotting
            groups_before = len(panel.objmodel.get_groups())

            panel.objview.selectAll()
            panel.show_results()
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="Δx")

            # Verify a Results group was created
            groups_after = panel.objmodel.get_groups()
            assert len(groups_after) == groups_before + 1, (
                f"Expected {groups_before + 1} groups, got {len(groups_after)}"
            )

            # Verify the new group is named "Results"
            expected_title = _("Results")
            result_group = groups_after[-1]
            assert result_group.title == expected_title, (
                f"Expected last group to be '{expected_title}', "
                f"got '{result_group.title}'"
            )

            # Verify the Results group contains the result signal
            assert len(result_group) > 0, (
                "Results group should contain at least one result signal"
            )

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


def test_plot_results_with_group_selection():
    """Test plot results with Results group.

    All plot results operations should create result signals in a reusable
    "Results" group for better organization.
    """
    with datalab_test_app_context() as win:
        panel = win.signalpanel

        with execenv.context(unattended=True):
            # Create a group with signals
            panel.add_group("Test Group")

            # Add signals and compute FWHM
            for i, (sig, _fwhm) in enumerate(
                iterate_noisy_signals(3, a=10.0, sigma=0.01)
            ):
                sig.title = f"Signal_{i + 1}"
                panel.add_object(sig)
                panel.processor.run_feature("fwhm", sigima.params.FWHMParam())

            # Get the number of groups before plotting results
            groups_before = len(panel.objmodel.get_groups())

            # Select the group (not individual objects)
            panel.objview.select_groups([1])

            # Verify the group is selected
            sel_groups = panel.objview.get_sel_groups()
            assert len(sel_groups) == 1, (
                f"Expected 1 selected group, got {len(sel_groups)}"
            )

            # Plot results - this should create or reuse a "Results" group
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="Δx")

            # Verify a new group was created
            groups_after = panel.objmodel.get_groups()
            assert len(groups_after) == groups_before + 1, (
                f"Expected {groups_before + 1} groups, got {len(groups_after)}"
            )

            # Check that the new group is named "Results" (or its translation)
            expected_title = _("Results")
            result_group = groups_after[-1]  # Last group should be Results
            assert result_group.title == expected_title, (
                f"Expected last group to be '{expected_title}', "
                f"got '{result_group.title}'"
            )

            # Check that the Results group contains at least one result signal
            assert len(result_group) > 0, (
                "Results group should contain at least one result signal"
            )

            # Verify that the result signal title includes source object short IDs
            result_signal = list(result_group)[0]
            # Should contain all three source signal IDs: s001, s002, s003
            # (s000 is the default group, so signals start at s001)
            assert "(s001, s002, s003)" in result_signal.title, (
                f"Result signal title should include source IDs (s001, s002, s003), "
                f"got '{result_signal.title}'"
            )

            # Test that the group is reused: create another group and plot results
            test_group_2 = panel.add_group("Test Group 2")
            test_group_2_id = get_uuid(test_group_2)
            for i, (sig, _fwhm) in enumerate(
                iterate_noisy_signals(2, a=10.0, sigma=0.01)
            ):
                sig.title = f"Signal2_{i + 1}"
                panel.add_object(sig, group_id=test_group_2_id)
                panel.processor.run_feature("fwhm", sigima.params.FWHMParam())

            # Select the second group and plot results again
            panel.objview.select_groups([3])

            # Plot results again
            num_results_before = len(result_group)
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="Δx")

            # Verify no new group was created (reused existing Results group)
            groups_final = panel.objmodel.get_groups()
            assert len(groups_final) == len(groups_after), (
                "Results group should be reused, no new group created"
            )

            # Verify more results were added to the existing Results group
            assert len(result_group) > num_results_before, (
                "More results should be added to the existing Results group"
            )

            # Test with many objects (more than 3) to verify "..." is used
            test_group_3 = panel.add_group("Test Group 3")
            test_group_3_id = get_uuid(test_group_3)
            for i, (sig, _fwhm) in enumerate(
                iterate_noisy_signals(5, a=10.0, sigma=0.01)
            ):
                sig.title = f"Signal3_{i + 1}"
                panel.add_object(sig, group_id=test_group_3_id)
                panel.processor.run_feature("fwhm", sigima.params.FWHMParam())

            # Select the third group
            panel.objview.select_groups([4])

            # Plot results
            num_results_before = len(result_group)
            panel.plot_results(kind="one_curve_per_title", xaxis="indices", yaxis="Δx")

            # Verify the result signal title uses "..." for many objects
            new_results = list(result_group)[num_results_before:]
            assert len(new_results) > 0, "Should have new results"
            result_signal_many = new_results[0]
            # With 5 source signals, should show first 2 IDs, "...", then last ID
            # Format: "fwhm (s..., s..., ..., s...): ..."
            assert ", ..., " in result_signal_many.title, (
                f"Result signal title should use '...' before last ID, "
                f"got '{result_signal_many.title}'"
            )


if __name__ == "__main__":
    test_plot_results_signals_one_curve_per_title()
    test_plot_results_images_one_curve_per_object()
    test_plot_results_images_with_rois()
    test_plot_results_with_group_selection()
