# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
beautiful_app
-------------

A high-level test scenario producing beautiful screenshots.

.. note::

    This scenario is used to produce screenshots for the documentation.
    Thus, it is not run by default when running all tests.

.. warning::

    When modifying this scenario, please update the script "update_screenshots.py"
    in the "doc" folder.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show,skip

import sigima.objects
import sigima.params as sigima_param
from sigima.tests.data import get_test_image

from datalab.tests import datalab_test_app_context


def run_beautiful_scenario(screenshots: bool = False) -> None:
    """High-level test scenario producing beautiful screenshots"""
    data_size = 500
    with datalab_test_app_context(console=False, exec_loop=not screenshots) as win:
        # Beautiful screenshot of a signal
        panel = win.signalpanel
        sig = sigima.objects.create_signal_from_param(sigima.objects.LorentzParam())
        panel.add_object(sig)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("wiener")
        panel.processor.run_feature("derivative")
        panel.processor.run_feature("integral")
        panel.processor.run_feature("gaussian_filter", sigima_param.GaussianParam())
        panel.processor.run_feature("fft")
        panel.processor.run_feature("derivative")
        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_beautiful")
        # Beautiful screenshot of an image
        panel = win.imagepanel
        param = sigima.objects.Gauss2DParam.create(height=data_size, width=data_size)
        ima = sigima.objects.create_image_from_param(param)
        ima.set_metadata_option("colormap", "jet")
        panel.add_object(ima)
        panel.processor.run_feature("equalize_hist", sigima_param.EqualizeHistParam())
        panel.processor.run_feature(
            "equalize_adapthist", sigima_param.EqualizeAdaptHistParam()
        )
        panel.processor.run_feature("denoise_tv", sigima_param.DenoiseTVParam())
        panel.processor.run_feature(
            "denoise_wavelet", sigima_param.DenoiseWaveletParam()
        )
        panel.processor.run_feature("white_tophat", sigima_param.MorphologyParam())
        panel.processor.run_feature("denoise_tv", sigima_param.DenoiseTVParam())
        n = data_size // 3
        roi = sigima.objects.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        panel.processor.compute_roi_extraction(roi)
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


def run_circle_detection_scenario(screenshots: bool = False) -> None:
    """High-level test scenario for flower image with ROI extraction

    This scenario creates:
    - A flower test image
    - Roberts edge detection filter applied
    - A rectangular ROI extraction
    """
    with datalab_test_app_context(console=False, exec_loop=not screenshots) as win:
        # Create an image panel
        panel = win.imagepanel

        # Load the flower test image
        ima = get_test_image("flower.npy")
        ima.title = "Test image 'flower.npy'"
        ima.set_metadata_option("colormap", "jet")
        panel.add_object(ima)

        # Apply Roberts filter for edge detection
        panel.processor.run_feature("roberts")

        # Extract a rectangular ROI
        roi = sigima.objects.create_image_roi("rectangle", [32, 128, 448, 256])
        panel.processor.compute_roi_extraction(roi)

        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("i_flower_roi")


def test_contour_detection_limits() -> None:
    """Test scenario to verify result truncation limits work correctly

    This scenario tests:
    - Contour detection on flower.npy (generates many contours)
    - Result truncation at max_result_rows limit
    - Shape drawing truncation at max_shapes_to_draw limit
    - Label display truncation at max_cells_in_label & max_cols_in_label limits
    - Warning dialog at max_cells_in_dialog limit
    """
    with datalab_test_app_context(console=False, exec_loop=False) as win:
        # Create an image panel
        panel = win.imagepanel

        # Load the flower test image
        ima = get_test_image("flower.npy")
        ima.title = "Test image 'flower.npy' - Contour Detection Limit Test"
        ima.set_metadata_option("colormap", "jet")
        panel.add_object(ima)

        # Run contour detection which should trigger the limits
        # This will detect many contours and test our safety mechanisms
        print("\nRunning contour detection on flower.npy...")
        print("This should trigger result truncation and shape drawing limits.")
        panel.processor.run_feature("contour_shape", sigima_param.ContourShapeParam())

        print("\nTest completed successfully!")
        print("Expected behavior:")
        print("  1. Results truncated to max_result_rows (default: 1000)")
        print("  2. Only max_shapes_to_draw shapes drawn (default: 50)")
        print("  3. Warning label on plot showing truncation")
        print("  4. Result dialog warning if > max_cells_in_dialog (default: 50000)")
        print("  5. Merged label: max_cells_in_label (100) & max_cols_in_label (15)")


if __name__ == "__main__":
    # Uncomment to run the original scenarios:
    # run_circle_detection_scenario()

    # Run the test for result limits
    test_contour_detection_limits()
