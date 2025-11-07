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
import sigima.params
from sigima.tests.data import get_test_image

from datalab.config import Conf
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
        panel.processor.run_feature("gaussian_filter", sigima.params.GaussianParam())
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
        panel.processor.run_feature("equalize_hist", sigima.params.EqualizeHistParam())
        panel.processor.run_feature(
            "equalize_adapthist", sigima.params.EqualizeAdaptHistParam()
        )
        panel.processor.run_feature("denoise_tv", sigima.params.DenoiseTVParam())
        panel.processor.run_feature(
            "denoise_wavelet", sigima.params.DenoiseWaveletParam()
        )
        panel.processor.run_feature("white_tophat", sigima.params.MorphologyParam())
        panel.processor.run_feature("denoise_tv", sigima.params.DenoiseTVParam())
        n = data_size // 3
        roi = sigima.objects.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        panel.processor.compute_roi_extraction(roi)
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


def run_blob_detection_on_flower_image(screenshots: bool = False) -> None:
    """High-level test scenario for flower image with ROI extraction

    This scenario creates:
    - A flower test image
    - Roberts edge detection filter applied
    - A rectangular ROI extraction
    - A closing morphological filter to clean up the result
    - Blob detection using OpenCV algorithm
    """
    with datalab_test_app_context(console=False, exec_loop=not screenshots) as win:
        # Create an image panel
        panel = win.imagepanel

        # Load the flower test image
        ima = get_test_image("flower.npy")
        ima.title = "Test image 'flower.npy'"
        panel.add_object(ima)

        # Apply Roberts filter for edge detection
        panel.processor.run_feature("roberts")

        # Extract a rectangular ROI
        roi = sigima.objects.create_image_roi("rectangle", [32, 64, 448, 384])
        panel.processor.compute_roi_extraction(roi)

        # Apply a closing morphological filter to clean up the result
        closing_param = sigima.params.MorphologyParam.create(radius=10)
        panel.processor.run_feature("closing", closing_param)

        # Detect blobs using OpenCV algorithm
        param = sigima.params.BlobOpenCVParam()
        param.filter_by_color = False
        param.min_area = 400
        param.max_area = 1000
        param.filter_by_circularity = True
        param.min_circularity = 0.7
        with Conf.proc.show_result_dialog.temp(False):
            with Conf.view.show_result_label.temp(False):
                panel.processor.run_feature("blob_opencv", param)
                if screenshots:
                    win.statusBar().hide()
                    win.take_screenshot("i_blob_detection_flower")


if __name__ == "__main__":
    run_blob_detection_on_flower_image(screenshots=False)
