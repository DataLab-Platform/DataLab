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

import sigima.obj
import sigima.param as sigima_param

from cdl.tests import cdltest_app_context


def run_beautiful_scenario(screenshots: bool = False) -> None:
    """High-level test scenario producing beautiful screenshots"""
    data_size = 500
    with cdltest_app_context(console=False, exec_loop=not screenshots) as win:
        # Beautiful screenshot of a signal
        panel = win.signalpanel
        base_param = sigima.obj.NewSignalParam.create(
            stype=sigima.obj.SignalTypes.LORENTZ
        )
        sig = sigima.obj.create_signal_from_param(
            base_param, sigima.obj.GaussLorentzVoigtParam()
        )
        panel.add_object(sig)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("wiener")
        panel.processor.run_feature("derivative")
        panel.processor.run_feature("integral")
        param = sigima_param.GaussianParam()
        panel.processor.run_feature("gaussian_filter", param)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("derivative")
        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_beautiful")
        # Beautiful screenshot of an image
        panel = win.imagepanel
        base_param = sigima.obj.NewImageParam.create(
            height=data_size, width=data_size, itype=sigima.obj.ImageTypes.GAUSS
        )
        ima = sigima.obj.create_image_from_param(base_param, sigima.obj.Gauss2DParam())
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
        roi = sigima.obj.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        panel.processor.compute_roi_extraction(roi)
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


if __name__ == "__main__":
    run_beautiful_scenario()
