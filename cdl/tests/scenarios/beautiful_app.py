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

import sigima_.param as sp
from cdl.tests import cdltest_app_context
from sigima_ import model


def run_beautiful_scenario(screenshots: bool = False):
    """High-level test scenario producing beautiful screenshots"""
    data_size = 500
    with cdltest_app_context(console=False, exec_loop=not screenshots) as win:
        # Beautiful screenshot of a signal
        panel = win.signalpanel
        base_param = model.NewSignalParam.create(stype=model.SignalTypes.LORENTZ)
        sig = model.create_signal_from_param(base_param, model.GaussLorentzVoigtParam())
        panel.add_object(sig)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("wiener")
        panel.processor.run_feature("derivative")
        panel.processor.run_feature("integral")
        param = sp.GaussianParam()
        panel.processor.run_feature("gaussian_filter", param)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("derivative")
        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_beautiful")
        # Beautiful screenshot of an image
        panel = win.imagepanel
        base_param = model.NewImageParam.create(
            height=data_size, width=data_size, itype=model.ImageTypes.GAUSS
        )
        ima = model.create_image_from_param(base_param, model.Gauss2DParam())
        ima.set_metadata_option("colormap", "jet")
        panel.add_object(ima)
        panel.processor.run_feature("equalize_hist", sp.EqualizeHistParam())
        panel.processor.run_feature("equalize_adapthist", sp.EqualizeAdaptHistParam())
        panel.processor.run_feature("denoise_tv", sp.DenoiseTVParam())
        panel.processor.run_feature("denoise_wavelet", sp.DenoiseWaveletParam())
        panel.processor.run_feature("white_tophat", sp.MorphologyParam())
        panel.processor.run_feature("denoise_tv", sp.DenoiseTVParam())
        n = data_size // 3
        roi = model.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        panel.processor.compute_roi_extraction(roi)
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


if __name__ == "__main__":
    run_beautiful_scenario()
