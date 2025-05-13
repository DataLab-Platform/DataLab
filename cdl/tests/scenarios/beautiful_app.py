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

import cdl.obj as dlo
import cdl.param as dlp
from cdl.tests import cdltest_app_context


def run_beautiful_scenario(screenshots: bool = False):
    """High-level test scenario producing beautiful screenshots"""
    data_size = 500
    with cdltest_app_context(console=False, exec_loop=not screenshots) as win:
        # Beautiful screenshot of a signal
        panel = win.signalpanel
        newparam = dlo.new_signal_param(stype=dlo.SignalTypes.LORENTZ)
        sig = dlo.create_signal_from_param(newparam)
        panel.add_object(sig)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("wiener")
        panel.processor.run_feature("derivative")
        panel.processor.run_feature("integral")
        param = dlp.GaussianParam()
        panel.processor.run_feature("gaussian_filter", param)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("derivative")
        if screenshots:
            win.statusBar().hide()
            win.take_screenshot("s_beautiful")
        # Beautiful screenshot of an image
        panel = win.imagepanel
        newparam = dlo.new_image_param(
            height=data_size, width=data_size, itype=dlo.ImageTypes.GAUSS
        )
        ima = dlo.create_image_from_param(newparam)
        ima.metadata["colormap"] = "jet"
        panel.add_object(ima)
        panel.processor.run_feature("equalize_hist", dlp.EqualizeHistParam())
        panel.processor.run_feature("equalize_adapthist", dlp.EqualizeAdaptHistParam())
        panel.processor.run_feature("denoise_tv", dlp.DenoiseTVParam())
        panel.processor.run_feature("denoise_wavelet", dlp.DenoiseWaveletParam())
        panel.processor.run_feature("white_tophat", dlp.MorphologyParam())
        panel.processor.run_feature("denoise_tv", dlp.DenoiseTVParam())
        n = data_size // 3
        roi = dlo.create_image_roi(
            "rectangle", [n, n, data_size - 2 * n, data_size - 2 * n]
        )
        panel.processor.compute_roi_extraction(roi)
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


if __name__ == "__main__":
    run_beautiful_scenario()
