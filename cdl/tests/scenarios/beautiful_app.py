# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

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
from cdl.tests import test_cdl_app_context


def run_beautiful_scenario(screenshots: bool = False):
    """High-level test scenario producing beautiful screenshots"""
    data_size = 500
    with test_cdl_app_context(console=False, exec_loop=not screenshots) as win:
        # Beautiful screenshot of a signal
        panel = win.signalpanel
        newparam = dlo.new_signal_param(stype=dlo.SignalTypes.LORENTZ)
        sig = dlo.create_signal_from_param(newparam)
        panel.add_object(sig)
        panel.processor.compute_fft()
        panel.processor.compute_wiener()
        panel.processor.compute_derivative()
        panel.processor.compute_integral()
        param = dlp.GaussianParam()
        panel.processor.compute_gaussian_filter(param)
        panel.processor.compute_fft()
        panel.processor.compute_derivative()
        if screenshots:
            win.take_screenshot("s_beautiful")
        # Beautiful screenshot of an image
        panel = win.imagepanel
        newparam = dlo.new_image_param(
            height=data_size, width=data_size, itype=dlo.ImageTypes.GAUSS
        )
        ima = dlo.create_image_from_param(newparam)
        panel.add_object(ima)
        panel.processor.compute_equalize_hist(dlp.EqualizeHistParam())
        panel.processor.compute_equalize_adapthist(dlp.EqualizeAdaptHistParam())
        panel.processor.compute_denoise_tv(dlp.DenoiseTVParam())
        panel.processor.compute_denoise_wavelet(dlp.DenoiseWaveletParam())
        panel.processor.compute_white_tophat(dlp.MorphologyParam())
        panel.processor.compute_denoise_tv(dlp.DenoiseTVParam())
        n = data_size // 5
        m = int(n * 1.25)
        panel.processor.compute_roi_extraction(
            dlp.ROIDataParam.create([[n, m, data_size - n, data_size - m]])
        )
        if screenshots:
            win.take_screenshot("i_beautiful")
            win.take_menu_screenshots()


if __name__ == "__main__":
    run_beautiful_scenario()
