# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
A beautiful image obtained with high-level test scenario
"""

import cdl.obj
import cdl.param
from cdl.tests import cdl_app_context

SHOW = False  # Show test in GUI-based test launcher


def test():
    """Dictionnary/List in metadata (de)serialization test"""
    data_size = 500
    with cdl_app_context(console=False) as win:
        panel = win.imagepanel
        newparam = cdl.obj.NewImageParam()
        newparam.width = newparam.height = data_size
        newparam.type = cdl.obj.ImageTypes.GAUSS
        image = cdl.obj.create_image_from_param(newparam)
        panel.add_object(image)
        panel.processor.compute_equalize_hist(cdl.param.EqualizeHistParam())
        panel.processor.compute_equalize_adapthist(cdl.param.EqualizeAdaptHistParam())
        panel.processor.compute_denoise_tv(cdl.param.DenoiseTVParam())
        panel.processor.compute_denoise_wavelet(cdl.param.DenoiseWaveletParam())
        panel.processor.compute_white_tophat(cdl.param.MorphologyParam())
        panel.processor.compute_denoise_tv(cdl.param.DenoiseTVParam())
        n = data_size // 5
        m = int(n * 1.25)
        panel.processor.extract_roi([[n, m, data_size - n, data_size - m]])
        win.take_screenshot("i_beautiful")


if __name__ == "__main__":
    test()
