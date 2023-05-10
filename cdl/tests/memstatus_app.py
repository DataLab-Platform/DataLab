# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Memory status widget application test
"""

import numpy as np
import psutil

from cdl.config import Conf
from cdl.core.model.image import Gauss2DParam, ImageTypes, new_image_param
from cdl.tests import cdl_app_context

SHOW = True  # Show test in GUI-based test launcher


def test_memory_alarm(threshold):
    """Memory alarm test"""
    Conf.main.available_memory_threshold.set(threshold)
    rng = np.random.default_rng()
    with cdl_app_context() as win:
        panel = win.imagepanel
        newparam = new_image_param(itype=ImageTypes.GAUSS)
        addparam = Gauss2DParam()
        addparam.x0 = addparam.y0 = rng.integers(-9, 9)
        addparam.sigma = rng.integers(1, 20)
        panel.new_object(newparam, addparam=addparam, edit=False)


def test():
    """Memory alarm test"""
    test_memory_alarm(psutil.virtual_memory().available // (1024**2) * 2)
    test_memory_alarm(psutil.virtual_memory().available // (1024**2) - 100)
    Conf.main.available_memory_threshold.reset()


if __name__ == "__main__":
    test()
