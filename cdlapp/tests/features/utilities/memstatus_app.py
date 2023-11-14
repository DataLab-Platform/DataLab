# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Memory status widget application test
"""

# guitest: show

import numpy as np
import psutil

from cdlapp.config import Conf
from cdlapp.env import execenv
from cdlapp.obj import Gauss2DParam, ImageTypes, new_image_param
from cdlapp.tests import test_cdl_app_context


def test_memory_alarm(threshold):
    """Memory alarm test"""
    Conf.main.available_memory_threshold.set(threshold)
    rng = np.random.default_rng()
    with test_cdl_app_context() as win:
        panel = win.imagepanel
        win.memorystatus.update_status()  # Force memory status update
        newparam = new_image_param(itype=ImageTypes.GAUSS)
        addparam = Gauss2DParam.create(
            x0=rng.integers(-9, 9), y0=rng.integers(-9, 9), sigma=rng.integers(1, 20)
        )
        panel.new_object(newparam, addparam=addparam, edit=False)


def test():
    """Memory alarm test"""
    mem_available = psutil.virtual_memory().available // (1024**2)
    execenv.print(f"Memory status widget test (memory available: {mem_available} MB):")
    for index, threshold in enumerate((mem_available * 2, mem_available - 100)):
        execenv.print(f"    Threshold {index}: {threshold} MB")
        test_memory_alarm(threshold)
    Conf.main.available_memory_threshold.reset()


if __name__ == "__main__":
    test()
