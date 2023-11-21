# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Memory status widget application test
"""

# guitest: show

import numpy as np
import psutil

from cdl.config import Conf
from cdl.env import execenv
from cdl.obj import Gauss2DParam, ImageTypes, new_image_param
from cdl.tests import cdltest_app_context


def memory_alarm(threshold):
    """Memory alarm test"""
    Conf.main.available_memory_threshold.set(threshold)
    rng = np.random.default_rng()
    with cdltest_app_context() as win:
        panel = win.imagepanel
        win.memorystatus.update_status()  # Force memory status update
        newparam = new_image_param(itype=ImageTypes.GAUSS)
        addparam = Gauss2DParam.create(
            x0=rng.integers(-9, 9), y0=rng.integers(-9, 9), sigma=rng.integers(1, 20)
        )
        panel.new_object(newparam, addparam=addparam, edit=False)


def test_mem_status():
    """Memory alarm test"""
    mem_available = psutil.virtual_memory().available // (1024**2)
    execenv.print(f"Memory status widget test (memory available: {mem_available} MB):")
    for index, threshold in enumerate((mem_available * 2, mem_available - 100)):
        execenv.print(f"    Threshold {index}: {threshold} MB")
        memory_alarm(threshold)
    Conf.main.available_memory_threshold.reset()


if __name__ == "__main__":
    test_mem_status()
