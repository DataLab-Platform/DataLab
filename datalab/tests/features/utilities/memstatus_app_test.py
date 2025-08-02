# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Memory status widget application test
"""

# guitest: show

import numpy as np
import psutil

from datalab import config
from datalab.env import execenv
from datalab.tests import datalab_test_app_context
from sigima.objects import Gauss2DParam


def memory_alarm(threshold):
    """Memory alarm test"""
    config.Conf.main.available_memory_threshold.set(threshold)
    rng = np.random.default_rng()
    with datalab_test_app_context() as win:
        panel = win.imagepanel
        win.memorystatus.update_status()  # Force memory status update
        param = Gauss2DParam.create(
            x0=rng.integers(-9, 9), y0=rng.integers(-9, 9), sigma=rng.integers(1, 20)
        )
        panel.new_object(param, edit=False)


def test_mem_status():
    """Memory alarm test"""
    mem_available = psutil.virtual_memory().available // (1024**2)
    execenv.print(f"Memory status widget test (memory available: {mem_available} MB):")
    for index, threshold in enumerate((mem_available * 2, mem_available - 100)):
        execenv.print(f"    Threshold {index}: {threshold} MB")
        memory_alarm(threshold)
    config.reset()


if __name__ == "__main__":
    test_mem_status()
