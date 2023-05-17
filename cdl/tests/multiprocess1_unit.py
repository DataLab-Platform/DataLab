# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Multiprocessing unit test
-------------------------

Requires the external package `multiprocess`, which relies on `dill` for serialization.

Using third-party multiprocess module, for better versatility (dill is used for
serialization, instead of pickle).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import scipy.ndimage as spi
import scipy.signal as sps
from guiqwt.plot import ImageWindow
from multiprocess import Pool
from qtpy import QtWidgets as QW

from cdl.env import execenv
from cdl.tests.data import create_2d_random, create_test_image2
from cdl.utils.qthelpers import create_progress_bar, qt_app_context

SHOW = True  # Show test in GUI-based test launcher


class Worker:
    """Multiprocessing worker, to run long-running tasks in a separate process"""

    def __init__(self) -> None:
        self.asyncresult = None
        self.result = None

    def run(self, func: Callable, args: tuple[Any]) -> None:
        """Run computation"""
        global POOL
        assert self.asyncresult is None
        self.asyncresult = POOL.apply_async(func, args)

    def terminate(self) -> None:
        """Terminate worker"""
        global POOL
        # Terminate the process and stop the timer
        POOL.terminate()
        POOL.join()
        print("Computation cancelled!")
        # Recreate the pool for the next computation
        POOL = Pool(processes=1)

    def is_computation_finished(self) -> bool:
        """Return True if computation is finished"""
        return self.asyncresult.ready()

    def get_result(self) -> Any:
        """Return computation result"""
        self.result = self.asyncresult.get()
        self.asyncresult = None
        return self.result


def test(iterations: int = 4) -> None:
    """Multiprocessing test"""
    global POOL
    with qt_app_context(exec_loop=True):
        win = ImageWindow("Multiprocessing test", icon="datalab.svg", toolbar=True)
        win.resize(800, 600)
        win.show()
        image = create_test_image2(1000, np.uint16)
        win.get_plot().add_item(image.make_item())
        worker = Worker()
        with create_progress_bar(win, "Computing", max_=iterations) as progress:
            for index in range(iterations):
                progress.setValue(index)
                progress.setLabelText(f"Computing {index}")

                def func(data: np.ndarray, size: int) -> np.ndarray:
                    return sps.medfilt(data, size) + create_2d_random(
                        data.shape[0], data.dtype
                    )

                worker.run(func, (image.data, 3))
                while not worker.is_computation_finished():
                    QW.QApplication.processEvents()
                    time.sleep(0.1)
                    if progress.wasCanceled():
                        worker.terminate()
                        break
                if worker.is_computation_finished():
                    image.data = worker.get_result()
                    win.get_plot().add_item(image.make_item())
                else:
                    break
        POOL.terminate()
        POOL.join()


if __name__ == "__main__":
    POOL = Pool(processes=1)
    test()
