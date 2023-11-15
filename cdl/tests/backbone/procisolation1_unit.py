# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Process isolation unit test
---------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: skip

from __future__ import annotations

import time
from collections.abc import Callable
from multiprocessing import Pool
from multiprocessing.pool import AsyncResult
from typing import Any

import numpy as np
import scipy.signal as sps
from guidata.qthelpers import qt_app_context
from plotpy.plot import PlotWindow
from qtpy import QtWidgets as QW

import cdl.obj
from cdl.env import execenv
from cdl.tests.data import create_2d_random, create_noisygauss_image
from cdl.utils.qthelpers import create_progress_bar

POOL: Pool = None


class Worker:
    """Multiprocessing worker, to run long-running tasks in a separate process"""

    def __init__(self) -> None:
        self.asyncresult: AsyncResult = None
        self.result: Any = None

    @staticmethod
    def create_pool() -> None:
        """Create multiprocessing pool"""
        global POOL  # pylint: disable=global-statement
        # Create a pool with one process
        POOL = Pool(processes=1)  # pylint: disable=not-callable,consider-using-with

    @staticmethod
    def terminate_pool() -> None:
        """Terminate multiprocessing pool"""
        global POOL  # pylint: disable=global-statement
        if POOL is not None:
            POOL.terminate()
            POOL.join()
            POOL = None

    def run(self, func: Callable, args: tuple[Any]) -> None:
        """Run computation"""
        global POOL  # pylint: disable=global-statement,global-variable-not-assigned
        assert POOL is not None
        self.asyncresult = POOL.apply_async(func, args)

    def terminate(self) -> None:
        """Terminate worker"""
        # Terminate the process and stop the timer
        self.terminate_pool()
        execenv.print("Computation cancelled!")
        # Recreate the pool for the next computation
        self.create_pool()

    def is_computation_finished(self) -> bool:
        """Return True if computation is finished"""
        return self.asyncresult.ready()

    def get_result(self) -> Any:
        """Return computation result"""
        self.result = self.asyncresult.get()
        self.asyncresult = None
        return self.result


def testfunc(data: np.ndarray, size: int, error: bool) -> np.ndarray:
    """Test function"""
    if error:
        raise ValueError("Test error")
    return sps.medfilt(data, size) + create_2d_random(data.shape[0], data.dtype)


def test(iterations: int = 4) -> None:
    """Multiprocessing test"""
    Worker.create_pool()
    with qt_app_context(exec_loop=True):
        win = PlotWindow(title="Multiprocessing test", icon="datalab.svg", toolbar=True)
        win.resize(800, 600)
        win.show()
        param = cdl.obj.new_image_param(
            height=1000, width=1000, dtype=cdl.obj.ImageDatatypes.UINT16
        )
        image = create_noisygauss_image(param)
        win.get_plot().add_item(image.make_item())
        worker = Worker()
        with create_progress_bar(win, "Computing", max_=iterations) as progress:
            for index in range(iterations):
                progress.setValue(index)
                progress.setLabelText(f"Computing {index}")
                test_error = index == 2
                worker.run(testfunc, (image.data, 3, test_error))
                while not worker.is_computation_finished():
                    QW.QApplication.processEvents()
                    time.sleep(0.1)
                    if progress.wasCanceled():
                        worker.terminate()
                        break
                if worker.is_computation_finished():
                    try:
                        image.data = worker.get_result()
                    except Exception as exc:  # pylint: disable=broad-except
                        execenv.print(f"Intercepted exception: {exc}")
                    win.get_plot().add_item(image.make_item())
                else:
                    break
        worker.terminate_pool()


if __name__ == "__main__":
    test()
