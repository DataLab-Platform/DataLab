# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

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
from multiprocessing.pool import AsyncResult, Pool
from typing import Any

import numpy as np
import scipy.signal as sps
from guidata.qthelpers import qt_app_context
from plotpy.plot import PlotWindow
from qtpy import QtWidgets as QW
from sigima.obj import ImageDatatypes, NewImageParam
from sigima.tests.data import create_2d_random, create_noisygauss_image

from datalab.adapters_plotpy.factories import create_adapter_from_object
from datalab.env import execenv
from datalab.utils.qthelpers import create_progress_bar

POOL: Pool | None = None


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


def test_multiprocessing1(iterations: int = 4) -> None:
    """Multiprocessing test"""
    Worker.create_pool()
    with qt_app_context(exec_loop=True):
        win = PlotWindow(title="Multiprocessing test", icon="datalab.svg", toolbar=True)
        win.resize(800, 600)
        win.show()
        param = NewImageParam.create(
            height=1000, width=1000, dtype=ImageDatatypes.UINT16
        )
        image = create_noisygauss_image(param, add_annotations=True)
        win.get_plot().add_item(create_adapter_from_object(image).make_item())
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
                    win.get_plot().add_item(
                        create_adapter_from_object(image).make_item()
                    )
                else:
                    break
        worker.terminate_pool()


if __name__ == "__main__":
    test_multiprocessing1()
