# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Process isolation unit test
---------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

import time
from multiprocessing import Pool

import numpy as np
import scipy.signal as sps
from guidata.configtools import get_icon
from guidata.qthelpers import qt_app_context
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW

from cdl.adapters_plotpy.factories import create_adapter_from_object
from cdl.env import execenv
from sigima_.obj import NewImageParam
from sigima_.tests.data import create_2d_random, create_noisygauss_image


class MainWindow(QW.QMainWindow):
    """Multiprocessing test main window"""

    SIG_COMPUTATION_FINISHED = QC.Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("DataLab Multiprocessing test")
        self.setWindowIcon(get_icon("datalab.svg"))

        # Setting up the layout and widgets
        self.imagewidget = PlotWidget(options=PlotOptions(type="image"))
        self.setCentralWidget(self.imagewidget)
        self.layout = QW.QVBoxLayout(self.imagewidget)

        self.start_button = QW.QPushButton("Start Computation", self)
        self.start_button.clicked.connect(self.start_computation)
        self.layout.addWidget(self.start_button)

        self.cancel_button = QW.QPushButton("Cancel Computation", self)
        self.cancel_button.clicked.connect(self.cancel_computation)
        self.layout.addWidget(self.cancel_button)

        # Create a test image and add it to the plot
        param = NewImageParam.create(height=1000, width=1000)
        image = create_noisygauss_image(param, add_annotations=True)
        self.imageitem = create_adapter_from_object(image).make_item()
        self.imagewidget.plot.add_item(self.imageitem)

        self.array = image.data
        self.result = None
        self.timer = QC.QTimer()
        self.timer.setInterval(10)  # Check every 100 ms
        self.timer.timeout.connect(self.check_process)
        self.start_time = None

        self.SIG_COMPUTATION_FINISHED.connect(self.update_plot)

    @staticmethod
    def long_running_task(array: np.ndarray) -> None:
        """
        A long running task that computes a median filter on the input array and puts
        the result in the queue. The done_event is set when the computation is finished.

        Args:
            array (numpy.ndarray): The input data to compute the filter on.
        """
        start_time = time.time()
        result = sps.medfilt(array, 1) + create_2d_random(array.shape[0], array.dtype)
        execenv.print(f"Computation done: delta={time.time() - start_time:.3f} s")
        return result

    def print_time(self, title: str) -> None:
        """
        Prints the time since the last call to this method and resets the start_time.

        Args:
            title (str): The title to print before the time.
        """
        execenv.print(f"{title}: {time.time() - self.start_time:.3f} s")
        self.start_time = time.time()

    def start_computation(self) -> None:
        """
        Starts the computation in a separate process and starts the timer.
        """
        global POOL  # pylint: disable=global-statement,global-variable-not-assigned
        if self.result and not self.result.ready():
            self.print_time("Computation already running!")
            return
        self.start_time = time.time()
        self.result = POOL.apply_async(self.long_running_task, (self.array,))
        self.timer.start()
        self.print_time("Computation started")

    def cancel_computation(self) -> None:
        """
        Cancels the computation by terminating the process and stopping the timer.
        """
        global POOL  # pylint: disable=global-statement
        # Terminate the process and stop the timer
        POOL.terminate()
        POOL.join()
        self.timer.stop()
        execenv.print("Computation cancelled!")
        # Recreate the pool for the next computation
        POOL = Pool(processes=1)  # pylint: disable=not-callable,consider-using-with

    def check_process(self) -> None:
        """
        Checks if the computation is finished. If it is, stops the timer and calls
        the method to handle the computation finishing.
        """
        if self.result.ready():
            self.print_time("Computation finished")
            self.timer.stop()
            self.on_computation_finished()

    def on_computation_finished(self) -> None:
        """
        Retrieves the result from the queue
        and emits the SIG_COMPUTATION_FINISHED signal.
        """
        self.array = self.result.get()
        self.SIG_COMPUTATION_FINISHED.emit()
        self.print_time("Computation result retrieved")

    def update_plot(self) -> None:
        """
        Updates the plot with the result of the computation.
        """
        self.imageitem.set_data(self.array)
        self.imagewidget.plot.replot()
        self.print_time("Plot updated")

    def closeEvent(self, event) -> None:
        """
        Overrides the closeEvent to stop the timer
        and terminate the process when the window is closed.

        Args:
            event: The close event.
        """
        self.timer.stop()
        super().closeEvent(event)


def test_multiprocessing2() -> None:
    """
    Creates a PyQt application context, shows the main window,
    and starts the event loop.
    """
    global POOL  # pylint: disable=global-statement,global-variable-not-assigned
    with qt_app_context(exec_loop=True):
        window = MainWindow()
        window.show()
    POOL.terminate()
    POOL.join()


if __name__ == "__main__":
    POOL = Pool(processes=1)  # pylint: disable=not-callable,consider-using-with
    test_multiprocessing2()
