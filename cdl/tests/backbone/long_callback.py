# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Long callback test
------------------

This test is not meant to be executed as part of the █`pytest` suite, hence the
name of the script.
"""

# guitest: show,skip

import time

from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

from cdl.utils.qthelpers import CallbackWorker, qt_long_callback


def long_computation_func(worker: CallbackWorker, delay: float) -> str:
    """Simulate long computation"""
    time.sleep(delay)
    return "OK"


def long_computation_progress_func(worker: CallbackWorker, delay: float) -> str:
    """Simulate long computation, with progress"""
    step_delay = 0.1
    maxiter = int(delay / step_delay)
    for idx in range(maxiter):
        worker.set_progress(int(100 * idx / maxiter))
        time.sleep(step_delay)
        if worker.was_canceled():
            return f"Interrupted at iteration #{idx}"
    return "Done"


class TestWindow(QW.QMainWindow):
    """Test window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test window")
        btn1 = QW.QPushButton("Run long computation")
        btn1.clicked.connect(self.run_long_computation)
        btn2 = QW.QPushButton("Run long computation with progress bar")
        btn2.clicked.connect(self.run_long_computation_with_progress)
        mainwidget = QW.QWidget()
        layout = QW.QHBoxLayout()
        mainwidget.setLayout(layout)
        layout.addWidget(btn1)
        layout.addWidget(btn2)
        self.setCentralWidget(mainwidget)

    def __execute_worker(self, worker: CallbackWorker, progress_max: int) -> None:
        """Execute worker"""
        ret = qt_long_callback(self, "Doing stuff...", worker, progress_max)
        QW.QMessageBox.information(self, "Result", f"Long computation result: {ret}")

    def run_long_computation(self):
        """Run long computation"""
        worker = CallbackWorker(long_computation_func, delay=5.0)
        self.__execute_worker(worker, 0)

    def run_long_computation_with_progress(self):
        """Run long computation with progress"""
        worker = CallbackWorker(long_computation_progress_func, delay=5.0)
        self.__execute_worker(worker, 100)


def busy_bar():
    """Busy bar test"""
    with qt_app_context(exec_loop=True):
        win = TestWindow()
        win.resize(800, 600)
        win.show()


if __name__ == "__main__":
    busy_bar()
