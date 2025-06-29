# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Catch error and warning test (during computation)

Unit test for BaseProcessor.handle_output method: catching error and warning
during computation. This test runs a computation function that raises an
error and/or a warning, and checks that the error and/or warning are/is correctly
caught and displayed in the GUI.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

import warnings

from cdl.config import Conf
from cdl.env import execenv
from cdl.tests import cdltest_app_context
from sigima_.obj import SignalObj
from sigima_.tests import data as test_data


def comp_error(src: SignalObj) -> SignalObj:  # pylint: disable=unused-argument
    """Computation function that raises an error"""
    raise ValueError("This is a test error")


def comp_warning(src: SignalObj) -> SignalObj:  # pylint: disable=unused-argument
    """Computation function that raises a warning"""
    warnings.warn("This is a test warning")
    return src.copy()


def comp_warning_error(src: SignalObj) -> SignalObj:  # pylint: disable=unused-argument
    """Computation function that raises a warning and an error"""
    warnings.warn("This is a test warning")
    raise ValueError("This is a test error")


def comp_no_error(src: SignalObj) -> SignalObj:  # pylint: disable=unused-argument
    """Computation function that does not raise an error"""
    return src.copy()


def test_catcher():
    """Catch error and warning test"""
    with execenv.context(catcher_test=True):
        with cdltest_app_context() as win:
            panel = win.signalpanel
            sig = test_data.create_paracetamol_signal()
            panel.add_object(sig)
            panel.processor.compute_1_to_1(comp_no_error, title="Test no error")
            panel.processor.compute_1_to_1(comp_error, title="Test error")
            Conf.proc.ignore_warnings.set(True)
            panel.processor.compute_1_to_1(comp_warning, title="Test warning (ignored)")
            Conf.proc.ignore_warnings.set(False)
            panel.processor.compute_1_to_1(
                comp_warning, title="Test warning (not ignored)"
            )
            panel.processor.compute_1_to_1(
                comp_warning_error, title="Test warning + error"
            )


if __name__ == "__main__":
    test_catcher()
