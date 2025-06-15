# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.tests import cdltest_app_context
from sigima_ import NewSignalParam, PeriodicParam, SignalTypes, create_signal_from_param


def test_fft1d_app():
    """FFT application test."""
    with cdltest_app_context() as win:
        panel = win.signalpanel
        newparam = NewSignalParam.create(stype=SignalTypes.COSINUS, size=10000)
        extra_param = PeriodicParam()
        s1 = create_signal_from_param(newparam, extra_param=extra_param)
        panel.add_object(s1)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("ifft")


if __name__ == "__main__":
    test_fft1d_app()
