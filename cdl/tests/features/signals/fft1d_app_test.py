# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.obj import SignalTypes, create_signal_from_param, new_signal_param
from cdl.tests import cdltest_app_context


def test_fft1d_app():
    """FFT application test."""
    with cdltest_app_context() as win:
        panel = win.signalpanel
        newparam = new_signal_param(stype=SignalTypes.COSINUS, size=10000)
        s1 = create_signal_from_param(newparam)
        panel.add_object(s1)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("ifft")


if __name__ == "__main__":
    test_fft1d_app()
