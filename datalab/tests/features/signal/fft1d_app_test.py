# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import CosineParam, create_signal_from_param

from datalab.tests import datalab_test_app_context


def test_fft1d_app():
    """FFT application test."""
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        s1 = create_signal_from_param(CosineParam.create(size=10000))
        panel.add_object(s1)
        panel.processor.run_feature("fft")
        panel.processor.run_feature("ifft")


if __name__ == "__main__":
    test_fft1d_app()
