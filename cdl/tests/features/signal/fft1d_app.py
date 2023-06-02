# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal FFT application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


from cdl.obj import SignalTypes, create_signal_from_param, new_signal_param
from cdl.tests import cdl_app_context

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Signal peak dialog test"""
    with cdl_app_context() as win:
        panel = win.signalpanel
        newparam = new_signal_param(stype=SignalTypes.COSINUS, size=10000)
        s1 = create_signal_from_param(newparam)
        panel.add_object(s1)
        panel.processor.compute_fft()
        panel.processor.compute_ifft()


if __name__ == "__main__":
    test()
