# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""Curve fitting dialog test

Testing fit dialogs: Gaussian, Lorentzian, Voigt, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from cdl.algorithms.signal import peak_indexes
from cdl.env import execenv
from cdl.tests.data import GaussianNoiseParam, create_noisy_signal, get_test_signal
from cdl.utils.qthelpers import qt_app_context
from cdl.utils.tests import get_default_test_name
from cdl.widgets import fitdialog as fdlg


def test():
    """Test function"""
    with qt_app_context():
        # Multi-gaussian curve fitting test
        s = get_test_signal("paracetamol.txt")
        peakidx = peak_indexes(s.y)
        execenv.print(
            fdlg.multigaussianfit(s.x, s.y, peakidx, name=get_default_test_name("00"))
        )

        # Gaussian curve fitting test
        noiseparam = GaussianNoiseParam()
        noiseparam.sigma = 5.0
        sig = create_noisy_signal(noiseparam)
        x, y = sig.x, sig.y
        execenv.print(fdlg.gaussianfit(x, y))

        # Lorentzian curve fitting test
        execenv.print(fdlg.lorentzianfit(x, y))

        # Voigt curve fitting test
        execenv.print(fdlg.voigtfit(x, y))

        # Polynomial curve fitting test
        execenv.print(fdlg.polynomialfit(x, y, 4))


if __name__ == "__main__":
    test()
