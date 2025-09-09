# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Curve fitting dialog test

Testing fit dialogs: Gaussian, Lorentzian, Voigt, etc.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import qt_app_context
from sigima.objects import NormalDistribution1DParam
from sigima.tests.data import create_noisy_signal, get_test_signal
from sigima.tests.helpers import get_default_test_name
from sigima.tools.signal.peakdetection import peak_indices

from datalab.env import execenv
from datalab.widgets import fitdialog as fdlg


def test_fit_dialog():
    """Test function"""
    with qt_app_context():
        # Multi-gaussian curve fitting test
        s = get_test_signal("paracetamol.txt")
        peakidx = peak_indices(s.y)
        execenv.print(
            fdlg.multigaussianfit(s.x, s.y, peakidx, name=get_default_test_name("00"))
        )

        noiseparam = NormalDistribution1DParam.create(sigma=5.0)
        sig = create_noisy_signal(noiseparam)
        x, y = sig.x, sig.y

        # Polynomial curve fitting test
        execenv.print(fdlg.polynomialfit(x, y, 4))

        # Linear curve fitting test
        execenv.print(fdlg.linearfit(x, y))

        # Gaussian curve fitting test
        execenv.print(fdlg.gaussianfit(x, y))

        # Lorentzian curve fitting test
        execenv.print(fdlg.lorentzianfit(x, y))

        # Multi-Lorentzian curve fitting test (needs peaks)
        execenv.print(fdlg.multigaussianfit(x, y, peakidx))

        # Multi-Lorentzian curve fitting test (needs peaks)
        execenv.print(fdlg.multilorentzianfit(x, y, peakidx))

        # Voigt curve fitting test
        execenv.print(fdlg.voigtfit(x, y))

        # Exponential curve fitting test
        execenv.print(fdlg.exponentialfit(x, y))

        # Sinusoidal curve fitting test
        execenv.print(fdlg.sinusoidalfit(x, y))

        # CDF curve fitting test
        execenv.print(fdlg.cdffit(x, y))

        # Planckian curve fitting test
        execenv.print(fdlg.planckianfit(x, y))

        # Two half-Gaussian curve fitting test
        execenv.print(fdlg.twohalfgaussianfit(x, y))

        # Double exponential curve fitting test
        execenv.print(fdlg.doubleexponentialfit(x, y))


if __name__ == "__main__":
    test_fit_dialog()
