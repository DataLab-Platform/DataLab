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
        s1 = get_test_signal("paracetamol.txt")
        peakidx = peak_indices(s1.y)
        s2 = create_noisy_signal(NormalDistribution1DParam.create(sigma=5.0))

        ep = execenv.print
        tn = get_default_test_name

        ep(fdlg.polynomialfit(s2.x, s2.y, 4, name=tn("00")))
        ep(fdlg.linearfit(s2.x, s2.y, name=tn("01")))
        ep(fdlg.gaussianfit(s2.x, s2.y, name=tn("02")))
        ep(fdlg.lorentzianfit(s2.x, s2.y, name=tn("03")))
        ep(fdlg.multigaussianfit(s1.x, s1.y, peakidx, name=tn("04")))
        ep(fdlg.multilorentzianfit(s1.x, s1.y, peakidx, name=tn("05")))
        ep(fdlg.voigtfit(s2.x, s2.y, name=tn("06")))
        ep(fdlg.exponentialfit(s2.x, s2.y, name=tn("07")))
        ep(fdlg.sinusoidalfit(s2.x, s2.y, name=tn("08")))
        ep(fdlg.cdffit(s2.x, s2.y, name=tn("09")))
        ep(fdlg.planckianfit(s2.x, s2.y, name=tn("10")))
        ep(fdlg.twohalfgaussianfit(s2.x, s2.y, name=tn("11")))
        ep(fdlg.doubleexponentialfit(s2.x, s2.y, name=tn("12")))


if __name__ == "__main__":
    test_fit_dialog()
