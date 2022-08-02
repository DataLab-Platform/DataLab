# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""Curve fitting dialog test

Testing the multi-Gaussian fit dialog.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...


from codraft.core.computation.signal import peak_indexes
from codraft.core.io.signal import read_signal
from codraft.env import execenv
from codraft.tests.data import get_test_fnames
from codraft.utils.qthelpers import qt_app_context
from codraft.utils.tests import get_default_test_name
from codraft.widgets.fitdialog import multigaussianfit

SHOW = True  # Show test in GUI-based test launcher


def test():
    """Test function"""
    with qt_app_context():
        s = read_signal(get_test_fnames("paracetamol.txt")[0])
        peakindexes = peak_indexes(s.y)
        execenv.print(
            multigaussianfit(s.x, s.y, peakindexes, name=get_default_test_name("00"))
        )


if __name__ == "__main__":
    test()
