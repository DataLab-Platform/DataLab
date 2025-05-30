# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal spectrum unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: show

from __future__ import annotations

from guidata.qthelpers import qt_app_context

import sigima.algorithms.signal as alg
from cdl.tests.data import get_test_signal
from cdl.utils.vistools import view_curves


def test_signal_magnitude_spectrum_interactive() -> None:
    """Interactive test of the magnitude spectrum of a signal."""
    with qt_app_context():
        obj = get_test_signal("dynamic_parameters.txt")
        x, y = obj.xydata
        xms, yms = alg.magnitude_spectrum(x, y, log_scale=True)
        view_curves(
            [(xms, yms)],
            title="Magnitude spectrum",
            xlabel="Frequency",
            ylabel="Magnitude",
        )


def test_signal_phase_spectrum_interactive() -> None:
    """Interactive test of the phase spectrum of a signal."""
    with qt_app_context():
        obj = get_test_signal("dynamic_parameters.txt")
        x, y = obj.xydata
        xps, yps = alg.phase_spectrum(x, y)
        view_curves(
            [(xps, yps)],
            title="Phase spectrum",
            xlabel="Frequency",
            ylabel="Phase",
        )


def test_signal_psd_interactive() -> None:
    """Interactive test of the power spectral density of a signal."""
    with qt_app_context():
        obj = get_test_signal("dynamic_parameters.txt")
        x, y = obj.xydata
        xpsd, ypsd = alg.psd(x, y, log_scale=True)
        view_curves(
            [(xpsd, ypsd)],
            title="Power spectral density",
            xlabel="Frequency",
            ylabel="Power",
        )


if __name__ == "__main__":
    test_signal_magnitude_spectrum_interactive()
    test_signal_phase_spectrum_interactive()
    test_signal_psd_interactive()
