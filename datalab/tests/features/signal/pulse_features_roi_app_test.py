# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Pulse features with ROIs application test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

from sigima.objects import SignalObj, TableResult
from sigima.params import PulseFeaturesParam
from sigima.tests.signal.pulse.pulse_features_roi_unit_test import (
    check_results_equal,
    generate_source_signal,
)

from datalab.gui.panel.signal import SignalPanel
from datalab.tests import datalab_test_app_context


def __extract_pulse_features(panel: SignalPanel, obj: SignalObj) -> TableResult:
    """Extract pulse features."""
    panel.objview.select_objects([obj])
    param = PulseFeaturesParam()
    param.update_from_obj(obj)
    rdata = panel.processor.run_feature("extract_pulse_features", param)
    assert len(rdata.results) == 1
    return rdata.results[0]


def test_pulse_features_roi_app():
    """Pulse features with ROIs application test."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel

        # Test signal with multiple ROIs defined around peaks of the spectrum
        sig = generate_source_signal()
        panel.add_object(sig)
        pf_sig = __extract_pulse_features(panel, sig)

        # Extract ROIs in separate signals and test each one
        panel.processor.run_feature("extract_roi", params=sig.roi.to_params(sig))
        pf_extracted_sigs = []
        for nb_sig in (2, 3, 4):
            extracted_sig = panel.objmodel.get_object_from_number(nb_sig)
            pf_extracted_sig = __extract_pulse_features(panel, extracted_sig)
            pf_extracted_sigs.append(pf_extracted_sig)

        check_results_equal(pf_sig, pf_extracted_sigs)


if __name__ == "__main__":
    test_pulse_features_roi_app()
