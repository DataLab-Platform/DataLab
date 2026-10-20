# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Features without menu entry are applied again from the Processing tab.

The results of extracting several ROIs into one object, and of DataLab-Web
features that DataLab Desktop has no menu for (image translation, signal peak
detection), must be recognized when they are applied again.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import sigima.proc.image as sipi
import sigima.proc.signal as sips
from guidata.qthelpers import qt_app_context
from sigima.objects import create_signal, create_signal_roi

from datalab.objectmodel import get_uuid
from datalab.processingcall import feature_for_function, resolve_processing_call
from datalab.tests import datalab_test_app_context


def test_extracted_rois_in_one_signal_are_applied_again() -> None:
    """Several ROIs extracted into one signal: Reapply is available and works."""
    with qt_app_context():
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            x = np.linspace(0.0, 10.0, 101)
            src = create_signal("S", x, np.sin(x))
            src.roi = create_signal_roi([[1.0, 3.0], [6.0, 8.0]])
            panel.add_object(src)
            params = src.roi.to_params(src)
            panel.processor._extract_multiple_roi_in_single_object(params)  # pylint: disable=protected-access
            result = panel.objview.get_current_object()
            assert result is not src
            call = resolve_processing_call(win, result)
            assert call is not None and call.can_reapply, call and call.reason
            expected = result.y.copy()
            report = panel.processor.recompute_processing(result, interactive=False)
            assert report.success, report.message
            assert np.array_equal(result.y, expected)
            assert get_uuid(win.signalpanel.objmodel[get_uuid(result)]) == get_uuid(
                result
            )


def test_features_of_other_editions_are_known() -> None:
    """Features recorded by DataLab-Web have a feature in Desktop."""
    with qt_app_context():
        with datalab_test_app_context() as win:
            for processor, function in (
                (win.signalpanel.processor, sips.extract_rois),
                (win.signalpanel.processor, sips.peak_detection),
                (win.imagepanel.processor, sipi.extract_rois),
                (win.imagepanel.processor, sipi.translate),
            ):
                assert feature_for_function(processor, function) is not None
