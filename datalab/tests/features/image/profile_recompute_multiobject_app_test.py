# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Multi-object recompute - parameter independence regression test

Non-regression test for a bug found on ``develop`` while investigating issue
#322: when recomputing *several* selected objects at once through the manual
"Recompute" action, the previous code read the processing parameters from the
shared Processing-tab editor (``ObjectProp.processing_param_editor``). During
the multi-object loop that editor lagged behind the object actually being
recomputed (it is only rebuilt through the asynchronous selection-changed
signal). As a result, one object could be recomputed with another object's
parameters, and its stored metadata got overwritten.

On this branch the recompute logic was refactored so that the multi-object
loop calls ``processor.recompute_processing(obj)`` without a parameter
override, making each object fall back to *its own* stored parameters. This
test locks that behaviour in by recomputing two different profiles selected
together and checking that each one keeps its own selection.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import sigima.params
from guidata.qthelpers import qt_app_context
from sigima.tests.data import create_noisy_gaussian_image

from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_multiobject_recompute_parameter_independence():
    """Recomputing several selected profiles at once must recompute each one
    with its own stored parameters (no cross-contamination)."""
    with qt_app_context():
        with datalab_test_app_context() as win:
            image_panel = win.imagepanel
            signal_panel = win.signalpanel
            proc = image_panel.processor

            # Source image
            image = create_noisy_gaussian_image(
                center=(0.0, 0.0), add_annotations=False
            )
            image_panel.add_object(image)

            # First profile (selection A: horizontal band)
            coords_a = {
                "direction": "horizontal",
                "row1": 10,
                "col1": 10,
                "row2": 40,
                "col2": 60,
            }
            proc.compute_average_profile(
                sigima.params.AverageProfileParam.create(**coords_a)
            )
            profile_a = signal_panel.objview.get_current_object()
            assert profile_a is not None
            data_a = profile_a.y.copy()

            # Second profile (selection B: vertical band, different shape)
            coords_b = {
                "direction": "vertical",
                "row1": 200,
                "col1": 150,
                "row2": 400,
                "col2": 350,
            }
            proc.compute_average_profile(
                sigima.params.AverageProfileParam.create(**coords_b)
            )
            profile_b = signal_panel.objview.get_current_object()
            assert profile_b is not None
            data_b = profile_b.y.copy()

            # Sanity: the two profiles have different shapes (A: 51, B: 201)
            assert profile_a.y.shape != profile_b.y.shape

            # Modify the source image by a known offset so that recompute
            # produces a detectable, deterministic change and we can prove the
            # recompute actually ran.
            offset = 5.0
            image.data = image.data + offset
            image.invalidate_maskdata_cache()

            # Select BOTH profiles and recompute them together (the multi-object
            # path that used to leak one profile's parameters into the other).
            signal_panel.objview.select_objects([profile_a, profile_b])
            signal_panel.recompute_selected()

            # Each profile must keep its own selection: same shape as before,
            # and values shifted by the offset applied to the source image.
            assert profile_a.y.shape == data_a.shape
            assert profile_b.y.shape == data_b.shape
            assert np.allclose(profile_a.y, data_a + offset)
            assert np.allclose(profile_b.y, data_b + offset)

            # Stored parameters must still describe each profile's own selection.
            pp_a = extract_processing_parameters(profile_a)
            pp_b = extract_processing_parameters(profile_b)
            assert pp_a is not None and pp_a.param is not None
            assert pp_b is not None and pp_b.param is not None
            assert pp_a.param.direction == coords_a["direction"]
            assert (pp_a.param.col1, pp_a.param.row1) == (
                coords_a["col1"],
                coords_a["row1"],
            )
            assert pp_b.param.direction == coords_b["direction"]
            assert (pp_b.param.col1, pp_b.param.row1) == (
                coords_b["col1"],
                coords_b["row1"],
            )

            # And the two profiles must remain distinct objects.
            assert get_uuid(profile_a) != get_uuid(profile_b)


if __name__ == "__main__":
    test_multiobject_recompute_parameter_independence()
