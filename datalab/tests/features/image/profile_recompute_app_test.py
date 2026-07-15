# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction - recompute independence test

Companion to `profile_reset_selection_unit_test.py` (issue #322).

The fix for issue #322 resets the profile parameters *in place* instead of
replacing `self.param` with a new instance. This test verifies that this
in-place reset does not introduce any cross-contamination at recompute time:
each extracted profile must keep its own parameters, so that recomputing an
older profile regenerates *its* original selection, independently of any
profile extracted afterwards.

This holds because the parameters committed for each profile are serialized to
JSON in the result object's metadata (i.e. an independent snapshot), and are
deserialized into a fresh dataset when recomputing - there is no shared
reference with the live `param` object mutated by the dialog.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import sigima.params
from sigima.tests.data import create_noisy_gaussian_image

from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_profile_recompute_independence():
    """Regression test for issue #322 (recompute side): two profiles extracted
    from the same image must keep independent parameters, and recomputing one
    must regenerate its own selection."""
    with datalab_test_app_context() as win:
        image_panel = win.imagepanel
        signal_panel = win.signalpanel
        proc = image_panel.processor

        # Source image
        image = create_noisy_gaussian_image(center=(0.0, 0.0), add_annotations=False)
        image_panel.add_object(image)

        # First profile (selection A)
        coords_a = dict(direction="horizontal", row1=10, col1=10, row2=40, col2=60)
        proc.compute_average_profile(
            sigima.params.AverageProfileParam.create(**coords_a)
        )
        profile_a = signal_panel.objview.get_current_object()
        assert profile_a is not None
        data_a = profile_a.y.copy()

        # Second profile (selection B, different region)
        coords_b = dict(direction="vertical", row1=200, col1=150, row2=400, col2=350)
        proc.compute_average_profile(
            sigima.params.AverageProfileParam.create(**coords_b)
        )
        profile_b = signal_panel.objview.get_current_object()
        assert profile_b is not None
        data_b = profile_b.y.copy()

        # The two profiles must be different objects with different data
        assert get_uuid(profile_a) != get_uuid(profile_b)
        assert profile_a.y.shape != profile_b.y.shape or not np.allclose(
            profile_a.y, profile_b.y
        )

        # Each profile must store its *own* selection parameters (JSON snapshot)
        pp_a = extract_processing_parameters(profile_a)
        pp_b = extract_processing_parameters(profile_b)
        assert pp_a is not None and pp_a.param is not None
        assert pp_b is not None and pp_b.param is not None
        assert (pp_a.param.col1, pp_a.param.row1) == (
            coords_a["col1"],
            coords_a["row1"],
        )
        assert (pp_b.param.col1, pp_b.param.row1) == (
            coords_b["col1"],
            coords_b["row1"],
        )

        # Recomputing each profile (through DataLab's recompute engine, using the
        # parameters restored from its own metadata snapshot) must regenerate its
        # own selection - not the other profile's. This proves there is no shared
        # reference between the two stored parameter sets, so the in-place reset
        # performed by the profile dialog (fix for issue #322) cannot leak one
        # selection into another profile at recompute time.
        new_a = proc.recompute_1_to_1(pp_a.func_name, image, pp_a.param)
        new_b = proc.recompute_1_to_1(pp_b.func_name, image, pp_b.param)
        assert new_a is not None and new_b is not None
        assert np.allclose(new_a.y, data_a)
        assert np.allclose(new_b.y, data_b)


if __name__ == "__main__":
    test_profile_recompute_independence()
