# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Profile extraction dialog - "Reset selection" regression test

Reproduces https://github.com/DataLab-Platform/DataLab/issues/322:
after clicking "Reset selection" and drawing a new region, the extracted
profile must correspond to the newly drawn selection, not to the previous
(discarded) one.

Root cause: `ProfileExtractionDialog.reset_to_initial` replaces `self.param`
with a brand new parameter instance instead of resetting the existing one
in place. Callers (e.g. `ImageProcessor.compute_average_profile`) keep their
own reference to the original `param` object and pass *that* object to
`run_feature` once the dialog is closed. Once `self.param` has been
reassigned, further updates made by `shape_to_param()` (triggered when the
new selection is drawn) are applied to the new object, not to the one the
caller still holds, so the caller ends up with the first selection's
coordinates.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import sigima.params
from guidata.qthelpers import qt_app_context
from qtpy import QtCore as QC
from sigima.tests.data import create_noisy_gaussian_image

from datalab.gui.profiledialog import ProfileExtractionDialog


def _simulate_draw(
    dialog: ProfileExtractionDialog, x1: float, y1: float, x2: float, y2: float
) -> None:
    """Simulate an interactive rectangular selection on the profile
    extraction dialog's plot.

    A degenerate shape is first created through the underlying plotpy tool
    (as production code does for the "initial shape", see
    `ProfileExtractionDialog.set_obj`), then its geometry is set directly in
    plot (physical) coordinates -- this avoids relying on the canvas pixel
    <-> axis transform, which is only established once the dialog widget has
    actually been shown/resized. Finally, the dialog is notified through the
    ``SIG_TOOL_JOB_FINISHED`` signal, exactly as it would be when the user
    releases the mouse button after drawing a shape (see
    `RectangularActionTool.end_rect`).

    Args:
        dialog: Profile extraction dialog
        x1, y1, x2, y2: Rectangle coordinates (plot/physical coordinates)
    """
    plot = dialog.get_plot()
    zero = QC.QPointF(0.0, 0.0)
    dialog.cstool.add_shape_to_plot(plot, zero, zero)
    shape = dialog.cstool.get_last_final_shape()
    shape.set_rect(x1, y1, x2, y2)
    dialog.cstool.SIG_TOOL_JOB_FINISHED.emit()


def test_profile_reset_selection_unit():
    """Regression test for issue #322: "Reset selection" must not break the
    reference to the `param` object held by the caller."""
    with qt_app_context():
        obj = create_noisy_gaussian_image(center=(0.0, 0.0), add_annotations=False)

        # This is the very same object the calling code (e.g.
        # ImageProcessor.compute_average_profile) keeps a reference to and
        # passes to `run_feature` after the dialog is closed.
        param = sigima.params.AverageProfileParam()
        dialog = ProfileExtractionDialog("rectangle", param, add_initial_shape=False)
        dialog.set_obj(obj)

        # The test image is a 2000x2000 pixel array with unit spacing and
        # origin at (0, 0), so physical coordinates map 1:1 to pixel indices:
        # each drawn rectangle is committed to `param` with identical values.

        # First user selection
        _simulate_draw(dialog, 10, 10, 50, 50)
        first_coords = (param.col1, param.row1, param.col2, param.row2)
        assert first_coords == (10, 10, 50, 50)

        # User clicks "Reset selection"
        dialog.reset_to_initial()
        assert dialog.param is param, (
            "reset_to_initial() replaced 'self.param' with a new instance, "
            "breaking the reference held by the caller (see issue #322)"
        )

        # Second, different user selection
        _simulate_draw(dialog, 100, 100, 150, 150)

        # User clicks "OK"
        dialog.accept()

        # The caller's `param` object must reflect the *second* selection: the
        # committed coordinates must match the newly drawn rectangle, not the
        # first (discarded) one (issue #322).
        second_coords = (param.col1, param.row1, param.col2, param.row2)
        assert second_coords == (100, 100, 150, 150), (
            f"'param' holds {second_coords} instead of the second selection "
            "(100, 100, 150, 150) after Reset selection + new draw (issue #322)"
        )
        assert second_coords != first_coords, (
            "'param' still holds the first selection's coordinates "
            f"{first_coords} after Reset selection + new draw (issue #322)"
        )


if __name__ == "__main__":
    test_profile_reset_selection_unit()
