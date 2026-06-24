# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Cross-panel references application test.

End-to-end validation that a signal title referencing an image (cross-panel
short ID, e.g. ``i001``) follows the physical source when images are reordered:
both the stored title *and* the signal panel view (object tree) are updated, even
though the reorder happens in the image panel.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
from sigima.objects import create_image, create_signal

from datalab.objectmodel import get_short_id, get_uuid
from datalab.tests import datalab_test_app_context


def test_cross_panel_reference_view_follows_image_reorder():
    """Reordering images updates a cross-panel reference in a signal title/view."""
    with datalab_test_app_context() as win:
        ipanel, spanel = win.imagepanel, win.signalpanel

        # Two images in the image panel: img1 -> i001, img2 -> i002
        img1 = create_image("First image", np.zeros((8, 8)))
        img2 = create_image("Second image", np.ones((8, 8)))
        ipanel.add_object(img1)
        ipanel.add_object(img2)
        assert get_short_id(img1) == "i001"
        assert get_short_id(img2) == "i002"

        # A signal extracted from the first image keeps a reference to it ("i001"),
        # exactly as a profile-extraction result would:
        sig = create_signal(
            "average profile(i001)", x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0]
        )
        spanel.add_object(sig)
        assert sig.title == "average profile(i001)"

        # Reorder the images so that the first image (the physical source) moves
        # down and becomes i002:
        iview = ipanel.objview
        iview.select_objects([img1])
        iview.move_down()
        assert get_short_id(img1) == "i002"
        assert get_short_id(img2) == "i001"

        # Stored signal title follows the physical source:
        assert sig.title == "average profile(i002)"

        # The signal panel view (object tree) reflects the updated reference:
        text = spanel.objview.get_item_from_id(get_uuid(sig)).text(0)
        assert "i002" in text
        assert "i001" not in text
