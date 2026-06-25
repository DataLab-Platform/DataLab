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

from datalab.config import Conf
from datalab.objectmodel import get_short_id, get_uuid
from datalab.tests import datalab_test_app_context


def test_cross_panel_reference_view_follows_image_reorder():
    """Reordering images updates a cross-panel reference in a signal title/view."""
    with datalab_test_app_context() as win:
        orig_mode = Conf.proc.result_title_mode.get()
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

        # In short-ID mode, the signal panel view (object tree) reflects the
        # updated short reference:
        Conf.proc.result_title_mode.set("short_id")
        spanel.objview.update_tree()
        text = spanel.objview.get_item_from_id(get_uuid(sig)).text(0)
        assert "i002" in text
        assert "i001" not in text

        # In title mode, the cross-panel reference is rendered as the long name
        # of the (reordered) physical source image:
        Conf.proc.result_title_mode.set("title")
        spanel.objview.update_tree()
        text = spanel.objview.get_item_from_id(get_uuid(sig)).text(0)
        assert "First image" in text
        assert "i002" not in text

        Conf.proc.result_title_mode.set(orig_mode)


def test_cross_panel_reference_view_follows_image_delete():
    """Deleting a referenced image updates the cross-panel reference in the view."""
    with datalab_test_app_context() as win:
        orig_mode = Conf.proc.result_title_mode.get()
        ipanel, spanel = win.imagepanel, win.signalpanel

        # One image referenced by a signal:
        img = create_image("First image", np.zeros((8, 8)))
        ipanel.add_object(img)
        assert get_short_id(img) == "i001"
        sig = create_signal(
            "average profile(i001)", x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0]
        )
        spanel.add_object(sig)

        # Delete the source image (frozen into a stable deleted token):
        iview = ipanel.objview
        iview.select_objects([img])
        ipanel.remove_object(force=True)
        assert sig.title == "average profile(id001)"

        # In short-ID mode, the signal view shows the deleted token:
        Conf.proc.result_title_mode.set("short_id")
        spanel.objview.update_tree()
        text = spanel.objview.get_item_from_id(get_uuid(sig)).text(0)
        assert "id001" in text

        # In title mode, the signal view shows the deleted image's long name:
        Conf.proc.result_title_mode.set("title")
        spanel.objview.update_tree()
        text = spanel.objview.get_item_from_id(get_uuid(sig)).text(0)
        assert "First image" in text

        Conf.proc.result_title_mode.set(orig_mode)


def test_cross_panel_group_rename_refreshes_sibling_view():
    """Renaming an image group updates a cross-panel reference shown in the
    signal panel view, without any manual refresh."""
    with datalab_test_app_context() as win:
        orig_mode = Conf.proc.result_title_mode.get()
        ipanel, spanel = win.imagepanel, win.signalpanel

        # An image group containing one image -> gi001:
        igroup = ipanel.add_group("My images")
        img = create_image("First image", np.zeros((8, 8)))
        ipanel.add_object(img, group_id=get_uuid(igroup))
        assert get_short_id(igroup) == "gi001"

        # A signal group produced by a projection references the image group:
        sgroup = spanel.add_group(f"average profile({get_short_id(igroup)})")
        sig = create_signal(
            "average profile(i001)", x=[0.0, 1.0, 2.0], y=[1.0, 2.0, 3.0]
        )
        spanel.add_object(sig, group_id=get_uuid(sgroup))
        assert sgroup.title == "average profile(gi001)"

        # In title mode, the signal panel shows the image group's long name:
        Conf.proc.result_title_mode.set("title")
        spanel.objview.update_tree()
        text = spanel.objview.get_item_from_id(get_uuid(sgroup)).text(0)
        assert "My images" in text

        # Rename the image group in the image panel. The signal panel view must
        # reflect the new name *without* any manual refresh (the rename refreshes
        # the sibling panel):
        ipanel.objview.select_groups([igroup])
        ipanel.rename_selected_object_or_group("Renamed images")
        text = spanel.objview.get_item_from_id(get_uuid(sgroup)).text(0)
        assert "Renamed images" in text
        assert "My images" not in text

        Conf.proc.result_title_mode.set(orig_mode)
