# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for set_object proxy feature

This test verifies that modifying an object retrieved via get_object and pushing
it back via set_object correctly updates the object in DataLab.

In particular, it covers the case of objects with computed (read-only) DataSet
items such as ImageObj.xmin/xmax/ymin/ymax (see DataLab Issue #305).
"""

# guitest: show

from __future__ import annotations

from sigima.tests.data import get_test_image, get_test_signal

from datalab.tests import datalab_in_background_context


def test_set_object() -> None:
    """Test set_object on both signal (no computed items) and image (computed items)"""
    with datalab_in_background_context() as proxy:
        # ---- Signal: no computed items ----
        signal = get_test_signal("paracetamol.txt")
        proxy.add_object(signal)
        proxy.set_current_panel("signal")
        sig_uuid = proxy.get_object_uuids("signal")[0]

        sig = proxy.get_object(sig_uuid)
        original_title = sig.title
        sig.title = "Modified signal title"
        sig.yunit = "modified_unit"
        proxy.set_object(sig)

        sig_back = proxy.get_object(sig_uuid)
        assert sig_back.title == "Modified signal title"
        assert sig_back.yunit == "modified_unit"

        # Restore
        sig_back.title = original_title
        proxy.set_object(sig_back)

        # ---- Image: has computed items (xmin, xmax, ymin, ymax) ----
        # This is the case that triggers the bug reported in Issue #305:
        # iterating all _items and calling setattr on computed items raises
        # ValueError: Computed item 'xmin' is read-only
        image = get_test_image("flower.npy")
        proxy.add_object(image)
        proxy.set_current_panel("image")
        img_uuid = proxy.get_object_uuids("image")[0]

        img = proxy.get_object(img_uuid)
        original_title = img.title
        img.title = "Modified image title"
        img.dx = 0.5
        img.dy = 0.5
        img.x0 = 1.0
        img.y0 = 2.0
        # This call must not raise ValueError on computed items
        proxy.set_object(img)

        img_back = proxy.get_object(img_uuid)
        assert img_back.title == "Modified image title"
        assert img_back.dx == 0.5
        assert img_back.dy == 0.5
        assert img_back.x0 == 1.0
        assert img_back.y0 == 2.0

        # Restore
        img_back.title = original_title
        proxy.set_object(img_back)


if __name__ == "__main__":
    test_set_object()
