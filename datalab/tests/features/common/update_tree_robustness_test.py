# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Update tree robustness test
---------------------------

This test verifies that the object tree view handles edge cases gracefully,
specifically when the tree becomes out of sync with the model.

This is a regression test for a bug where update_tree() crashed with
AttributeError: 'NoneType' object has no attribute 'setText' when an
object existed in the model but had no corresponding tree item.

The fix makes update_tree() defensive by calling populate_tree() if
an item is not found.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

import numpy as np
import sigima.objects

from datalab.tests import datalab_test_app_context


def test_update_tree_with_model_tree_desync():
    """Test that update_tree handles model/tree desynchronization gracefully.

    This test simulates a scenario where an object exists in the model
    but doesn't have a corresponding tree item, which previously caused
    an AttributeError in update_tree().

    The fix ensures update_tree() detects this and calls populate_tree()
    to restore consistency.
    """
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")

        # Create and add a base image
        data = np.random.rand(100, 100).astype(np.float64)
        img1 = sigima.objects.create_image("Base Image", data)
        win.imagepanel.add_object(img1)

        # Create a second image and add it to the MODEL only (not to the tree)
        # This simulates a desync between model and tree
        data2 = np.random.rand(100, 100).astype(np.float64)
        img2 = sigima.objects.create_image("Second Image", data2)

        # Get current group ID
        group_id = win.imagepanel.objview.get_current_group_id()

        # Add to model ONLY (bypassing add_object_item which adds to tree)
        win.imagepanel.objmodel.add_object(img2, group_id)

        # Now the model has 2 objects but the tree only shows 1
        # Calling update_tree() should NOT crash
        # With the fix, it should call populate_tree() to resync

        # This is the call that previously caused:
        # AttributeError: 'NoneType' object has no attribute 'setText'
        win.imagepanel.objview.update_tree()

        # Verify objects are now in sync
        assert len(win.imagepanel) == 2
