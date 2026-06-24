# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Rename legend refresh application test.

End-to-end validation that renaming an object updates its curve/image legend in
the plot (the legend is derived from the stored object title). Previously the
legend kept the old title until another refresh occurred.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import CosineParam, create_signal_from_param

from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_rename_updates_legend():
    """Renaming a signal updates its curve legend label in the plot."""
    with datalab_test_app_context() as win:
        panel = win.signalpanel
        s1 = create_signal_from_param(CosineParam.create(size=100))
        s1.title = "Original name"
        panel.add_object(s1)
        uuid = get_uuid(s1)

        # The curve legend is derived from the object title:
        panel.objview.select_objects([uuid])
        item = panel.plothandler[uuid]
        assert item.param.label == "Original name"

        # Renaming the object updates the legend immediately:
        panel.objview.set_current_item_id(uuid)
        panel.rename_selected_object_or_group("New name")
        assert s1.title == "New name"
        assert panel.plothandler[uuid].param.label == "New name"
