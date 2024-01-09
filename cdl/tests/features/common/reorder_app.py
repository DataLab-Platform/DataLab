# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Groups/signals/images reorder test:

    - Testing groups/signals reorder (images are not tested but the code is the same)
    - When executed in unattended mode, the test only covers the "move up" and
      "move down" actions
    - When executed in interactive mode, the user has to test the "drag and
      drop" actions manually
"""

# guitest: show

from cdl import app
from cdl.utils.qthelpers import cdl_app_context
from cdl.utils.tests import get_test_fnames


def test_reorder():
    """Run signals/images reorder test scenario"""
    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=get_test_fnames("reorder*"))
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Select multiple signals
        objs = [model.get_object_from_number(idx) for idx in (2, 4, 5, 9)]
        view.select_objects(objs)
        # Move up
        view.move_up()
        # Check that the order is correct (note: objects 4 and 5 are not affected
        # by the move up action because they are moved up from the top of their group
        # to the bottom of the previous group)
        assert [obj.number for obj in objs] == [1, 4, 5, 8]
        # Move down
        view.move_down()
        # Check that the order is correct (note: objects 4 and 5 are not affected
        # by the move down action because they are moved down from the bottom of their
        # group to the top of the next group)
        assert [obj.number for obj in objs] == [2, 4, 5, 9]

        # Select multiple groups
        groups = [model.get_group_from_number(idx) for idx in (2, 3)]
        view.select_groups(groups)
        # Move up
        view.move_up()
        assert [group.number for group in groups] == [1, 2]
        # Move down
        view.move_down()
        assert [group.number for group in groups] == [2, 3]


if __name__ == "__main__":
    test_reorder()
