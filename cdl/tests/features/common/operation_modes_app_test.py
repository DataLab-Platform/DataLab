# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation modes test
--------------------

"""

# guitest: show

from __future__ import annotations

from cdl import app
from cdl.utils.qthelpers import cdl_app_context
from cdl.utils.tests import get_test_fnames


def test_single_operand_mode():
    """Run single operand mode test scenario"""
    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=[get_test_fnames("reorder*")[0]], console=False)
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        view.select_groups(groups)

        # Perform a sum operation
        panel.processor.compute_sum()

        # Default operation mode is single operand mode, so the sum operation
        # is applied to the selected groups, and we should have a new group
        # with two signals being the sum of signals from each group:
        # - signal 1: group 1 signal 1 + group 1 signal 2
        # - signal 2: group 2 signal 1 + group 2 signal 2
        assert len(model.get_groups()) == n_groups + 1
        new_group = model.get_group_from_number(n_groups + 1)
        assert len(new_group.get_objects()) == 2
        for idx, obj in enumerate(new_group.get_objects()):
            pfx_orig = ", ".join(obj.short_id for obj in groups[idx].get_objects())
            assert obj.title == f"Σ({pfx_orig})"

        # Remove new group
        view.select_groups([new_group])
        panel.remove_object(force=True)

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first signals of the first two groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        objs = groups[0][:2] + groups[1][:2]
        view.select_objects(objs)

        # Perform a sum operation
        panel.processor.compute_sum()

        # Default operation mode is single operand mode, so the sum operation
        # is applied to the selected signals, and we should have a new resulting
        # signal being the sum of the selected signals added in each group:
        # - signal added to group 1: group 1 signal 1 + group 2 signal 1
        # - signal added to group 2: group 1 signal 2 + group 2 signal 2
        assert len(model.get_groups()) == n_groups  # no new group
        for idx in range(1):
            pfx_orig = ", ".join(obj.short_id for obj in groups[idx][:2])
            assert groups[idx][-1].title == f"Σ({pfx_orig})"

        # Removing resulting signals
        view.select_objects([groups[0][-1], groups[1][-1]])
        panel.remove_object()


if __name__ == "__main__":
    test_single_operand_mode()
