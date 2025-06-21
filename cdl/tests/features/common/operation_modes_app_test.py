# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Operation modes test
--------------------

DataLab has two operation modes:

- **Single operand mode**: the operation is applied to the selected objects (this
  is the default mode)

- **Pairwise mode**: the operation is applied to the selected pairs of objects

This test scenario covers the pairwise mode and the operations that can be
performed in this mode: sum, difference, product, division, ...
"""

# guitest: show

from __future__ import annotations

from cdl import app
from cdl.config import Conf
from cdl.env import execenv
from cdl.gui.processor.base import is_pairwise_mode
from cdl.objectmodel import get_short_id
from cdl.utils.qthelpers import cdl_app_context
from cdl.utils.tests import get_test_fnames


def check_titles(title, titles):
    """Check that the title is one of the expected titles"""
    execenv.print(f"{title}:")
    for actual_title, expected_title in titles:
        execenv.print(f"  {actual_title} == {expected_title}", end=" ")
        assert actual_title == expected_title
        if actual_title == expected_title:
            execenv.print("✓")
        else:
            execenv.print("✗")


def test_single_operand_mode_compute_n1():
    """Run single operand mode test scenario
    with `compute_n_to_1` operation (e.g. sum)"""
    original_mode = Conf.proc.operation_mode.get()
    Conf.proc.operation_mode.set("single")

    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=[get_test_fnames("reorder*")[0]], console=False)
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Checking the operation mode:
        assert not is_pairwise_mode()

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        view.select_groups(groups)

        # Perform a sum operation
        panel.processor.run_feature("addition")

        # Default operation mode is single operand mode, so the sum operation
        # is applied to the selected groups, and we should have a new group
        # with two signals being the sum of signals from each group:
        # - signal 1: group 1 signal 1 + group 1 signal 2
        # - signal 2: group 2 signal 1 + group 2 signal 2
        assert len(model.get_groups()) == n_groups + 1
        new_group = model.get_group_from_number(n_groups + 1)
        assert len(new_group) == 2
        titles = []
        for idx, obj in enumerate(new_group):
            pfx_orig = ", ".join(get_short_id(obj) for obj in groups[idx].get_objects())
            titles.append((obj.title, f"Σ({pfx_orig})"))
        check_titles(f"Single operand mode Σ[{new_group.title}]", titles)

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
        panel.processor.run_feature("addition")

        # Default operation mode is single operand mode, so the sum operation
        # is applied to the selected signals, and we should have a new resulting
        # signal being the sum of the selected signals added in each group:
        # - signal added to group 1: group 1 signal 1 + group 2 signal 1
        # - signal added to group 2: group 1 signal 2 + group 2 signal 2
        assert len(model.get_groups()) == n_groups  # no new group
        titles = []
        for idx in range(1):
            pfx_orig = ", ".join(get_short_id(obj) for obj in groups[idx][:2])
            titles.append((groups[idx][-1].title, f"Σ({pfx_orig})"))
        check_titles(f"Single operand mode Σ[{groups[1].title}]", titles)

    Conf.proc.operation_mode.set(original_mode)


def test_pairwise_operations_mode_compute_n1():
    """Run pairwise operations mode test scenario
    with `compute_n_to_1` operation (e.g. sum)"""
    original_mode = Conf.proc.operation_mode.get()
    Conf.proc.operation_mode.set("pairwise")

    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=[get_test_fnames("reorder*")[0]], console=False)
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Checking the operation mode:
        assert is_pairwise_mode()

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        view.select_groups(groups)

        # Checking that each group contains the same number of signals (this is
        # required for pairwise operations - this part of the test is checking
        # if the data file is the one we expect)
        n_objects = len(groups[0])
        assert all(len(group) == n_objects for group in groups)

        # Perform a sum operation
        panel.processor.run_feature("addition")

        # Operation mode is now pairwise, so the sum operation is applied to the
        # selected groups, and we should have a new group with as many signals as
        # the original groups, each signal being the sum of the corresponding signals:
        # - signal 1: group 1 signal 1 + group 2 signal 1
        # - signal 2: group 1 signal 1 + group 2 signal 2
        # ...
        assert len(model.get_groups()) == n_groups + 1
        new_group = model.get_group_from_number(n_groups + 1)
        assert len(new_group.get_objects()) == n_objects
        titles = []
        for idx in range(len(groups[0])):
            obj = new_group[idx]
            pfx_orig = ", ".join(
                get_short_id(obj) for obj in (grp[idx] for grp in groups)
            )
            titles.append((obj.title, f"Σ({pfx_orig})"))
        check_titles(f"Pairwise operations mode Σ[{new_group.title}]", titles)

        # Remove new group
        view.select_groups([new_group])
        panel.remove_object(force=True)

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select two signals of the first two groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        objs = [groups[0][0]] + [groups[0][-1]] + groups[1][-2:]
        view.select_objects(objs)

        # Perform a sum operation
        panel.processor.run_feature("addition")

        # Operation mode is now pairwise, so the sum operation is applied to the
        # selected signals, and we should have a new group with as many signals as
        # the selected signals, each signal being the sum of the corresponding signals:
        # - signal 1: group 1 signal 1 + group 2 signal 1
        # - signal 2: group 1 signal 1 + group 2 signal 2
        # ...
        assert len(model.get_groups()) == n_groups + 1
        new_group = model.get_group_from_number(n_groups + 1)
        assert len(new_group) == 2  # 2 signals were selected
        titles = []
        for idx, obj in enumerate(new_group):
            pfx_orig = ", ".join(get_short_id(obj) for obj in objs[idx::2])
            titles.append((obj.title, f"Σ({pfx_orig})"))
        check_titles(f"Pairwise operations mode Σ[{new_group.title}]", titles)

    Conf.proc.operation_mode.set(original_mode)


def test_single_operand_mode_compute_n1n():
    """Run single operand mode test scenario
    with `compute_2_to_1` operation (e.g. difference)"""
    original_mode = Conf.proc.operation_mode.get()
    Conf.proc.operation_mode.set("single")

    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=[get_test_fnames("reorder*")[0]], console=False)
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Checking the operation mode:
        assert not is_pairwise_mode()

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        view.select_groups(groups)
        n_objects = [len(grp) for grp in groups]

        # Perform a difference operation with the first signal of the third group
        group3 = model.get_group_from_number(3)
        panel.processor.run_feature("difference", group3[0])

        # Default operation mode is single operand mode, so we should have new signals
        # in each selected group being the difference between the original signals and
        # the selected signal:
        # - in group 1:
        #   - signal 1: group 1 signal 1 - group 3 signal 1
        #   - signal 2: group 1 signal 2 - group 3 signal 1
        # - in group 2:
        #   - signal 1: group 2 signal 1 - group 3 signal 1
        #   - signal 2: group 2 signal 2 - group 3 signal 1
        assert len(model.get_groups()) == n_groups
        new_objs = []
        for i_group, group in enumerate(groups):
            titles = []
            for i_obj in range(n_objects[i_group]):
                obj = group[i_obj + n_objects[i_group]]
                titles.append(
                    (
                        obj.title,
                        f"{get_short_id(group[i_obj])}-{get_short_id(group3[0])}",
                    )
                )
                new_objs.append(obj)
            check_titles(f"Single operand mode Δ[{group.title}]", titles)

        # Remove new signals
        view.select_objects(new_objs)
        panel.remove_object(force=True)

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first signals of the first two groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        objs = groups[0][:2] + groups[1][:2]
        view.select_objects(objs)
        n_objects = [2, 2]

        # Perform a difference operation with the first signal of the third group
        panel.processor.run_feature("difference", group3[0])

        # Default operation mode is single operand mode, so we should have new signals
        # being the difference between the original signals and the selected signal:
        # - in group 1:
        #   - signal 1: group 1 signal 1 - group 3 signal 1
        #   - signal 2: group 1 signal 2 - group 3 signal 1
        # - in group 2:
        #   - signal 1: group 2 signal 1 - group 3 signal 1
        #   - signal 2: group 2 signal 2 - group 3 signal 1
        assert len(model.get_groups()) == n_groups  # no new group
        for i_group, group in enumerate(groups):
            titles = []
            for i_obj in range(n_objects[i_group]):
                obj = group[len(group) - n_objects[i_group] + i_obj]
                titles.append(
                    (
                        obj.title,
                        f"{get_short_id(group[i_obj])}-{get_short_id(group3[0])}",
                    )
                )
            check_titles(f"Single operand mode Δ[{group.title}]", titles)

    Conf.proc.operation_mode.set(original_mode)


def test_pairwise_operations_mode_compute_n1n():
    """Run pairwise operations mode test scenario
    with `compute_2_to_1` operation (e.g. difference)"""
    original_mode = Conf.proc.operation_mode.get()
    Conf.proc.operation_mode.set("pairwise")

    with cdl_app_context(exec_loop=True):
        win = app.create(h5files=[get_test_fnames("reorder*")[0]], console=False)
        panel = win.signalpanel
        view, model = panel.objview, panel.objmodel

        # Checking the operation mode:
        assert is_pairwise_mode()

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select the two first groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        view.select_groups(groups)

        # Checking that each group contains the same number of signals (this is
        # required for pairwise operations - this part of the test is checking
        # if the data file is the one we expect)
        n_objects = len(groups[0])
        assert all(len(group) == n_objects for group in groups)

        # Perform a difference operation with the third group
        group3 = model.get_group_from_number(3)
        assert len(group3) == n_objects
        panel.processor.run_feature("difference", group3.get_objects())

        # Operation mode is now pairwise, so the difference operation is applied to the
        # selected groups, and we should have a new group with as many signals as the
        # original groups, each signal being the difference of the corresponding
        # signals:
        # - signal 1: group 1 signal 1 - group 2 signal 1
        # - signal 2: group 1 signal 1 - group 2 signal 2
        # ...
        assert len(model.get_groups()) == n_groups + 2
        new_groups = [
            model.get_group_from_number(idx) for idx in (n_groups + 1, n_groups + 2)
        ]
        execenv.print("Δ|pairwise")
        for i_new_grp, new_grp in enumerate(new_groups):
            assert len(new_grp.get_objects()) == n_objects
            titles = []
            for idx in range(n_objects):
                obj = new_grp[idx]
                obj1, obj2 = groups[i_new_grp][idx], group3[idx]
                titles.append((obj.title, f"{get_short_id(obj1)}-{get_short_id(obj2)}"))
            check_titles(f"Pairwise operations mode Δ[{new_grp.title}]", titles)

        # Remove new groups
        view.select_groups(new_groups)
        panel.remove_object(force=True)

        # Store the number of groups before the operations
        n_groups = len(model.get_groups())

        # Select two signals of the first two groups
        groups = [model.get_group_from_number(idx) for idx in (1, 2)]
        objs = [groups[0][0]] + [groups[0][-1]] + groups[1][-2:]
        view.select_objects(objs)
        n_objects = 2

        # Perform a difference operation with two signals from the third group
        objs2 = group3[:2]
        panel.processor.run_feature("difference", objs2)

        # Operation mode is now pairwise, so the difference operation is applied to the
        # selected signals, and we should have a new group with as many signals as the
        # selected signals, each signal being the difference of the corresponding
        # signals:
        # - signal 1: group 1 signal 1 - group 3 signal 1
        # - signal 2: group 1 signal 1 - group 3 signal 2
        # ...
        assert len(model.get_groups()) == n_groups + 2
        new_groups = [
            model.get_group_from_number(idx) for idx in (n_groups + 1, n_groups + 2)
        ]
        i_obj1 = 0
        execenv.print("Δ|pairwise")
        for i_new_grp, new_grp in enumerate(new_groups):
            assert len(new_grp.get_objects()) == n_objects
            titles = []
            for idx in range(n_objects):
                obj = new_grp[idx]
                obj1, obj2 = objs[i_obj1], objs2[idx]
                i_obj1 += 1
                titles.append((obj.title, f"{get_short_id(obj1)}-{get_short_id(obj2)}"))
            check_titles(f"Pairwise operations mode Δ[{new_grp.title}]", titles)

    Conf.proc.operation_mode.set(original_mode)


if __name__ == "__main__":
    test_single_operand_mode_compute_n1()
    test_pairwise_operations_mode_compute_n1()
    test_single_operand_mode_compute_n1n()
    test_pairwise_operations_mode_compute_n1n()
