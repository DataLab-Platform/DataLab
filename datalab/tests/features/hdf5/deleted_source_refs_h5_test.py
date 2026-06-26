# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Deleted source references HDF5 persistence test.

Validates that a group's "deleted source references" registry (token -> frozen
title) survives a workspace save/load round-trip, and that it is stored as an
HDF5 *attribute* (not a child group) so that older DataLab versions -- whose
deserializer only iterates child groups -- can still open the file.
"""

# guitest: show

import os.path as osp

import h5py
from sigima.tests.data import create_paracetamol_signal

from datalab.objectmodel import find_deleted_refs_in_title
from datalab.tests import datalab_test_app_context, helpers


def _find_attr_recursive(node, attr_name):
    """Return the first (node_name, value) where ``attr_name`` is an attribute,
    and the set of child group keys of that node, or ``(None, None, None)``."""
    if attr_name in node.attrs:
        return node.name, node.attrs[attr_name], set(node.keys())
    for key in node.keys():
        child = node[key]
        if isinstance(child, h5py.Group):
            result = _find_attr_recursive(child, attr_name)
            if result[0] is not None:
                return result
    return None, None, None


def test_group_deleted_refs_persist_and_forward_compatible():
    """A group registry survives save/load and is stored as an attribute."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            panel = win.signalpanel
            model = panel.objmodel

            # Group A (gs001) holding a signal:
            panel.add_object(create_paracetamol_signal())
            group_a = model.get_groups()[0]
            # Give the group a real (non-empty) title so the rendered reference
            # is meaningful (an empty title would fall back to the short token):
            group_a.title = "Group A"
            group_a_title = group_a.title

            # Group B whose title references group A (gs001):
            group_b = panel.add_group("merge(gs001)")
            group_b_uuid = group_b.uuid

            # Deleting group A freezes its reference into group B's registry:
            model.remove_group(group_a)
            assert group_b.title == "merge(gsd001)"
            assert model.get_deleted_refs(group_b) == {"gsd001": group_a_title}

            # === Save workspace
            fname = osp.join(tmpdir, "deleted_refs.h5")
            win.save_h5_workspace(fname)

            # === Forward-compatibility: the registry is an HDF5 attribute,
            # not a child group (older readers iterate child groups only).
            with h5py.File(fname, "r") as h5:
                node_name, value, child_keys = _find_attr_recursive(
                    h5, "deleted_source_refs"
                )
                assert node_name is not None, "registry attribute not found"
                assert "deleted_source_refs" not in child_keys, (
                    "registry must not be a child group (breaks old readers)"
                )
                assert isinstance(value, (str, bytes))

            # === Clear and reload
            for p in win.panels:
                p.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            # === Registry restored after reload
            loaded_b = next(
                g
                for g in win.signalpanel.objmodel.get_groups()
                if g.title == "merge(gsd001)"
            )
            assert loaded_b.uuid == group_b_uuid or loaded_b.title == "merge(gsd001)"
            refs = win.signalpanel.objmodel.get_deleted_refs(loaded_b)
            assert refs == {"gsd001": group_a_title}
            # The deleted-reference token still renders to the frozen name:
            assert find_deleted_refs_in_title(loaded_b.title) == [(6, 12, "gsd001")]
            assert (
                win.signalpanel.objmodel.get_display_title(loaded_b, use_titles=True)
                == f"merge({group_a_title})"
            )


if __name__ == "__main__":
    test_group_deleted_refs_persist_and_forward_compatible()
    print("Group deleted-source-reference persistence test passed.")
