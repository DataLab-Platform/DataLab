# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""NIfTI image import application tests: groups, batches, history and workspace."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sigima.client import utils as rpc_utils
from sigima.io import NiftiImageImportParam
from sigima.objects import create_image

from datalab.control.proxy import LocalProxy
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import select_tree_entry

nib = pytest.importorskip("nibabel")


def save_nifti(filename, shape: tuple[int, ...]) -> str:
    """Save a NIfTI file whose voxels are their linear index, in millimeters."""
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    img = nib.Nifti1Image(data, np.diag([0.5, 2.0, 3.0, 1.0]))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(filename))
    return str(filename)


def group_titles(panel) -> list[str]:
    """Return the titles of the groups of a panel."""
    return [group.title for group in panel.objmodel.get_groups()]


def test_mixed_batch_groups_volume_slices_and_replays(tmp_path) -> None:
    """Open 2D and 3D NIfTI files with a TIFF image, then replay the import."""
    image2d = save_nifti(tmp_path / "a.nii", (4, 3))
    volume = save_nifti(tmp_path / "b.nii.gz", (4, 3, 5))
    tiff = str(tmp_path / "c.tif")
    param = NiftiImageImportParam.create(first_slice=1, last_slice=3)

    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.imagepanel
        history.toggle_record_mode(True)
        panel.add_object(create_image("TIFF", np.ones((6, 8), dtype=np.uint8)))
        panel.save_to_files([tiff])
        panel.remove_all_objects()

        objects = panel.load_from_files([tiff, volume, image2d], param=param)

        assert [obj.title for obj in objects] == [
            "a.nii",
            "b.nii.gz [k=1]",
            "b.nii.gz [k=2]",
            "b.nii.gz [k=3]",
            "c.tif",
        ]
        assert group_titles(panel).count("b.nii.gz") == 1
        slices = objects[1:4]
        assert [obj.metadata["nifti"]["slice_index"] for obj in slices] == [1, 2, 3]
        assert (slices[0].dx, slices[0].dy, slices[0].xunit) == (0.5, 2.0, "mm")
        volume_data = np.arange(60).reshape(4, 3, 5)
        np.testing.assert_array_equal(slices[1].data, volume_data[:, :, 2].T)

        action = history[len(history)]
        assert action.method_name == "load_from_files"
        recorded = action.kwargs["import_params"]
        assert recorded[0] is None and recorded[2] is None
        assert (recorded[1].first_slice, recorded[1].last_slice) == (1, 3)

        panel.objview.select_objects([get_uuid(obj) for obj in objects])
        panel.remove_object(force=True)
        assert len(panel.objmodel) == 0

        select_tree_entry(history, action.uuid)
        with patch.object(NiftiImageImportParam, "edit") as edit:
            history.replay_restore_actions(restore_selection=False)
        edit.assert_not_called()
        titles = [panel.objmodel[uuid].title for uuid in action.output_uuids]
        assert titles == [obj.title for obj in objects]


def test_group_policy_and_objects_not_added(tmp_path) -> None:
    """Slices follow ``create_group``; objects not added create no group."""
    volume = save_nifti(tmp_path / "vol.nii", (4, 3, 5))
    param = NiftiImageImportParam.create(first_slice=0, last_slice=1)

    with datalab_test_app_context() as win:
        panel = win.imagepanel
        groups = len(panel.objmodel.get_groups())

        objects = panel.load_from_files([volume], param=param, add_objects=False)
        assert len(objects) == 2 and len(panel.objmodel) == 0
        assert len(panel.objmodel.get_groups()) == groups

        panel.load_from_files([volume], param=param, create_group=False)
        assert len(panel.objmodel) == 2
        assert "vol.nii" not in group_titles(panel)


def test_volume_without_parameters_fails_explicitly(tmp_path) -> None:
    """In unattended mode, a volume without parameters is not read."""
    image2d = save_nifti(tmp_path / "a.nii", (4, 3))
    volume = save_nifti(tmp_path / "b.nii", (4, 3, 5))

    with datalab_test_app_context() as win:
        panel = win.imagepanel
        with pytest.raises(ValueError, match="NiftiImageImportParam"):
            panel.load_from_files([volume])
        assert len(panel.objmodel) == 0

        objects = panel.load_from_files([image2d, volume], ignore_errors=True)
        assert [obj.title for obj in objects] == ["a.nii"]


def test_reading_cancellation_stops_the_batch(tmp_path) -> None:
    """Objects read before a cancellation are kept; the next files are skipped."""
    first = save_nifti(tmp_path / "a.nii", (4, 3, 5))
    second = save_nifti(tmp_path / "b.nii", (4, 3))
    param = NiftiImageImportParam.create(first_slice=0, last_slice=1)

    with datalab_test_app_context() as win:
        panel = win.imagepanel
        registry_read = panel.IO_REGISTRY.read

        def read_then_cancel(filename, worker, **kwargs):
            """Read the file, then cancel as if during the reading."""
            objects = registry_read(filename, None, **kwargs)
            worker.cancel()
            return objects

        with patch.object(panel.IO_REGISTRY, "read", side_effect=read_then_cancel):
            objects = panel.load_from_files([first, second], param=param)
        assert [obj.title for obj in objects] == ["a.nii [k=0]", "a.nii [k=1]"]
        assert len(panel.objmodel) == 2
        assert "a.nii" in group_titles(panel)


def test_failed_addition_removes_the_volume_group(tmp_path) -> None:
    """If adding the slices fails, their group is removed."""
    volume = save_nifti(tmp_path / "vol.nii", (4, 3, 5))
    param = NiftiImageImportParam.create(first_slice=0, last_slice=1)

    with datalab_test_app_context() as win:
        panel = win.imagepanel
        groups = len(panel.objmodel.get_groups())
        with (
            patch.object(panel, "_add_objects", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            panel.load_from_files([volume], param=param)
        assert len(panel.objmodel.get_groups()) == groups
        assert len(panel.objmodel) == 0


def test_proxy_directory_import_and_workspace(tmp_path) -> None:
    """Import a directory through the proxy and keep NIfTI metadata in HDF5."""
    folder = tmp_path / "data"
    folder.mkdir()
    save_nifti(folder / "vol.nii.gz", (4, 3, 5))
    param = NiftiImageImportParam.create(first_slice=2, last_slice=4)
    remote_param = rpc_utils.rpcjson_to_dataset(rpc_utils.dataset_to_rpcjson(param))
    assert isinstance(remote_param, NiftiImageImportParam)
    h5file = str(tmp_path / "workspace.h5")

    with datalab_test_app_context() as win:
        proxy = LocalProxy(win)
        proxy.set_current_panel("image")
        proxy.load_from_directory(str(folder), remote_param)
        panel = win.imagepanel
        objects = [panel.objmodel[uuid] for uuid in panel.objmodel.get_object_ids()]
        assert [obj.title for obj in objects] == [
            f"vol.nii.gz [k={k}]" for k in (2, 3, 4)
        ]
        expected = objects[0].metadata["nifti"]

        win.save_to_h5_file(h5file)
        panel.remove_all_objects()
        win.open_h5_files([h5file], import_all=True, reset_all=True)
        reloaded = panel.objmodel[panel.objmodel.get_object_ids()[0]]
        assert reloaded.metadata["nifti"] == expected
