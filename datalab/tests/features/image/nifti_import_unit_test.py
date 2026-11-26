# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""NIfTI image import unit tests: slice selection dialog and import parameters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sigima.io import NiftiImageImportParam, RawImageImportParam, get_nifti_image_info

from datalab.gui.panel import image as panel_image
from datalab.gui.panel.image import (
    NIFTI_INVALID_SELECTION,
    ImagePanel,
    create_nifti_import_gui_param,
    format_memory_size,
    get_nifti_import_param,
    get_nifti_selection_summary,
    is_nifti_image_file,
)

nib = pytest.importorskip("nibabel")


def save_nifti(filename, shape: tuple[int, ...], dtype=np.int16) -> str:
    """Save a NIfTI file whose voxels are their linear index."""
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(filename))
    return str(filename)


def make_panel(**methods) -> SimpleNamespace:
    """Return a panel stub on which ``ImagePanel`` methods may be called."""
    panel = SimpleNamespace(parentWidget=lambda: None, **methods)
    panel.edit_nifti_import_parameters = lambda files: (
        ImagePanel.edit_nifti_import_parameters(panel, files)
    )
    return panel


@pytest.mark.parametrize(
    "nbytes, expected",
    [
        (512, "512 B"),
        (1536, "1.5 KiB"),
        (3 * 1024**2, "3.0 MiB"),
        (2**40, "1024.0 GiB"),
    ],
)
def test_format_memory_size(nbytes: int, expected: str) -> None:
    """Memory sizes are shown with binary units."""
    assert format_memory_size(nbytes) == expected


def test_nifti_files_are_recognized(tmp_path) -> None:
    """NIfTI files are recognized from their suffix, whatever its case."""
    assert is_nifti_image_file(str(tmp_path / "a.nii"))
    assert is_nifti_image_file(str(tmp_path / "a.NII.GZ"))
    assert not is_nifti_image_file(str(tmp_path / "a.gz"))
    assert not is_nifti_image_file(str(tmp_path / "a.raw"))
    assert not is_nifti_image_file(str(tmp_path / "a.unknown"))


def test_slice_selection_dialog(tmp_path) -> None:
    """The dialog preselects the central slice and validates the selection."""
    info = get_nifti_image_info(save_nifti(tmp_path / "vol.nii.gz", (4, 3, 5)))
    param = create_nifti_import_gui_param(info)

    assert (param.first_slice, param.last_slice) == (2, 2)
    assert "vol.nii.gz" in param.get_title()
    assert "4 × 3 pixels and 5 slices" in param.get_comment()
    assert param.selection == get_nifti_selection_summary(info, 2, 2)
    assert "1 slice(s) selected (about 24 B in memory)" in param.selection

    selection_item = next(
        item for item in param.get_items() if item.get_name() == "selection"
    )
    invalid = get_nifti_selection_summary(info, 3, 1)
    assert invalid.startswith(NIFTI_INVALID_SELECTION)
    assert not selection_item.check_value(invalid)
    assert selection_item.check_value(get_nifti_selection_summary(info, 0, 4))
    assert get_nifti_selection_summary(info, 0, 5).startswith(NIFTI_INVALID_SELECTION)

    param.first_slice, param.last_slice = 1, 3
    result = get_nifti_import_param(param)
    assert isinstance(result, NiftiImageImportParam)
    assert (result.first_slice, result.last_slice) == (1, 3)


def test_volume_dialogs_are_shown_once_per_volume(tmp_path) -> None:
    """Each volume has its own dialog; 2D images and cancelled volumes have none."""
    image2d = save_nifti(tmp_path / "a.nii", (4, 3))
    volume1 = save_nifti(tmp_path / "b.nii", (4, 3, 5))
    volume2 = save_nifti(tmp_path / "c.nii.gz", (4, 3, 6))
    tiff = str(tmp_path / "d.tif")
    shown: list[str] = []

    def create_param(info):
        param = create_nifti_import_gui_param(info)
        shown.append(info.filename)

        def edit(parent=None):
            del parent
            param.first_slice, param.last_slice = 0, 1
            return info.filename == volume1  # The second dialog is cancelled

        param.edit = edit
        return param

    panel = make_panel()
    with (
        patch.object(panel_image, "execenv", SimpleNamespace(unattended=False)),
        patch.object(panel_image, "create_nifti_import_gui_param", create_param),
    ):
        pairs = ImagePanel.edit_import_parameters(
            panel, [image2d, volume1, volume2, tiff]
        )

    assert shown == [volume1, volume2]
    assert [fname for fname, _param in pairs] == [image2d, volume1, tiff]
    assert pairs[0][1] is None and pairs[2][1] is None
    assert (pairs[1][1].first_slice, pairs[1][1].last_slice) == (0, 1)


def test_unattended_volumes_get_no_parameters(tmp_path) -> None:
    """Without a dialog, volumes get no parameters: reading them then fails."""
    volume = save_nifti(tmp_path / "vol.nii", (4, 3, 5))
    panel = make_panel()
    with (
        patch.object(panel_image, "execenv", SimpleNamespace(unattended=True)),
        patch.object(panel_image, "create_nifti_import_gui_param") as create_param,
    ):
        assert ImagePanel.edit_import_parameters(panel, [volume]) == [(volume, None)]
    create_param.assert_not_called()


def test_common_parameters_apply_to_their_format(tmp_path) -> None:
    """Common parameters apply to the files of their format only."""
    image2d = save_nifti(tmp_path / "a.nii", (4, 3))
    volume = save_nifti(tmp_path / "b.nii", (4, 3, 5))
    broken = tmp_path / "c.nii"
    broken.write_bytes(b"not a NIfTI file")
    raw = str(tmp_path / "d.raw")
    files = [image2d, volume, str(broken), raw]
    panel = make_panel()

    nifti_param = NiftiImageImportParam.create(first_slice=1, last_slice=2)
    pairs = dict(ImagePanel.edit_import_parameters(panel, files, nifti_param))
    assert pairs[image2d] is None and pairs[raw] is None
    assert pairs[volume].last_slice == 2 and pairs[volume] is not nifti_param
    # Reading the file reports the error of an unreadable header
    assert isinstance(pairs[str(broken)], NiftiImageImportParam)

    raw_param = RawImageImportParam.create(width=3, height=2)
    pairs = dict(ImagePanel.edit_import_parameters(panel, files, raw_param))
    assert pairs[raw].width == 3
    assert pairs[image2d] is None and pairs[volume] is None

    with pytest.raises(TypeError, match="NiftiImageImportParam"):
        ImagePanel.edit_import_parameters(panel, files, object())


def test_nifti_files_are_grouped_by_default(tmp_path) -> None:
    """Only the slices of NIfTI files get a group of their own by default."""
    assert ImagePanel.groups_file_objects(None, str(tmp_path / "a.nii.gz"))
    assert not ImagePanel.groups_file_objects(None, str(tmp_path / "a.tif"))
