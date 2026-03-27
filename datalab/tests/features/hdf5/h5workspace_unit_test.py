# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
HDF5 workspace API unit tests
-----------------------------

Tests for the headless HDF5 workspace API methods:
- load_h5_workspace: Load native DataLab HDF5 files without GUI elements
- save_h5_workspace: Save workspace to native DataLab HDF5 file without GUI elements

These methods are designed for use from the internal console where Qt GUI elements
would cause thread-safety issues.

Also includes integration tests for the ``fix/HDF5_format`` Sigima changes:
- to_dict callable sanitization
- __check_value str-enum handling
- column_formats API
"""

# guitest: show

import enum
import os.path as osp

import h5py
import pytest
from numpy import ma
from sigima.objects.scalar import NO_ROI, TableResult, TableResultBuilder
from sigima.tests.data import (
    create_noisy_gaussian_image,
    create_paracetamol_signal,
    create_test_signal_rois,
)

from datalab.adapters_metadata.table_adapter import TableAdapter
from datalab.tests import datalab_test_app_context, helpers


def test_save_and_load_h5_workspace():
    """Test save_h5_workspace and load_h5_workspace methods"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # === Create test objects
            sig1 = create_paracetamol_signal()
            win.signalpanel.add_object(sig1)

            ima1 = create_noisy_gaussian_image()
            win.imagepanel.add_object(ima1)

            # Store object counts and titles for verification
            sig_count_before = len(win.signalpanel.objmodel)
            ima_count_before = len(win.imagepanel.objmodel)
            sig_title = sig1.title
            ima_title = ima1.title

            # === Test save_h5_workspace
            fname = osp.join(tmpdir, "test_workspace.h5")
            win.save_h5_workspace(fname)
            assert osp.exists(fname), "HDF5 file was not created"

            # === Clear workspace
            for panel in win.panels:
                panel.remove_all_objects()

            assert len(win.signalpanel.objmodel) == 0
            assert len(win.imagepanel.objmodel) == 0

            # === Test load_h5_workspace
            win.load_h5_workspace([fname], reset_all=True)

            # Verify objects were restored
            assert len(win.signalpanel.objmodel) == sig_count_before
            assert len(win.imagepanel.objmodel) == ima_count_before

            # Verify titles (get objects in order from groups)
            loaded_sig = win.signalpanel.objmodel.get_all_objects()[0]
            loaded_ima = win.imagepanel.objmodel.get_all_objects()[0]
            assert loaded_sig.title == sig_title
            assert loaded_ima.title == ima_title


def test_load_h5_workspace_invalid_file():
    """Test load_h5_workspace raises ValueError for non-native HDF5 files"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create a non-native HDF5 file (just an empty HDF5)
            fname = osp.join(tmpdir, "not_native.h5")
            with h5py.File(fname, "w") as f:
                f.create_dataset("dummy", data=[1, 2, 3])

            # Should raise ValueError for non-native file
            with pytest.raises(ValueError, match="not a native DataLab HDF5 file"):
                win.load_h5_workspace([fname])


def test_load_h5_workspace_append():
    """Test load_h5_workspace with reset_all=False (append mode)"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create and save first signal
            sig1 = create_paracetamol_signal()
            sig1.title = "Signal 1"
            win.signalpanel.add_object(sig1)

            fname = osp.join(tmpdir, "workspace1.h5")
            win.save_h5_workspace(fname)

            # Clear and create second signal
            win.signalpanel.remove_all_objects()
            sig2 = create_paracetamol_signal()
            sig2.title = "Signal 2"
            win.signalpanel.add_object(sig2)

            assert len(win.signalpanel.objmodel) == 1

            # Load first workspace with reset_all=False (append)
            win.load_h5_workspace([fname], reset_all=False)

            # Should now have both signals (appended)
            assert len(win.signalpanel.objmodel) == 2


def test_save_h5_workspace_modified_flag():
    """Test that save_h5_workspace clears the modified flag"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            # Create an object (this sets modified flag)
            sig1 = create_paracetamol_signal()
            win.signalpanel.add_object(sig1)

            # Workspace should be modified
            assert win.is_modified()

            # Save workspace
            fname = osp.join(tmpdir, "test.h5")
            win.save_h5_workspace(fname)

            # Modified flag should be cleared
            assert not win.is_modified()


# ---------------------------------------------------------------------------
#  HDF5 format fix: str-enum helper
# ---------------------------------------------------------------------------


class _SampleStrEnum(str, enum.Enum):
    """Str-based enum mimicking ``guidata.dataset.LabeledEnum``."""

    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"


def _get_table(sig, func_name: str) -> TableResult:
    """Retrieve a TableResult from signal metadata by func_name."""
    for adapter in TableAdapter.iterate_from_obj(sig):
        if adapter.func_name == func_name:
            return adapter.result
    raise KeyError(f"No table with func_name={func_name!r} found in metadata")


# ---------------------------------------------------------------------------
#  HDF5 format fix: callable sanitization
# ---------------------------------------------------------------------------


def test_h5_workspace_callable_attrs_stripped():
    """Test that callable attrs in TableResult are stripped after workspace I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            table = TableResult(
                title="Sanitize",
                headers=["col1"],
                data=[[1.0]],
                roi_indices=[NO_ROI],
                func_name="sanitize_test",
                attrs={"method": "peak", "callback": print},
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "sanitize_test")
            assert restored_table.attrs["method"] == "peak"
            assert "callback" not in restored_table.attrs


def test_h5_workspace_nested_callable_stripped():
    """Test that nested callable attrs are stripped after workspace I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            table = TableResult(
                title="NestedSanitize",
                headers=["col1"],
                data=[[1.0]],
                roi_indices=[NO_ROI],
                func_name="nested_sanitize_test",
                attrs={"info": {"label": "ok", "fn": abs}},
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "nested_sanitize_test")
            assert restored_table.attrs["info"] == {"label": "ok"}


def test_h5_workspace_table_data_survives_with_tainted_attrs():
    """Test that title, headers, and data survive despite tainted attrs"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            table = TableResult(
                title="Stats",
                headers=["min", "max"],
                data=[[1.5, 9.8], [2.3, 7.6]],
                roi_indices=[NO_ROI, 0],
                func_name="stats_test",
                attrs={"clean": "yes", "dirty": lambda: 0},
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "stats_test")
            assert restored_table.title == "Stats"
            assert restored_table.headers == ["min", "max"]
            assert len(restored_table.data) == 2
            assert restored_table.data[0][0] == pytest.approx(1.5)
            assert restored_table.data[1][1] == pytest.approx(7.6)


# ---------------------------------------------------------------------------
#  HDF5 format fix: str-enum handling
# ---------------------------------------------------------------------------


def test_h5_workspace_str_enum_survives():
    """Test that str-based enum values survive workspace I/O as plain str"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            roi = list(create_test_signal_rois(sig))[0]
            sig.roi = roi

            builder = TableResultBuilder("EnumTest")
            builder.add(lambda _data: _SampleStrEnum.GAUSSIAN, "shape")
            builder.add(ma.mean, "mean")
            table = builder.compute(sig)
            table = TableResult(
                title=table.title,
                headers=list(table.headers),
                data=table.data,
                roi_indices=table.roi_indices,
                func_name="enum_test",
                attrs=dict(table.attrs),
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "enum_test")
            for row in restored_table.data:
                assert isinstance(row[0], str)
                assert isinstance(row[1], float)


def test_h5_workspace_str_enum_value_preserved():
    """Test that original str enum value is preserved after workspace I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            roi = list(create_test_signal_rois(sig))[0]
            sig.roi = roi

            builder = TableResultBuilder("EnumVal")
            builder.add(lambda _data: _SampleStrEnum.LORENTZIAN, "shape")
            table = builder.compute(sig)
            table = TableResult(
                title=table.title,
                headers=list(table.headers),
                data=table.data,
                roi_indices=table.roi_indices,
                func_name="enum_val_test",
                attrs=dict(table.attrs),
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "enum_val_test")
            assert restored_table.data[0][0] == str(_SampleStrEnum.LORENTZIAN)


# ---------------------------------------------------------------------------
#  HDF5 format fix: column_formats API
# ---------------------------------------------------------------------------


def test_h5_workspace_column_formats_survive():
    """Test that per-column formats survive workspace I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            table = TableResult(
                title="Formats",
                headers=["x", "y"],
                data=[[1.0, 2.0]],
                roi_indices=[NO_ROI],
                func_name="formats_test",
            )
            table.set_column_formats({"x": ".2e", "y": ".3g"})
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "formats_test")
            assert restored_table.get_column_formats() == {"x": ".2e", "y": ".3g"}


def test_h5_workspace_empty_column_formats():
    """Test that table with no column_formats keeps empty dict after I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            table = TableResult(
                title="NoFormats",
                headers=["a"],
                data=[[1.0]],
                roi_indices=[NO_ROI],
                func_name="no_formats_test",
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "no_formats_test")
            assert restored_table.get_column_formats() == {}


def test_h5_workspace_builder_column_formats():
    """Test that builder column formats survive workspace I/O"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            sig = create_paracetamol_signal()
            roi = list(create_test_signal_rois(sig))[0]
            sig.roi = roi

            builder = TableResultBuilder("BuilderFmt")
            builder.add(ma.min, "min")
            builder.add(ma.max, "max")
            builder.set_column_formats({"min": ".2e", "max": ".3g"})
            table = builder.compute(sig)
            table = TableResult(
                title=table.title,
                headers=list(table.headers),
                data=table.data,
                roi_indices=table.roi_indices,
                func_name="builder_fmt_test",
                attrs=dict(table.attrs),
            )
            TableAdapter(table).add_to(sig)
            win.signalpanel.add_object(sig)

            fname = osp.join(tmpdir, "workspace.h5")
            win.save_h5_workspace(fname)
            for panel in win.panels:
                panel.remove_all_objects()
            win.load_h5_workspace([fname], reset_all=True)

            restored = win.signalpanel.objmodel.get_all_objects()[0]
            restored_table = _get_table(restored, "builder_fmt_test")
            assert restored_table.get_column_formats() == {
                "min": ".2e",
                "max": ".3g",
            }


if __name__ == "__main__":
    test_save_and_load_h5_workspace()
    test_load_h5_workspace_invalid_file()
    test_load_h5_workspace_append()
    test_save_h5_workspace_modified_flag()
    test_h5_workspace_callable_attrs_stripped()
    test_h5_workspace_nested_callable_stripped()
    test_h5_workspace_table_data_survives_with_tainted_attrs()
    test_h5_workspace_str_enum_survives()
    test_h5_workspace_str_enum_value_preserved()
    test_h5_workspace_column_formats_survive()
    test_h5_workspace_empty_column_formats()
    test_h5_workspace_builder_column_formats()
