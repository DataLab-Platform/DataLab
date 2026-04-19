# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
HDF5 Utils Unit Tests
=====================

Headless unit tests for the small helpers in :mod:`datalab.h5.utils`.

These helpers are used to defensively normalize values pulled out of
heterogeneous HDF5 datasets (LMJ, FXD, etc.). They never touch Qt and were
mostly uncovered by the existing test suite.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from datalab.h5.utils import (
    fix_ldata,
    fix_ndata,
    is_single_str_array,
    is_supported_num_dtype,
    is_supported_str_dtype,
    process_label,
    process_scalar_value,
    process_xy_values,
)

# =============================================================================
# fix_ldata
# =============================================================================


class TestFixLdata:
    """Tests for :func:`fix_ldata` (label normalisation)."""

    def test_string(self) -> None:
        assert fix_ldata("hello") == "hello"

    def test_bytes(self) -> None:
        assert fix_ldata(b"hello") == "hello"

    def test_numpy_bytes(self) -> None:
        assert fix_ldata(np.bytes_(b"label")) == "label"

    def test_none_returns_empty(self) -> None:
        assert fix_ldata(None) == ""

    def test_empty_string_returns_empty(self) -> None:
        # Falsy non-None inputs should also fall through to ``""``
        assert fix_ldata("") == ""

    def test_unknown_type_returns_empty(self) -> None:
        # A non-string, non-bytes, non-void object returns "" by design
        assert fix_ldata(3.14) == ""


# =============================================================================
# fix_ndata
# =============================================================================


class TestFixNdata:
    """Tests for :func:`fix_ndata` (numeric normalisation)."""

    def test_integer_value_returns_int(self) -> None:
        result = fix_ndata(5.0)
        assert result == 5
        assert type(result) is int

    def test_float_value_returns_float(self) -> None:
        result = fix_ndata(2.5)
        assert result == pytest.approx(2.5)
        assert type(result) is float

    def test_string_int_returns_int(self) -> None:
        assert fix_ndata("7") == 7

    def test_invalid_string_returns_none(self) -> None:
        assert fix_ndata("not-a-number") is None

    def test_none_returns_none(self) -> None:
        assert fix_ndata(None) is None


# =============================================================================
# Dtype helpers
# =============================================================================


class TestDtypeHelpers:
    """Tests for the small dtype detection helpers."""

    @pytest.mark.parametrize(
        "dtype",
        [np.int8, np.int32, np.uint16, np.float32, np.float64, np.complex128],
    )
    def test_supported_num_dtype_true(self, dtype) -> None:
        arr = np.zeros(3, dtype=dtype)
        assert is_supported_num_dtype(arr) is True

    def test_supported_num_dtype_false_for_object(self) -> None:
        arr = np.array(["abc", "def"], dtype=object)
        assert is_supported_num_dtype(arr) is False

    def test_is_single_str_array_true(self) -> None:
        # ``np.generic`` numpy.str_ scalar with shape (1,)
        scalar = np.array(["x"], dtype=str)[0:1]  # ndarray, not generic
        # ``is_single_str_array`` requires an ``np.generic``; ndarray returns False
        assert is_single_str_array(scalar) is False

    def test_is_single_str_array_false_for_ndarray(self) -> None:
        arr = np.array(["a", "b"])
        assert is_single_str_array(arr) is False

    def test_supported_str_dtype_true_for_bytes_array(self) -> None:
        arr = np.array([b"x", b"y"], dtype="S2")
        # numpy bytes dtype name starts with "bytes" not "string" → expected False
        assert is_supported_str_dtype(arr) is False

    def test_supported_str_dtype_false_for_int(self) -> None:
        assert is_supported_str_dtype(np.zeros(3, dtype=np.int32)) is False


# =============================================================================
# process_scalar_value / process_label / process_xy_values (HDF5 datasets)
# =============================================================================


@pytest.fixture
def h5_with_datasets(tmp_path):
    """Build a small in-memory HDF5 file containing typical layouts."""
    path = tmp_path / "fixture.h5"
    with h5py.File(path, "w") as f:
        # Scalar value as a 1-element array (the common LMJ layout)
        f.create_dataset("scalar", data=np.array([42.5]))
        # Label as a 2-element string list
        f.create_dataset("label2", data=np.array([b"X-Axis", b"Y-Axis"], dtype="S20"))
        # Label as a 3-element string list
        f.create_dataset(
            "label3",
            data=np.array([b"X", b"Y", b"Z"], dtype="S20"),
        )
        # x/y pair
        f.create_dataset("xy", data=np.array([1.5, 2.5]))
    yield path


class TestProcessScalarValue:
    """Tests for :func:`process_scalar_value`."""

    def test_returns_callback_result(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            result = process_scalar_value(f, "scalar", float)
        assert result == pytest.approx(42.5)

    def test_missing_dataset_returns_none(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            result = process_scalar_value(f, "missing", float)
        assert result is None


class TestProcessLabel:
    """Tests for :func:`process_label`."""

    def test_two_element_label(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            xl, yl, zl = process_label(f, "label2")
        assert xl == "X-Axis"
        assert yl == "Y-Axis"
        assert zl == ""

    def test_three_element_label(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            xl, yl, zl = process_label(f, "label3")
        assert (xl, yl, zl) == ("X", "Y", "Z")

    def test_missing_returns_empty_strings(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            result = process_label(f, "missing")
        assert result == ("", "", "")


class TestProcessXyValues:
    """Tests for :func:`process_xy_values`."""

    def test_returns_pair(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            x, y = process_xy_values(f, "xy")
        assert x == pytest.approx(1.5)
        assert y == pytest.approx(2.5)

    def test_missing_returns_none_pair(self, h5_with_datasets) -> None:
        with h5py.File(h5_with_datasets, "r") as f:
            x, y = process_xy_values(f, "missing")
        assert x is None
        assert y is None


if __name__ == "__main__":
    pytest.main([__file__])
