# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
HDF5 reference-attribute unit tests
===================================

Headless regression tests for :meth:`datalab.h5.common.BaseNode.collect_attributes`.

Files converted from HDF4 with ``h4toh5convert`` carry HDF5 object/region
*reference* attributes (e.g. ``DIMENSION_LIST`` / ``REFERENCE_LIST``). h5py
exposes those as :class:`h5py.h5r.Reference` values, which are **not** picklable
(``TypeError: no default __reduce__ due to non-trivial __cinit__``). Copying them
into an object's metadata used to crash the first computation, because DataLab
pickles the object to run it in a worker process.
"""

from __future__ import annotations

import pickle
import uuid

import h5py
import numpy as np
import pytest

from datalab.h5.common import BaseNode


class _LeafNode(BaseNode):
    """Minimal concrete node exposing :meth:`collect_attributes`."""


def _new_memory_file() -> h5py.File:
    """Return a fresh in-memory HDF5 file with a unique name."""
    return h5py.File(f"{uuid.uuid4()}.h5", "w", driver="core", backing_store=False)


def test_collect_attributes_keeps_serialisable_values() -> None:
    """Numeric, boolean and string attributes are copied to metadata."""
    h5file = _new_memory_file()
    try:
        dset = h5file.create_dataset("data", data=np.zeros((4, 4)))
        dset.attrs["gain"] = 2.5
        dset.attrs["count"] = np.int32(7)
        dset.attrs["enabled"] = True
        dset.attrs["label"] = b"detector"
        dset.attrs["profile"] = np.array([1.0, 2.0, 3.0])
        dset.attrs["tags"] = np.array([b"a", b"b"])

        node = _LeafNode(h5file, "data")
        node.collect_attributes()

        assert node.metadata["gain"] == 2.5
        assert node.metadata["count"] == 7
        assert node.metadata["enabled"]
        assert node.metadata["label"] == "detector"
        assert np.array_equal(node.metadata["profile"], [1.0, 2.0, 3.0])
        assert list(node.metadata["tags"]) == [b"a", b"b"]
        # The whole metadata mapping must stay picklable.
        pickle.dumps(node.metadata)
    finally:
        h5file.close()


def test_collect_attributes_skips_reference_attributes() -> None:
    """HDF5 reference attributes are dropped (they are not picklable)."""
    ref_dtype = h5py.special_dtype(ref=h5py.Reference)
    h5file = _new_memory_file()
    try:
        h5file.create_dataset("data", data=np.zeros((4, 4)))
        h5file.create_dataset("scale0", data=np.arange(4))
        h5file.create_dataset("scale1", data=np.arange(4))
        # Sanity check: a raw HDF5 reference is indeed not picklable.
        with pytest.raises(TypeError):
            pickle.dumps(h5file["scale0"].ref)
        # Array of references, as produced by h4toh5convert (DIMENSION_LIST).
        refs = np.array([h5file["scale0"].ref, h5file["scale1"].ref], dtype=ref_dtype)
        h5file["data"].attrs.create("DIMENSION_LIST", refs)
        # Scalar reference attribute.
        h5file["data"].attrs["REFERENCE"] = h5file["scale0"].ref
        # A regular attribute alongside the reference ones.
        h5file["data"].attrs["gain"] = 1.5

        node = _LeafNode(h5file, "data")
        node.collect_attributes()

        assert "DIMENSION_LIST" not in node.metadata
        assert "REFERENCE" not in node.metadata
        assert node.metadata["gain"] == 1.5
        # The surviving metadata is picklable (the whole point of the fix).
        pickle.dumps(node.metadata)
    finally:
        h5file.close()


if __name__ == "__main__":
    test_collect_attributes_keeps_serialisable_values()
    test_collect_attributes_skips_reference_attributes()
