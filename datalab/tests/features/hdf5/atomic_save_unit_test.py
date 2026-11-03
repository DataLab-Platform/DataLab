# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Atomic HDF5 save unit tests
---------------------------

A failed save must never destroy the previous version of a workspace (or
``.dlhist``) file: the file is written to a temporary file of the same directory,
which replaces the previous file only once completely written.
"""

# guitest: show

from __future__ import annotations

import os
import os.path as osp
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from sigima.tests.data import create_paracetamol_signal

from datalab.gui.h5io import H5InputOutput
from datalab.h5.native import DATALAB_VERSION_NAME, atomic_native_writer
from datalab.tests import datalab_test_app_context, helpers


def _write_previous_file(filename: str) -> bytes:
    """Write a previous version of a file and return its contents."""
    with h5py.File(filename, "w") as h5file:
        h5file["original"] = np.arange(5.0)
    with open(filename, "rb") as fdesc:
        return fdesc.read()


def _read_bytes(filename: str) -> bytes:
    """Return the contents of a file."""
    with open(filename, "rb") as fdesc:
        return fdesc.read()


def _assert_no_temporary_file(dirname: str, filename: str) -> None:
    """Check that no temporary file was left next to the saved file."""
    assert sorted(os.listdir(dirname)) == [osp.basename(filename)]


def test_atomic_writer_replaces_file_once_written(tmp_path) -> None:
    """A successful write replaces the previous file."""
    filename = str(tmp_path / "workspace.h5")
    _write_previous_file(filename)
    with atomic_native_writer(filename) as writer:
        writer.h5["partial"] = np.arange(3.0)
        assert not osp.samefile(writer.filename, filename)
    with h5py.File(filename, "r") as h5file:
        assert set(h5file) == {DATALAB_VERSION_NAME, "partial"}
    _assert_no_temporary_file(str(tmp_path), filename)


def test_atomic_writer_creates_new_file(tmp_path) -> None:
    """A new file is created with the permissions of a newly created file."""
    filename = str(tmp_path / "new.h5")
    reference = str(tmp_path / "reference")
    with open(reference, "wb"):
        pass
    with atomic_native_writer(filename) as writer:
        writer.h5["data"] = np.arange(3.0)
    assert os.stat(filename).st_mode == os.stat(reference).st_mode
    os.remove(reference)
    _assert_no_temporary_file(str(tmp_path), filename)


def test_atomic_writer_keeps_previous_file_on_failure(tmp_path) -> None:
    """A failed write leaves the previous file untouched and no temporary file."""
    filename = str(tmp_path / "workspace.h5")
    previous = _write_previous_file(filename)
    with pytest.raises(OSError, match="disk full"):
        with atomic_native_writer(filename) as writer:
            writer.h5["partial"] = np.arange(3.0)
            raise OSError("disk full")
    assert _read_bytes(filename) == previous
    _assert_no_temporary_file(str(tmp_path), filename)


class _Panel:
    """Panel double writing one dataset, then optionally failing."""

    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail

    def serialize_to_hdf5(self, writer) -> None:
        """Serialize the panel, like ``BaseDataPanel.serialize_to_hdf5``."""
        writer.h5[self.name] = np.arange(3.0)
        if self.fail:
            raise OSError("write error")


class _Provenance:
    """Provenance double, optionally failing."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def save(self, h5file: h5py.File) -> None:
        """Save the provenance block, like ``ProvenanceService.save``."""
        h5file["provenance"] = np.arange(2.0)
        if self.fail:
            raise ValueError("provenance error")


@pytest.mark.parametrize(
    "panels, provenance",
    [
        ([_Panel("signals"), _Panel("images", fail=True)], _Provenance()),
        ([_Panel("signals"), _Panel("images")], _Provenance(fail=True)),
    ],
    ids=["panel", "provenance"],
)
def test_save_file_keeps_previous_workspace_on_failure(
    tmp_path, panels: list[_Panel], provenance: _Provenance
) -> None:
    """Regression: saving truncated the previous workspace file before writing,
    so that an error in a panel or in the provenance destroyed it."""
    filename = str(tmp_path / "workspace.h5")
    previous = _write_previous_file(filename)
    h5io = H5InputOutput(SimpleNamespace(panels=panels, provenance=provenance))
    with pytest.raises((OSError, ValueError)):
        h5io.save_file(filename)
    assert _read_bytes(filename) == previous
    _assert_no_temporary_file(str(tmp_path), filename)


def test_failed_save_keeps_workspace_modified(monkeypatch) -> None:
    """After a failed save, the previous file is intact and the workspace is still
    marked as modified."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        filename = osp.join(tmpdir, "workspace.h5")
        previous = _write_previous_file(filename)
        with datalab_test_app_context(console=False) as win:
            win.signalpanel.add_object(create_paracetamol_signal())
            assert win.is_modified()

            def fail(_writer) -> None:
                raise OSError("write error")

            monkeypatch.setattr(win.imagepanel, "serialize_to_hdf5", fail)
            with pytest.raises(OSError, match="write error"):
                win.save_h5_workspace(filename)
            assert win.is_modified()
            assert _read_bytes(filename) == previous
            monkeypatch.undo()
            win.save_h5_workspace(filename)
            assert not win.is_modified()
        assert not [name for name in os.listdir(tmpdir) if name.endswith(".tmp")]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
