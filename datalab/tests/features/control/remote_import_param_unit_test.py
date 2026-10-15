# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test format-specific import parameters through the remote control API

RAW image files require import parameters describing their binary layout. Remote
clients pass them to ``load_from_files`` and ``load_from_directory`` as a DataSet,
like computation parameters with ``calc``, so that no dialog is shown.
"""

# guitest: show

from __future__ import annotations

import time

import numpy as np
from sigima.io import RawImageImportParam

from datalab.control.proxy import RemoteProxy
from datalab.tests import datalab_in_background_context


def wait_for_image_uuids(proxy: RemoteProxy, count: int) -> list[str]:
    """Return image UUIDs once ``count`` images exist.

    The remote call may return before the main window has finished adding the
    loaded objects, so the object list is polled for a short time.
    """
    deadline = time.monotonic() + 10.0
    uuids = proxy.get_object_uuids("image")
    while len(uuids) < count and time.monotonic() < deadline:
        time.sleep(0.1)
        uuids = proxy.get_object_uuids("image")
    return uuids


def test_remote_raw_import_with_param(tmp_path) -> None:
    """Load RAW files and directories remotely with import parameters."""
    file_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    dir_data = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.uint16)
    filename = tmp_path / "image.raw"
    file_data.tofile(filename)
    directory = tmp_path / "directory"
    directory.mkdir()
    dir_data.tofile(directory / "image.raw")

    with datalab_in_background_context() as proxy:
        proxy.set_current_panel("image")

        param = RawImageImportParam.create(dtype="uint16", width=3, height=2)
        proxy.load_from_files([str(filename)], param=param)
        uuids = wait_for_image_uuids(proxy, 1)
        assert len(uuids) == 1
        np.testing.assert_array_equal(proxy.get_object(uuids[0]).data, file_data)

        param = RawImageImportParam.create(dtype="uint16", width=2, height=3)
        proxy.load_from_directory(str(directory), param=param)
        uuids = wait_for_image_uuids(proxy, 2)
        assert len(uuids) == 2
        np.testing.assert_array_equal(proxy.get_object(uuids[1]).data, dir_data)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_remote_raw_import_with_param(Path(tmpdir))
