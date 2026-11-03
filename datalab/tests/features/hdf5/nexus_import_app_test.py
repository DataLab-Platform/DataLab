# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
NeXus import test
-----------------

A NeXus ``NXdata`` group is imported with its axes and its origin, and the
workspace saved afterwards is reopened without the NeXus file: the origin keeps
its JSON types, and dimensions without axis or decreasing axes are preserved.
Datasets that cannot be imported are reported without preventing the import of
the others, and the NeXus file is closed afterwards.
"""

# guitest: show

from __future__ import annotations

import json
import os
import os.path as osp

import h5py
import numpy as np
import pytest
from sigima.objects import ImageObj, get_experimental_source

from datalab.tests import datalab_test_app_context, helpers


def _create_nexus_file(fname: str) -> None:
    """Create a NeXus file with a spectrum, a map with non-uniform axes, and a
    detector image with a decreasing axis and a dimension without axis"""
    with h5py.File(fname, "w") as h5file:
        entry = h5file.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        spectrum = entry.create_group("spectrum")
        spectrum.attrs.update({"NX_class": "NXdata", "signal": "I0", "axes": "E"})
        spectrum["I0"] = np.array([3, 2**31 + 5, 7], dtype=np.uint32)
        spectrum["E"] = np.array([7.1, 7.2, 7.4])
        spectrum["E"].attrs["units"] = "keV"
        mapping = entry.create_group("map")
        mapping.attrs.update({"NX_class": "NXdata", "signal": "counts"})
        mapping.attrs["axes"] = np.array([b"y", b"x"])
        mapping["counts"] = np.array([[1, 2, 3], [4, 5, 2**32 - 1]], dtype=np.uint32)
        mapping["x"] = np.array([0.0, 1.0, 4.0])
        mapping["y"] = np.array([0.0, 2.0])
        detector = entry.create_group("detector")
        detector.attrs.update({"NX_class": "NXdata", "signal": "frame"})
        detector.attrs["axes"] = np.array([b".", b"x"])
        detector["frame"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
        detector["x"] = np.array([4.0, 2.0, 0.0])


def _two_images(win) -> tuple[ImageObj, ImageObj]:
    """Return the two images of the workspace"""
    images = win.imagepanel.objmodel.get_all_objects()
    assert len(images) == 2
    return images[0], images[1]


def test_nexus_import_and_workspace() -> None:
    """NXdata import, then workspace round trip without the NeXus file"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        nxsname = osp.join(tmpdir, "scan.nxs")
        _create_nexus_file(nxsname)
        wsname = osp.join(tmpdir, "workspace.h5")
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            h5io = win.h5inputoutput
            h5io.import_dataset_from_file(nxsname, "/entry/spectrum")
            h5io.import_dataset_from_file(nxsname, "/entry/map")
            h5io.import_dataset_from_file(nxsname, "/entry/detector")
            signal = win.signalpanel.objmodel.get_all_objects()[0]
            image, detector = _two_images(win)
            assert np.array_equal(signal.x, [7.1, 7.2, 7.4])
            assert signal.y[1] == 2**31 + 5
            assert signal.xunit == "keV"
            assert image.data[1, 2] == 2**32 - 1
            assert np.array_equal(image.xcoords, [0.0, 1.0, 4.0])
            # Decreasing axis: reversed with the image, as PlotPy needs
            assert (detector.x0, detector.dx) == (0.0, 2.0)
            assert np.array_equal(detector.data, [[3, 2, 1], [6, 5, 4]])
            win.save_h5_workspace(wsname)
            os.remove(nxsname)

            win.load_h5_workspace([wsname], reset_all=True)
            signal = win.signalpanel.objmodel.get_all_objects()[0]
            image, detector = _two_images(win)
            assert np.array_equal(signal.x, [7.1, 7.2, 7.4])
            assert image.data[1, 2] == 2**32 - 1
            assert np.array_equal(image.xcoords, [0.0, 1.0, 4.0])
            assert np.array_equal(detector.data, [[3, 2, 1], [6, 5, 4]])
            source = get_experimental_source(signal)
            assert source["kind"] == "nexus"
            assert source["location"] == nxsname
            assert source["entry"] == "/entry/spectrum"
            assert source["selection"] == {"signal": "I0", "axes": ["E"]}
            assert get_experimental_source(image)["selection"] == {
                "signal": "counts",
                "axes": ["y", "x"],
            }
            source = get_experimental_source(detector)
            assert source["selection"] == {"signal": "frame", "axes": [".", "x"]}
            assert source["conversions"] == ["axis 'x': reversed to increasing order"]
            # The reloaded origin keeps its JSON types, so it can be stored again
            for obj in (signal, image, detector):
                stored = obj.metadata["experimental_source"]
                version = stored["version"]
                assert isinstance(version, int) and not isinstance(version, bool)
                json.dumps(stored)


def _is_open(fname: str) -> bool:
    """Return True if an HDF5 file of this name is open in this process"""
    names = [fid.name for fid in h5py.h5f.get_obj_ids(types=h5py.h5f.OBJ_FILE)]
    return osp.basename(fname) in [osp.basename(os.fsdecode(name)) for name in names]


def test_import_continues_after_error() -> None:
    """A dataset that cannot be imported does not prevent importing the next ones:
    errors are reported once all datasets were imported, and the file is closed."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        nxsname = osp.join(tmpdir, "errors.nxs")
        with h5py.File(nxsname, "w") as h5file:
            bad = h5file.create_group("a_bad")
            bad.attrs.update({"NX_class": "NXdata", "signal": "data"})
            bad.attrs["axes"] = np.array([b".", b"x"])
            bad["data"] = np.ones((2, 4))
            bad["x"] = np.array([0.0, 2.0, 1.0, 3.0])  # Not monotonic
            good = h5file.create_group("b_good")
            good.attrs.update({"NX_class": "NXdata", "signal": "I0"})
            good["I0"] = np.array([1.0, 2.0, 3.0])
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            with pytest.raises(ValueError, match="a_bad: Axis 'x' is not strictly"):
                win.h5inputoutput.import_files([nxsname], True, False)
            assert not _is_open(nxsname)
            entries = [
                get_experimental_source(obj)["entry"]
                for obj in win.signalpanel.objmodel.get_all_objects()
                if get_experimental_source(obj) is not None
            ]
            assert entries == ["/b_good"]
            with pytest.raises(ValueError, match="Axis 'x' is not strictly"):
                win.h5inputoutput.import_dataset_from_file(nxsname, "/a_bad")
            assert not _is_open(nxsname)


if __name__ == "__main__":
    test_nexus_import_and_workspace()
    test_import_continues_after_error()
