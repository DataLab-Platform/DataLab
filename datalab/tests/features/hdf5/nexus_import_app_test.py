# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
NeXus import test
-----------------

A NeXus ``NXdata`` group is imported with its axes and its origin, and the
workspace saved afterwards is reopened without the NeXus file.
"""

# guitest: show

from __future__ import annotations

import os
import os.path as osp

import h5py
import numpy as np
from sigima.objects import get_experimental_source

from datalab.tests import datalab_test_app_context, helpers


def _create_nexus_file(fname: str) -> None:
    """Create a NeXus file with a spectrum and a map with non-uniform axes"""
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
            signal = win.signalpanel.objmodel.get_all_objects()[0]
            image = win.imagepanel.objmodel.get_all_objects()[0]
            assert np.array_equal(signal.x, [7.1, 7.2, 7.4])
            assert signal.y[1] == 2**31 + 5
            assert signal.xunit == "keV"
            assert image.data[1, 2] == 2**32 - 1
            assert np.array_equal(image.xcoords, [0.0, 1.0, 4.0])
            win.save_h5_workspace(wsname)
            os.remove(nxsname)

            win.load_h5_workspace([wsname], reset_all=True)
            signal = win.signalpanel.objmodel.get_all_objects()[0]
            image = win.imagepanel.objmodel.get_all_objects()[0]
            assert np.array_equal(signal.x, [7.1, 7.2, 7.4])
            assert image.data[1, 2] == 2**32 - 1
            assert np.array_equal(image.xcoords, [0.0, 1.0, 4.0])
            source = get_experimental_source(signal)
            assert source["kind"] == "nexus"
            assert source["location"] == nxsname
            assert source["entry"] == "/entry/spectrum"
            assert source["selection"] == {"signal": "I0", "axes": ["E"]}
            assert get_experimental_source(image)["selection"] == {
                "signal": "counts",
                "axes": ["y", "x"],
            }


if __name__ == "__main__":
    test_nexus_import_and_workspace()
