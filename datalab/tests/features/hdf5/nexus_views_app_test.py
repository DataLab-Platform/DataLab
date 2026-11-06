# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
NeXus scans and multidimensional views test
--------------------------------------------

A scan written with the legacy NeXus conventions (as recent SOLEIL scans) is
imported with its axes: its counter versus the scanned motor and its detector
spectra as an image with the motor along Y. The frames of an image stack whose
view is enabled in the HDF5 browser are imported as images titled and located
by the motor position, and their origin survives a workspace round trip.
"""

# guitest: show

from __future__ import annotations

import json
import os.path as osp

import h5py
import numpy as np
import pytest
from sigima.objects import get_experimental_source
from sigimax.h5.selection import NDView
from sigimax.widgets.h5browser import H5BrowserDialog

from datalab.tests import datalab_test_app_context, helpers


def _write_scan(fname: str) -> None:
    """Write a step scan: legacy axis attributes, a counter, a detector writing a
    spectrum per point and a detector writing an image per point"""
    with h5py.File(fname, "w") as h5file:
        entry = h5file.create_group("scan_0494")
        entry.attrs["NX_class"] = "NXentry"
        entry["title"] = "LaB6 calibration"
        scan = entry.create_group("scan_data")
        scan.attrs["NX_class"] = "NXdata"
        scan["actuator_1_1"] = np.array([-8.0, -7.5, -7.0])
        scan["actuator_1_1"].attrs.update(axis=np.int32(1), primary=np.int32(1))
        scan["actuator_1_1"].attrs["units"] = "deg"
        scan["counter"] = np.array([5.0, 9.0, 4.0])
        scan["spectra"] = np.arange(3 * 16, dtype=np.int32).reshape(3, 16)
        scan["frames"] = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
        for name in ("counter", "spectra", "frames"):
            scan[name].attrs["signal"] = np.int32(1)


def _nexus_objects(panel, signal: str) -> list:
    """Return the objects of a panel imported from a signal of an NXdata group"""
    return [
        obj
        for obj in panel.objmodel.get_all_objects()
        if (get_experimental_source(obj) or {}).get("selection", {}).get("signal")
        == signal
    ]


def test_scan_frames_and_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy scan counters and detectors, then frames of a stack enabled in the
    browser, and workspace round trip"""
    original_open = H5BrowserDialog.open_file

    def open_and_enable_frames(dialog: H5BrowserDialog, fname: str) -> None:
        """Open the file, and enable the import of the first two frames"""
        original_open(dialog, fname)
        tree = dialog.browser.tree
        item = tree.find_node_item(tree.topLevelItem(0), "/scan_0494/scan_data::frames")
        tree.get_node(item).set_view(NDView((1, 2), (0,), 0, 1))
        tree.set_item_importable(item, True)

    monkeypatch.setattr(H5BrowserDialog, "open_file", open_and_enable_frames)
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        nxsname = osp.join(tmpdir, "scan.nxs")
        _write_scan(nxsname)
        wsname = osp.join(tmpdir, "workspace.h5")
        with datalab_test_app_context(console=False, exec_loop=False) as win:
            win.h5inputoutput.import_files([nxsname], True, False)
            (counter,) = _nexus_objects(win.signalpanel, "counter")
            np.testing.assert_array_equal(counter.x, [-8.0, -7.5, -7.0])
            assert counter.xunit == "deg"
            source = get_experimental_source(counter)
            assert source["details"] == {"entry_title": "LaB6 calibration"}
            (spectra,) = _nexus_objects(win.imagepanel, "spectra")
            assert spectra.data.shape == (3, 16)
            assert (spectra.y0, spectra.dy) == (-8.0, 0.5)
            frames = _nexus_objects(win.imagepanel, "frames")
            assert len(frames) == 2
            assert frames[1].title.startswith("frames[1, :, :] (actuator_1_1=-7.5)")
            np.testing.assert_array_equal(
                frames[1].data, np.arange(20, 40).reshape(4, 5)
            )
            win.save_h5_workspace(wsname)

            win.load_h5_workspace([wsname], reset_all=True)
            frame = _nexus_objects(win.imagepanel, "frames")[1]
            source = get_experimental_source(frame)
            assert source["selection"]["view"] == {"displayed": [1, 2], "indices": [1]}
            assert source["selection"]["positions"] == {"actuator_1_1": -7.5}
            json.dumps(frame.metadata["experimental_source"])


if __name__ == "__main__":
    pytest.main([__file__])
