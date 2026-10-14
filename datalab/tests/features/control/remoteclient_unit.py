# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Remote client test
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code
# guitest: skip

import os.path as osp

import numpy as np
from guidata.qthelpers import qt_app_context
from plotpy.builder import make
from sigima.params import XYCalibrateParam
from sigima.tests.data import create_2d_gaussian, create_paracetamol_signal

from datalab.control.proxy import RemoteProxy
from datalab.env import execenv
from datalab.tests import datalab_in_background_context, helpers


def multiple_commands(remote: RemoteProxy):
    """Execute multiple XML-RPC commands"""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        x, y = create_paracetamol_signal().get_data()
        remote.add_signal("tutu", x, y)

        z = create_2d_gaussian(2000, np.uint16)
        remote.add_image("toto", z)
        rect = make.annotated_rectangle(100, 100, 200, 200, title="Test")
        area = rect.get_rect()
        remote.add_annotations_from_items([rect])
        uuid = remote.get_sel_object_uuids()[0]
        assert remote.get_current_object_uuid() == uuid
        canonical_annotations = remote.get_object(uuid).get_annotations()
        assert len(canonical_annotations) == 1
        assert canonical_annotations[0]["format"] == "sigima.annotation"
        assert "plotpy_json" not in canonical_annotations[0]
        items = remote.get_object_shapes()
        assert len(items) == 1 and items[0].get_rect() == area
        remote.add_label_with_title(f"Image uuid: {uuid}")
        remote.select_groups([1])
        remote.select_objects([uuid])
        remote.delete_metadata()
        canonical_annotations = remote.get_object(uuid).get_annotations()

        annotations_workspace = osp.join(tmpdir, "annotations_workspace.h5")
        remote.save_h5_workspace(annotations_workspace)
        remote.reset_all()
        remote.load_h5_workspace([annotations_workspace], reset_all=True)
        remote.set_current_panel("image")
        restored_annotations = [
            remote.get_object(image_uuid).get_annotations()
            for image_uuid in remote.get_object_uuids()
        ]
        assert canonical_annotations in restored_annotations, (
            "Canonical annotations were not restored from the workspace: "
            f"expected {canonical_annotations!r}, got {restored_annotations!r}"
        )

        fname = osp.join(tmpdir, osp.basename("remote_test.h5"))
        remote.save_to_h5_file(fname)
        remote.reset_all()
        remote.open_h5_files([fname], True, False)
        remote.import_h5_file(fname, True)

        # Test new headless workspace API methods (Issue #275)
        fname_workspace = osp.join(tmpdir, "workspace_test.h5")
        remote.save_h5_workspace(fname_workspace)
        assert osp.exists(fname_workspace), "Workspace file was not created"
        remote.reset_all()
        remote.load_h5_workspace([fname_workspace], reset_all=True)
        # Verify objects were restored
        assert len(remote.get_object_titles()) > 0, "No objects after load_h5_workspace"

        remote.set_current_panel("signal")
        assert remote.get_current_panel() == "signal"

        # Test set_object round-trip (get → modify → set → verify)
        uuids = remote.get_object_uuids()
        obj = remote.get_object(uuids[0])
        original_title = obj.title
        obj.title = "Modified by set_object"
        remote.set_object(obj)
        obj2 = remote.get_object(uuids[0])
        assert obj2.title == "Modified by set_object", (
            f"set_object failed: expected 'Modified by set_object', got '{obj2.title}'"
        )
        obj2.title = original_title
        remote.set_object(obj2)

        remote.calc("log10")

        param = XYCalibrateParam.create(a=1.2, b=0.1)
        remote.calc("calibration", param)


def test_remoteclient_unit():
    """Remote client test"""
    execenv.print("Executing multiple commands...", end="")
    with qt_app_context():  # needed for building plot items
        with datalab_in_background_context() as remote:
            multiple_commands(remote)
    execenv.print("OK")


if __name__ == "__main__":
    test_remoteclient_unit()
