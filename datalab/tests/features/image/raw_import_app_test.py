# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""RAW image import and History replay application test."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from sigima.io import RawImageImportParam

from datalab.gui.panel.image import RawImageImportGUIParam
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context
from datalab.tests.features.common.history_test_helpers import select_tree_entry


def test_raw_import_replay_reloads_after_recorded_deletion(tmp_path) -> None:
    """Replay a recorded RAW import after its image was deleted."""
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    filename = tmp_path / "image.raw"
    expected.tofile(filename)
    param = RawImageImportParam.create(dtype="uint16", width=3, height=2)

    with datalab_test_app_context(history=True) as win:
        history, panel = win.historypanel, win.imagepanel
        history.toggle_record_mode(True)

        with patch.object(RawImageImportGUIParam, "edit") as edit:
            objects = panel.load_from_files([str(filename)], import_params=[param])

            assert len(objects) == 1
            np.testing.assert_array_equal(objects[0].data, expected)
            action = history[len(history)]
            assert action.method_name == "load_from_files"
            history_param = action.kwargs["import_params"][0]
            assert isinstance(history_param, RawImageImportParam)
            assert history_param is not param
            param.width = 1
            assert history_param.width == 3

            output_uuid = get_uuid(objects[0])
            assert action.output_uuids == [output_uuid]
            panel.objview.select_objects([output_uuid])
            panel.remove_object(force=True)
            assert len(panel.objmodel) == 0
            assert history[len(history)].method_name == "remove_object"

            select_tree_entry(history, action.uuid)
            history.replay_restore_actions(restore_selection=False)

            edit.assert_not_called()

        assert len(panel.objmodel) == 1
        assert action.is_stale is False
        replayed_output_uuid = action.output_uuids[0]
        assert replayed_output_uuid != output_uuid
        assert history.runtime.objects.action_output_uuids[action.uuid] == [
            replayed_output_uuid
        ]
        assert (
            history.runtime.objects.output_to_action[replayed_output_uuid]
            == action.uuid
        )
        recreated = panel.objmodel[replayed_output_uuid]
        np.testing.assert_array_equal(recreated.data, expected)
