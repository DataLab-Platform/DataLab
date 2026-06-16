# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Result title rendering mode application test (issue #149).

End-to-end validation that, after a real processing operation, the object tree
displays source short IDs or source titles depending on
``Conf.proc.result_title_mode`` — without altering the stored object title.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import CosineParam, create_signal_from_param

from datalab.config import Conf
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context


def test_result_title_mode_app():
    """Result-title rendering mode: short IDs vs source titles (display only)."""
    previous_mode = Conf.proc.result_title_mode.get()
    try:
        with datalab_test_app_context() as win:
            panel = win.signalpanel
            s1 = create_signal_from_param(CosineParam.create(size=100))
            s1.title = "My cosine"
            panel.add_object(s1)
            src_uuid = get_uuid(s1)

            panel.processor.run_feature("fft")
            result = panel.objmodel.get_all_objects()[-1]
            result_uuid = get_uuid(result)
            # Stored title references the source short ID (canonical form):
            assert "s001" in result.title

            # Default short-ID mode: tree shows the canonical title.
            Conf.proc.result_title_mode.set("short_id")
            panel.objview.populate_tree()
            text = panel.objview.get_item_from_id(result_uuid).text(0)
            assert "s001" in text
            assert "My cosine" not in text

            # Title mode: tree shows the source title; stored title unchanged.
            Conf.proc.result_title_mode.set("title")
            panel.objview.populate_tree()
            text = panel.objview.get_item_from_id(result_uuid).text(0)
            assert "My cosine" in text
            assert "s001" not in text
            assert "s001" in result.title

            # Renaming the source updates the dependent display title live.
            panel.objview.set_current_item_id(src_uuid)
            panel.rename_selected_object_or_group("Renamed cosine")
            text = panel.objview.get_item_from_id(result_uuid).text(0)
            assert "Renamed cosine" in text

            # Toggling back reverts to short IDs.
            Conf.proc.result_title_mode.set("short_id")
            panel.objview.populate_tree()
            text = panel.objview.get_item_from_id(result_uuid).text(0)
            assert "s001" in text
            assert "Renamed cosine" not in text
    finally:
        Conf.proc.result_title_mode.set(previous_mode)


if __name__ == "__main__":
    test_result_title_mode_app()
