# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Computed title regression tests.

Real GUI regression tests covering the "Reset to computed title" feature and the
provenance of the computed title across selection changes, properties edition,
metadata paste, HDF5 workspace append and object duplication.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from __future__ import annotations

import os.path as osp

from sigima.objects import CosineParam, create_signal_from_param

from datalab.gui import actionhandler
from datalab.gui.panel.base import PasteMetadataParam
from datalab.objectmodel import get_computed_title, get_short_id
from datalab.tests import datalab_test_app_context, helpers


def get_reset_action(panel):
    """Return the context-menu action that resets computed titles."""
    return next(
        action
        for action in panel.get_category_actions(
            actionhandler.ActionCategory.CONTEXT_MENU
        )
        if action is not None and action.text() == "Reset to computed title"
    )


def create_average_result_group(panel):
    """Create two cosine signals and their average result group."""
    panel.remove_all_objects()
    panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
    panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
    source_group = panel.objmodel.get_groups()[0]
    panel.objview.select_groups([source_group])
    panel.processor.run_feature("average")
    result_group = panel.objmodel.get_groups()[-1]
    assert panel.objmodel.get_computed_title(result_group) == result_group.title
    return source_group, result_group


def test_reset_action_enabled_after_group_reselection() -> None:
    """Reset stays enabled after a renamed result group is reselected."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        source_group, result_group = create_average_result_group(panel)
        reset_action = get_reset_action(panel)

        panel.objview.select_groups([result_group])
        panel.rename_selected_object_or_group("Custom result")
        panel.objview.select_groups([source_group])
        panel.objview.select_groups([result_group])

        selected_groups = panel.objview.get_sel_groups()
        selected_objects = panel.objview.get_sel_objects(include_groups=True)
        reset_enabled = reset_action.isEnabled()
        print(
            "Result group reselection: "
            f"groups={len(selected_groups)}, objects={len(selected_objects)}, "
            f"reset_enabled={reset_enabled}"
        )

        assert reset_enabled


def test_reset_action_enabled_after_properties_title_edit() -> None:
    """Applying a custom title immediately enables the Reset action."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        panel.remove_all_objects()
        panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
        panel.processor.run_feature("fft")
        result = panel.objview.get_current_object()
        assert result is not None
        computed_title = get_computed_title(result)
        assert computed_title is not None
        assert result.title == computed_title

        reset_action = get_reset_action(panel)
        assert not reset_action.isEnabled()

        panel.objprop.properties.dataset.title = "Custom FFT"
        panel.properties_changed()

        reset_enabled = reset_action.isEnabled()
        print(f"Properties title={result.title}, reset_enabled={reset_enabled}")
        assert result.title == "Custom FFT"
        assert reset_enabled


def test_metadata_paste_preserves_target_computed_title() -> None:
    """Metadata paste preserves the target result's computed title."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        panel.remove_all_objects()
        source = create_signal_from_param(CosineParam.create(size=100))
        panel.add_object(source)

        panel.objview.select_objects([source])
        panel.processor.run_feature("fft")
        fft_result = panel.objview.get_current_object()
        assert fft_result is not None
        fft_computed_title = get_computed_title(fft_result)

        panel.objview.select_objects([source])
        panel.processor.run_feature("derivative")
        derivative_result = panel.objview.get_current_object()
        assert derivative_result is not None
        derivative_computed_title = get_computed_title(derivative_result)

        assert fft_computed_title is not None
        assert derivative_computed_title is not None
        assert fft_computed_title != derivative_computed_title
        print(
            "Computed titles before paste: "
            f"FFT={fft_computed_title}, derivative={derivative_computed_title}"
        )

        panel.objview.select_objects([fft_result])
        panel.copy_metadata()
        panel.objview.select_objects([derivative_result])
        param = PasteMetadataParam("Paste metadata")
        param.keep_other = True
        param.keep_roi = False
        param.keep_geometry = False
        param.keep_tables = False
        panel.paste_metadata(param)

        pasted_computed_title = get_computed_title(derivative_result)
        panel.objmodel.reset_title_to_computed(derivative_result)
        reset_title = derivative_result.title
        print(
            "Derivative title provenance: "
            f"before={derivative_computed_title}, "
            f"after_paste={pasted_computed_title}, after_reset={reset_title}"
        )

        assert pasted_computed_title == derivative_computed_title
        assert reset_title == derivative_computed_title


def test_h5_append_remaps_group_computed_title() -> None:
    """HDF5 append remaps source IDs in a group's computed title."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            for current_panel in win.panels:
                current_panel.remove_all_objects()
            panel = win.signalpanel
            panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
            panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
            source_group = panel.objmodel.get_groups()[0]
            panel.objview.select_groups([source_group])
            panel.rename_selected_object_or_group("Imported sources")
            panel.objview.select_groups([source_group])
            panel.processor.run_feature("average")

            result_group = panel.objmodel.get_groups()[-1]
            saved_computed_title = panel.objmodel.get_computed_title(result_group)
            assert saved_computed_title is not None
            assert result_group.title == saved_computed_title
            panel.objview.select_groups([result_group])
            panel.rename_selected_object_or_group("Custom result")

            fname = osp.join(tmpdir, "issue149_workspace.h5")
            win.save_h5_workspace(fname)
            for current_panel in win.panels:
                current_panel.remove_all_objects()
            panel.add_object(create_signal_from_param(CosineParam.create(size=100)))
            win.load_h5_workspace([fname], reset_all=False)

            model = panel.objmodel
            imported_source_group = model.get_group_from_title("Imported sources")
            imported_result_group = model.get_group_from_title("Custom result")
            imported_source_group_id = get_short_id(imported_source_group)
            imported_result_group_id = get_short_id(imported_result_group)
            assert imported_source_group_id == "gs002"

            stored_computed_title = model.get_computed_title(imported_result_group)
            model.reset_title_to_computed(imported_result_group)
            reset_title = imported_result_group.title
            rendered_title = model.get_display_title(
                imported_result_group, use_titles=True
            )
            print(
                "HDF5 append values: "
                f"saved={saved_computed_title}, "
                f"source_id={imported_source_group_id}, "
                f"result_id={imported_result_group_id}, "
                f"stored={stored_computed_title}, reset={reset_title}, "
                f"rendered={rendered_title}"
            )

            assert stored_computed_title is not None
            assert "gs002" in stored_computed_title
            assert reset_title == stored_computed_title
            assert "gs002" in reset_title
            assert "Imported sources" in rendered_title


def test_duplicate_group_preserves_computed_title() -> None:
    """Duplicating a result group preserves its computed-title provenance."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel
        source_group, result_group = create_average_result_group(panel)
        original_computed_title = panel.objmodel.get_computed_title(result_group)
        assert original_computed_title is not None

        panel.objview.select_groups([result_group])
        panel.duplicate_object()
        duplicate_group = panel.objmodel.get_groups()[-1]
        assert duplicate_group is not source_group
        assert duplicate_group is not result_group
        duplicate_computed_title = panel.objmodel.get_computed_title(duplicate_group)

        panel.objview.select_groups([duplicate_group])
        panel.rename_selected_object_or_group("Custom duplicate")
        panel.objview.select_groups([duplicate_group])
        reset_action = get_reset_action(panel)
        reset_enabled = reset_action.isEnabled()
        print(
            "Duplicated group provenance: "
            f"original={original_computed_title}, "
            f"duplicate={duplicate_computed_title}, "
            f"reset_enabled={reset_enabled}"
        )

        assert duplicate_computed_title == original_computed_title
        assert reset_enabled


if __name__ == "__main__":
    test_reset_action_enabled_after_group_reselection()
    test_reset_action_enabled_after_properties_title_edit()
    test_metadata_paste_preserves_target_computed_title()
    test_h5_append_remaps_group_computed_title()
    test_duplicate_group_preserves_computed_title()
