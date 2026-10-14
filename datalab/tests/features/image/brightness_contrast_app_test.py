# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Brightness and contrast application integration test."""

from __future__ import annotations

import os.path as osp

import numpy as np
import sigima.params
from guidata.dataset.qtitemwidgets import HistogramRangeWidget
from guidata.dataset.qtwidgets import DataSetEditDialog
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from sigima.objects import create_image

from datalab.gui.actionhandler import SelectCond
from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers


def test_brightness_contrast_integration() -> None:
    """One parameter window produces an independent result for every source."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel
        processor = panel.processor
        feature = processor.get_feature("adjust_brightness_contrast")
        assert feature.preview_enabled

        source_data = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        source = create_image("Source", source_data)
        other_data = source_data.astype(np.uint16)
        other = create_image("Other", other_data)
        panel.add_object(source)
        panel.add_object(other)

        managed_actions = panel.acthandler._BaseActionHandler__actions
        action = next(
            action
            for action in managed_actions[SelectCond.at_least_one]
            if action.text().startswith("Brightness and contrast")
        )

        panel.objview.select_objects([source, other])
        assert action.isEnabled()
        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(source)
        param.minimum, param.maximum = 64.0, 192.0
        processor.run_feature(feature, param, edit=False)

        first_result, second_result = panel.objmodel.get_all_objects()[-2:]
        np.testing.assert_array_equal(first_result.data, [[0, 0, 128, 255]])
        np.testing.assert_array_equal(second_result.data, [[0, 0, 32768, 65535]])
        assert extract_processing_parameters(first_result).source_uuid == get_uuid(
            source
        )
        assert extract_processing_parameters(second_result).source_uuid == get_uuid(
            other
        )
        np.testing.assert_array_equal(source.data, source_data)
        np.testing.assert_array_equal(other.data, other_data)

        assert panel.objprop.setup_processing_tab(first_result)
        edited = panel.objprop.processing_param_editor.dataset
        assert (edited.minimum, edited.maximum) == (64.0, 192.0)
        assert edited.histogram["domain"] == [0.0, 255.0]


def test_brightness_contrast_h5_roundtrip_and_missing_source() -> None:
    """Saved bounds remain inspectable after reload and source deletion."""
    with helpers.WorkdirRestoringTempDir() as tmpdir:
        with datalab_test_app_context(console=False) as win:
            panel = win.imagepanel
            source = create_image(
                "Source", np.array([[0, 64, 128, 255]], dtype=np.uint8)
            )
            panel.add_object(source)
            param = sigima.params.BrightnessContrastParam()
            param.update_from_obj(source)
            param.minimum, param.maximum = 64.0, 192.0
            feature = panel.processor.get_feature("adjust_brightness_contrast")
            panel.processor.run_feature(feature, param, edit=False)
            result = panel.objmodel.get_all_objects()[-1]
            source_uuid = get_uuid(source)
            result_uuid = get_uuid(result)

            filename = osp.join(tmpdir, "brightness_contrast.h5")
            win.save_h5_workspace(filename)
            panel.remove_all_objects()
            win.load_h5_workspace([filename], reset_all=True)

            loaded_source = win.find_object_by_uuid(source_uuid)
            loaded_result = win.find_object_by_uuid(result_uuid)
            assert loaded_source is not None
            assert loaded_result is not None
            assert panel.objprop.setup_processing_tab(loaded_result)
            restored = panel.objprop.processing_param_editor.dataset
            assert (restored.minimum, restored.maximum) == (64.0, 192.0)
            assert restored.histogram["domain"] == [0.0, 255.0]

            panel.objview.set_current_object(loaded_source)
            panel.remove_object(force=True)
            assert panel.objprop.setup_processing_tab(loaded_result)
            unavailable = panel.objprop.processing_param_editor.dataset
            assert (unavailable.minimum, unavailable.maximum) == (64.0, 192.0)
            assert unavailable.histogram == {}
            report = panel.objprop.apply_processing_parameters(interactive=False)
            assert not report.success
            assert "no longer exists" in report.message.lower()


def test_grouped_brightness_contrast_uses_each_source_output_range() -> None:
    """A grouped batch shares input bounds but keeps per-image output ranges."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel
        source_group = panel.add_group("Images")
        float_data = np.array([[0.0, 0.25, 0.5, 1.0]], dtype=np.float32)
        integer_data = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        float_source = create_image("Float", float_data)
        integer_source = create_image("Integer", integer_data)
        group_id = get_uuid(source_group)
        panel.add_object(float_source, group_id=group_id)
        panel.add_object(integer_source, group_id=group_id)
        panel.objview.select_groups([source_group])

        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(float_source)
        param.minimum, param.maximum = 0.25, 0.75
        feature = panel.processor.get_feature("adjust_brightness_contrast")
        panel.processor.run_feature(feature, param, edit=False)

        result_group = panel.objmodel.get_groups()[-1]
        float_result, integer_result = result_group.get_objects()
        np.testing.assert_allclose(float_result.data, [[0.0, 0.0, 0.5, 1.0]])
        np.testing.assert_array_equal(integer_result.data, [[0, 255, 255, 255]])
        assert float_result.data.dtype == np.float32
        assert integer_result.data.dtype == np.uint8
        assert extract_processing_parameters(float_result).source_uuid == get_uuid(
            float_source
        )
        assert extract_processing_parameters(integer_result).source_uuid == get_uuid(
            integer_source
        )
        np.testing.assert_array_equal(float_source.data, float_data)
        np.testing.assert_array_equal(integer_source.data, integer_data)


def test_brightness_contrast_reapply_preserves_narrow_float64_range() -> None:
    """An unchanged Processing editor round-trips exact narrow bounds."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel
        source = create_image(
            "Narrow",
            np.array([[1.0, 1.00000000000025, 1.0000000000005]]),
        )
        panel.add_object(source)
        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(source)
        expected_range = (1.0000000000001, 1.0000000000004)
        param.minimum, param.maximum = expected_range
        feature = panel.processor.get_feature("adjust_brightness_contrast")
        panel.processor.run_feature(feature, param, edit=False)
        first_result = panel.objmodel.get_all_objects()[-1]

        assert panel.objprop.setup_processing_tab(first_result)
        editor = panel.objprop.processing_param_editor
        assert editor is not None
        assert (editor.dataset.minimum, editor.dataset.maximum) == expected_range
        editor.set()

        reapplied_result = panel.objmodel.get_all_objects()[-1]
        reapplied_param = extract_processing_parameters(reapplied_result).param
        assert (reapplied_param.minimum, reapplied_param.maximum) == expected_range


def test_brightness_contrast_edit_outside_float64_domain():
    """Actual input and Apply preserve the exact next representable bound."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")
        panel = win.imagepanel
        data = np.array([[0.0, 0.5, 1.0]])
        source = create_image("Float source", data.copy())
        panel.add_object(source)
        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(source)
        feature = panel.processor.get_feature("adjust_brightness_contrast")
        panel.processor.run_feature(feature, param, edit=False)
        result = panel.objmodel.get_all_objects()[-1]
        assert panel.objprop.setup_processing_tab(result)
        editor = panel.objprop.processing_param_editor
        widget = next(
            item
            for item in editor.edit.get_terminal_widgets()
            if isinstance(item, HistogramRangeWidget)
        )
        assert widget.presentation == "brightness_contrast"
        assert not widget.brightness_slider.isHidden()
        assert not widget.contrast_slider.isHidden()
        panel.objprop.tabwidget.setCurrentWidget(panel.objprop.processing_scroll)
        win.show()
        widget.minimum_edit.setFocus()
        widget.minimum_edit.selectAll()
        QTest.keyClicks(widget.minimum_edit, "2")
        assert widget.minimum_edit.text() == "2"
        assert widget.group.isEnabled()
        QTest.keyClick(widget.minimum_edit, Qt.Key_Return)
        assert widget._range() == (2.0, np.nextafter(2.0, np.inf))
        editor.set()
        reapplied = panel.objmodel.get_all_objects()[-1]
        metadata = extract_processing_parameters(reapplied)
        assert (metadata.param.minimum, metadata.param.maximum) == widget._range()
        assert metadata.source_uuid == get_uuid(source)
        np.testing.assert_array_equal(reapplied.data, [[0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(source.data, data)


def test_brightness_contrast_cancel_preserves_caller_parameters(monkeypatch):
    """The processor owns the transactional copy edited by the Qt dialog."""
    with datalab_test_app_context(console=False) as win:
        panel = win.imagepanel
        source = create_image("Source", np.array([[0.0, 0.5, 1.0]]))
        panel.add_object(source)
        param = sigima.params.BrightnessContrastParam()
        param.update_from_obj(source)

        def cancel_edit(draft, function, sources, parent, allowed, results, **kwargs):
            assert draft is not param
            assert allowed
            dialog = DataSetEditDialog(draft, parent=parent)
            widget = next(
                item
                for item in dialog.edit_layout[0].get_terminal_widgets()
                if isinstance(item, HistogramRangeWidget)
            )
            widget._set_range(0.2, 0.8)
            dialog.edit_layout[0].accept_changes()
            preview = function(sources[0].copy(), draft)
            np.testing.assert_allclose(preview.data, [[0.0, 0.5, 1.0]])
            dialog.reject()
            return False

        monkeypatch.setattr(
            "datalab.widgets.processingpreview.edit_processing_parameters", cancel_edit
        )
        feature = panel.processor.get_feature("adjust_brightness_contrast")
        panel.processor.run_feature(feature, param, edit=True)
        assert (param.minimum, param.maximum) == (0.0, 1.0)
        assert panel.objmodel.get_all_objects() == [source]
        np.testing.assert_array_equal(source.data, [[0.0, 0.5, 1.0]])


if __name__ == "__main__":
    test_brightness_contrast_integration()
