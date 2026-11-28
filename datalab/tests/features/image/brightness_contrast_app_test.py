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
from sigima.objects import create_image, create_image_roi

from datalab.gui.actionhandler import SelectCond
from datalab.gui.processor.base import extract_processing_parameters
from datalab.objectmodel import get_uuid
from datalab.tests import datalab_test_app_context, helpers


def test_create_image_from_contrast_range(monkeypatch) -> None:
    """The contrast action uses the histogram bounds and records the operation."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")
        panel = win.imagepanel
        action = panel.contrast_to_image_action
        assert not action.isEnabled()
        data = np.array([[0, 64, 128, 255]], dtype=np.uint8)
        source = create_image("Source", data.copy())
        panel.add_object(source)
        panel.objview.select_objects([source])

        contrast = panel.plothandler.plotwidget.contrast
        contrast.histogram.range.set_range(64.0, 192.0)
        assert action.isEnabled()

        def unexpected_init(*args):
            raise AssertionError("The histogram bounds are already known")

        with monkeypatch.context() as patch:
            patch.setattr(
                sigima.params.BrightnessContrastParam,
                "update_from_obj",
                unexpected_init,
            )
            QTest.mouseClick(contrast.toolbar.widgetForAction(action), Qt.LeftButton)

        result = panel.objmodel.get_all_objects()[-1]
        assert result is not source
        metadata = extract_processing_parameters(result)
        assert metadata.source_uuid == get_uuid(source)
        assert (metadata.param.minimum, metadata.param.maximum) == (64.0, 192.0)
        np.testing.assert_array_equal(result.data, [[0, 0, 128, 255]])
        np.testing.assert_array_equal(source.data, data)
        assert panel.objprop.setup_processing_tab(result)
        edited = panel.objprop.processing_param_editor.dataset
        assert (edited.minimum, edited.maximum) == (64.0, 192.0)
        assert edited.histogram["domain"] == [0.0, 255.0]


def test_contrast_action_requires_current_histogram() -> None:
    """A missing or stale display range is never mistaken for a usable window."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")
        win.toggle_auto_refresh(False)
        panel = win.imagepanel
        action = panel.contrast_to_image_action
        a = create_image("A", np.array([[100, 128, 200]], dtype=np.uint8))
        b = create_image("B", np.array([[0.0, 0.5, 1.0]]))
        panel.add_object(a)
        assert not action.isEnabled()
        assert "refresh" in action.toolTip().lower()
        panel.manual_refresh()
        assert action.isEnabled()
        panel.add_object(b)
        assert not action.isEnabled()
        panel.create_image_from_contrast_range()
        assert panel.objmodel.get_all_objects() == [a, b]
        panel.manual_refresh()
        assert action.isEnabled()
        b.data = np.array([[1.0, 1.5, 2.0]])
        panel.refresh_plot(get_uuid(b), update_items=True)
        assert not action.isEnabled()
        panel.manual_refresh()
        assert action.isEnabled()
        panel.remove_object(force=True)
        assert not action.isEnabled()


def test_contrast_action_refuses_complex_batch() -> None:
    """Neither an FFT alone nor its position in a group starts a partial batch."""
    with datalab_test_app_context(console=False, history=True) as win:
        win.set_current_panel("image")
        panel = win.imagepanel
        action = panel.contrast_to_image_action
        for complex_first in (True, False):
            group = panel.add_group("Mixed")
            real = create_image("Real", np.array([[0, 2]], dtype=np.uint8))
            fft = create_image("FFT", np.array([[1 + 1j, 2 + 3j]]))
            pair = [fft, real] if complex_first else [real, fft]
            panel.add_objects(pair, group_id=get_uuid(group))
            panel.objview.select_objects([fft])
            assert not action.isEnabled()
            panel.objview.select_groups([group])
            assert not action.isEnabled()
            assert "real" in action.toolTip().lower()
            groups = panel.objmodel.get_groups()
            objects = panel.objmodel.get_all_objects()
            panel.create_image_from_contrast_range()
            assert panel.objmodel.get_groups() == groups
            assert panel.objmodel.get_all_objects() == objects
        panel.remove_all_objects()
        assert not action.isEnabled()


def test_contrast_action_recovers_after_invalid_bounds(monkeypatch) -> None:
    """Dragging the handles or synchronizing the ROI editor updates readiness."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")
        panel = win.imagepanel
        src = create_image("Float", np.array([[0.0, 0.5, 1.0]]))
        panel.add_object(src)
        action = panel.contrast_to_image_action
        bounds = panel.plothandler.plotwidget.contrast.histogram.range
        for low in (1.0, 2.0):
            bounds.set_range(0.0, 1.0)
            bounds.move_point_to(0, (low, 0.0))
            assert not action.isEnabled()
            assert "minimum" in action.toolTip().lower()
            panel.apply_shared_contrast(src, 0.0, 1.0)
            assert action.isEnabled()
        with monkeypatch.context() as patch:
            patch.setattr(bounds, "get_range", lambda: (0.0, np.inf))
            panel.create_image_from_contrast_range()
            assert not action.isEnabled()
            assert panel.objmodel.get_all_objects() == [src]


def test_contrast_action_processes_group_from_one_visible_image() -> None:
    """The visible window serves all real sources, respecting their dtype and ROI."""
    with datalab_test_app_context(console=False) as win:
        win.set_current_panel("image")
        win.toggle_show_first_only(True)
        panel = win.imagepanel
        group = panel.add_group("Images")
        data = np.array([[10, 64, 96, 128, 240]] * 3, np.uint8)
        a = create_image("ROI", data.copy())
        a.roi = create_image_roi("rectangle", [1, 0, 3, 3], indices=True)
        b = create_image("Float", np.array([[0.0, 64.0, 96.0, 128.0, 255.0]]))
        panel.add_objects([a, b], group_id=get_uuid(group))
        panel.objview.select_groups([group])
        histogram = panel.plothandler.plotwidget.contrast.histogram
        histogram.set_range(64.0, 128.0)
        assert panel.contrast_to_image_action.isEnabled()
        panel.contrast_to_image_action.trigger()
        dst_a, dst_b = panel.objmodel.get_groups()[-1].get_objects()
        np.testing.assert_array_equal(dst_a.data, [[10, 0, 128, 255, 240]] * 3)
        np.testing.assert_array_equal(dst_b.data, [[0.0, 0.0, 127.5, 255.0, 255.0]])
        np.testing.assert_array_equal(a.data, data)
        for src, dst in ((a, dst_a), (b, dst_b)):
            metadata = extract_processing_parameters(dst)
            assert metadata.source_uuid == get_uuid(src)
            assert (metadata.param.minimum, metadata.param.maximum) == (64.0, 128.0)
            assert dst.data.dtype == src.data.dtype


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
