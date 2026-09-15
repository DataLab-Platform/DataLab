"""Speculative requests cannot publish workspace objects or stale parameters."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW
from sigima.objects import create_signal
from sigima.params import GaussianParam
from sigima.proc.signal import gaussian_filter

from datalab.gui.processor.catcher import CompOut
from datalab.gui.processor.preview import PreviewController, PreviewExecutorCache
from datalab.objectmodel import set_number
from datalab.widgets.processingpreview import ProcessingPreviewDialog


class FakeExecutor:
    """Manually completed tasks make request ordering deterministic."""

    def __init__(self):
        self.requests = []
        self.closed = False

    def submit(self, function, args):
        future = Future()
        self.requests.append((future, function, args))
        return future

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def drain_pending_qt_timers():
    """Prevent unattended close timers from affecting the next preview test."""
    yield
    if QW.QApplication.instance() is not None:
        for _index in range(3):
            QW.QApplication.processEvents()


def test_preview_latest_request_and_invalidation():
    """Only the latest pending request runs; invalid data discards old results."""
    with qt_app_context():
        executor = FakeExecutor()
        controller = PreviewController(
            gaussian_filter, executor_factory=lambda: executor
        )
        outputs = []
        errors = []
        controller.SIG_RESULT.connect(
            lambda result, current: outputs.append((result, current))
        )
        controller.SIG_ERROR.connect(errors.append)
        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        param = GaussianParam.create(sigma=1.0)
        controller.request(source, param)
        assert not executor.requests
        controller.set_enabled(True)
        controller.request(source, param)
        param.sigma = 2.0
        controller.request(source, param)
        param.sigma = 3.0
        controller.request(source, param)
        assert len(executor.requests) == 1
        assert executor.requests[0][2][1].sigma == 1.0
        executor.requests[0][2][0].y[:] = 99
        assert not np.all(source.y == 99)
        executor.requests[0][0].set_result(CompOut(result=source.copy()))
        controller.poll()
        assert outputs[-1][1] is False
        assert len(executor.requests) == 2
        assert executor.requests[-1][2][1].sigma == 3.0
        controller.invalidate()
        executor.requests[-1][0].set_result(CompOut(error_msg="obsolete"))
        controller.poll()
        assert not errors
        assert len(outputs) == 1
        controller.request(source, param)
        executor.requests[-1][0].set_result(CompOut(result=source.copy()))
        controller.poll()
        assert outputs[-1][1] is True
        assert controller.take_current_result(source) is outputs[-1][0]
        assert controller.take_current_result(source) is None
        controller.request(source, param)
        executor.requests[-1][0].set_result(CompOut(result=source.copy()))
        controller.poll()
        controller.mark_dirty()
        assert controller.take_current_result(source) is None
        controller.close()
        assert executor.closed


def test_controller_caches_only_completed_executors():
    """Completed work is reusable while active work keeps cancellation semantics."""
    with qt_app_context():
        executors = []

        def create_executor():
            executor = FakeExecutor()
            executors.append(executor)
            return executor

        cache = PreviewExecutorCache(create_executor)
        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        param = GaussianParam.create(sigma=1.0)

        completed = PreviewController(gaussian_filter, executor_cache=cache)
        completed.set_enabled(True)
        completed.request(source, param)
        executors[0].requests[0][0].set_result(CompOut(result=source.copy()))
        completed.close()

        active = PreviewController(gaussian_filter, executor_cache=cache)
        active.set_enabled(True)
        active.request(source, param)
        assert len(executors) == 1
        active.close()
        assert executors[0].closed

        replacement = PreviewController(gaussian_filter, executor_cache=cache)
        replacement.set_enabled(True)
        replacement.request(source, param)
        assert len(executors) == 2
        replacement.close()
        cache.close()


def test_dialog_is_opt_in_and_transactional():
    """Editing and rendering stay private until OK, including a source switch."""
    from sigima.objects import create_image

    with qt_app_context():
        executor = FakeExecutor()
        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        second = create_signal("Other", np.arange(10.0), np.zeros(10))
        set_number(source, 1)
        set_number(second, 2)
        param = GaussianParam.create(sigma=1.0)
        dialog = ProcessingPreviewDialog(
            param,
            gaussian_filter,
            [source, second],
            controller_factory=lambda function, parent: PreviewController(
                function, parent, executor_factory=lambda: executor
            ),
        )
        assert not executor.requests
        assert dialog.preview.plotwidget is not None
        assert not dialog.preview.plotwidget.isHidden()
        assert not dialog.preview.disabled_overlay.isHidden()
        np.testing.assert_allclose(dialog.preview.item.get_data()[1], source.y)
        field = dialog.edit_layout.get_terminal_widgets()[0]
        field.edit.setText("2.5")
        assert param.sigma == 1.0
        dialog.preview.enabled.setChecked(True)
        assert dialog.preview.disabled_overlay.isHidden()
        assert len(executor.requests) == 1
        assert executor.requests[0][2][1].sigma == 2.5
        result = gaussian_filter(source, dialog.instance)
        executor.requests[0][0].set_result(CompOut(result=result))
        dialog.preview.controller.poll()
        np.testing.assert_allclose(dialog.preview.item.get_data()[1], result.y)
        dialog.preview.source_combo.setCurrentIndex(1)
        assert dialog.instance.sigma == 2.5
        dialog.preview._timer.stop()
        dialog.preview._request()
        np.testing.assert_array_equal(executor.requests[-1][2][0].y, second.y)
        dialog.preview._show_result(
            CompOut(result=create_image("Image", np.arange(12.0).reshape(3, 4))), True
        )
        np.testing.assert_array_equal(
            dialog.preview.item.data, np.arange(12.0).reshape(3, 4)
        )
        dialog.preview.enabled.setChecked(False)
        assert not dialog.preview.plotwidget.isHidden()
        assert not dialog.preview.disabled_overlay.isHidden()
        np.testing.assert_array_equal(
            dialog.preview.item.data, np.arange(12.0).reshape(3, 4)
        )
        dialog.reject()
        assert param.sigma == 1.0
        assert executor.closed
        assert not dialog.preview._timer.isActive()
        accepted = ProcessingPreviewDialog(param, gaussian_filter, [source])
        accepted.edit_layout.get_terminal_widgets()[0].edit.setText("3.5")
        accepted.accept()
        assert param.sigma == 3.5
        assert accepted.preview.controller._executor is None


def test_preview_busy_overlay_tracks_request_queue():
    """Delayed progress avoids flicker and covers the full request queue."""
    from qtpy.QtTest import QTest

    with qt_app_context():
        executor = FakeExecutor()
        source = create_signal("Source", np.arange(10.0), np.arange(10.0))
        set_number(source, 1)
        dialog = ProcessingPreviewDialog(
            GaussianParam.create(sigma=1.0),
            gaussian_filter,
            [source],
            controller_factory=lambda function, parent: PreviewController(
                function, parent, executor_factory=lambda: executor
            ),
        )
        preview = dialog.preview
        assert preview.busy_overlay.isHidden()
        assert preview.busy_progress.minimum() == 0
        assert preview.busy_progress.maximum() == 0

        preview.enabled.setChecked(True)
        assert preview.busy_overlay.isHidden()
        assert preview._busy_overlay_timer.isActive()
        executor.requests[0][0].set_result(CompOut(result=source.copy()))
        preview.controller.poll()
        assert preview.busy_overlay.isHidden()
        assert not preview._busy_overlay_timer.isActive()
        QTest.qWait(preview._busy_overlay_timer.interval() + 50)
        assert preview.busy_overlay.isHidden()

        field = dialog.edit_layout.get_terminal_widgets()[0]
        field.edit.setText("2.0")
        preview._timer.stop()
        preview._request()
        assert preview.busy_overlay.isHidden()
        QTest.qWait(preview._busy_overlay_timer.interval() + 50)
        assert not preview.busy_overlay.isHidden()

        field.edit.setText("3.0")
        preview._timer.stop()
        preview._request()
        executor.requests[1][0].set_result(CompOut(result=source.copy()))
        preview.controller.poll()
        assert len(executor.requests) == 3
        assert not preview.busy_overlay.isHidden()

        executor.requests[2][0].set_result(CompOut(result=source.copy()))
        preview.controller.poll()
        assert preview.busy_overlay.isHidden()

        field.edit.setText("4.0")
        preview._timer.stop()
        preview._request()
        preview._show_busy_overlay()
        assert not preview.busy_overlay.isHidden()
        executor.requests[3][0].set_result(CompOut(error_msg="preview error"))
        preview.controller.poll()
        assert preview.busy_overlay.isHidden()
        assert not preview.details.isHidden()

        field.edit.setText("5.0")
        preview._timer.stop()
        preview._request()
        assert preview._busy_overlay_timer.isActive()
        preview.close_preview()
        assert preview.busy_overlay.isHidden()
        assert not preview._busy_overlay_timer.isActive()
        assert executor.closed
        dialog.reject()


def test_processor_cancel_and_accept(monkeypatch):
    """Cancel keeps defaults and objects; OK uses normal processing for the lot."""
    from datalab.config import Conf
    from datalab.tests import datalab_test_app_context
    from datalab.widgets import processingpreview

    with qt_app_context(), Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context(history=True) as window:
            panel = window.signalpanel
            source = create_signal("Source", np.arange(20.0), np.sin(np.arange(20.0)))
            other = create_signal("Other", np.arange(20.0), np.cos(np.arange(20.0)))
            panel.add_object(source)
            panel.add_object(other)
            panel.objview.select_objects([1, 2])
            window.historypanel.toggle_record_mode(True)
            history_count = len(window.historypanel)
            processor = panel.processor
            defaults = GaussianParam.create(sigma=1.2)
            monkeypatch.setitem(processor.PARAM_DEFAULTS, "GaussianParam", defaults)
            before = panel.objmodel.get_object_ids()

            def reject(dialog):
                assert len(dialog.preview.sources) == 2
                dialog.edit_layout.get_terminal_widgets()[0].edit.setText("4.0")
                dialog.reject()
                return 0

            monkeypatch.setattr(processingpreview, "exec_dialog", reject)
            processor.run_feature("gaussian_filter")
            assert panel.objmodel.get_object_ids() == before
            assert len(window.historypanel) == history_count
            assert processor.PARAM_DEFAULTS["GaussianParam"] is defaults
            assert defaults.sigma == 1.2

            def accept(dialog):
                dialog.edit_layout.get_terminal_widgets()[0].edit.setText("2.0")
                dialog.preview.source_combo.setCurrentIndex(1)
                panel.objview.select_objects([2])
                dialog.accept()
                return 1

            monkeypatch.setattr(processingpreview, "exec_dialog", accept)
            processor.run_feature("gaussian_filter")
            assert len(panel.objmodel) == 4
            assert len(window.historypanel) == history_count + 1
            assert processor.PARAM_DEFAULTS["GaussianParam"].sigma == 2.0
            results = [
                panel.objmodel[uid] for uid in panel.objmodel.get_object_ids()[2:]
            ]
            for original, result in zip([source, other], results):
                np.testing.assert_allclose(
                    result.y, gaussian_filter(original, sigma=2.0).y
                )
            remembered = processor.PARAM_DEFAULTS["GaussianParam"]
            processor.run_feature(
                "gaussian_filter", GaussianParam.create(sigma=5.0), edit=False
            )
            assert processor.PARAM_DEFAULTS["GaussianParam"] is remembered
            assert remembered.sigma == 2.0


def test_processor_reuses_completed_executor_between_dialogs(monkeypatch):
    """Successive processor dialogs share the window's idle executor."""
    from datalab.config import Conf
    from datalab.tests import datalab_test_app_context
    from datalab.widgets import processingpreview

    executor = FakeExecutor()
    with qt_app_context(), Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context() as window:
            panel = window.signalpanel
            source = create_signal("Source", np.arange(20.0), np.sin(np.arange(20.0)))
            panel.add_object(source)
            window.preview_executor_cache.reset()
            window.preview_executor_cache._executor_factory = lambda: executor

            def complete_and_reject(dialog):
                dialog.preview.enabled.setChecked(True)
                future = executor.requests[-1][0]
                future.set_result(
                    CompOut(result=gaussian_filter(source, dialog.instance))
                )
                dialog.preview.controller.poll()
                dialog.reject()
                return 0

            monkeypatch.setattr(processingpreview, "exec_dialog", complete_and_reject)
            for sigma in (1.0, 2.0):
                panel.processor.compute_1_to_1(
                    gaussian_filter,
                    param=GaussianParam.create(sigma=sigma),
                    title="Gaussian filter",
                    edit=True,
                )

            assert len(executor.requests) == 2
            assert not executor.closed
        assert executor.closed


def test_processor_reuses_only_current_single_object_preview(monkeypatch):
    """OK reuses one current preview but computes a multi-selection normally."""
    from datalab.config import Conf
    from datalab.tests import datalab_test_app_context
    from datalab.widgets import processingpreview

    with qt_app_context(), Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context(history=True) as window:
            panel = window.signalpanel
            source = create_signal("Source", np.arange(20.0), np.sin(np.arange(20.0)))
            panel.add_object(source)
            executor = FakeExecutor()
            window.preview_executor_cache.reset()
            window.preview_executor_cache._executor_factory = lambda: executor
            nominal_calls = []

            def counted_filter(src, param):
                nominal_calls.append((src, param))
                return gaussian_filter(src, param)

            def accept_current_preview(dialog):
                dialog.preview.enabled.setChecked(True)
                assert len(executor.requests) == 1
                preview_result = gaussian_filter(source, dialog.instance)
                executor.requests[0][0].set_result(CompOut(result=preview_result))
                dialog.preview.controller.poll()
                dialog.accept()
                return 1

            monkeypatch.setattr(
                processingpreview, "exec_dialog", accept_current_preview
            )
            window.historypanel.toggle_record_mode(True)
            history_count = len(window.historypanel)
            panel.processor.compute_1_to_1(
                counted_filter,
                param=GaussianParam.create(sigma=2.0),
                title="Gaussian filter",
                edit=True,
            )

            assert nominal_calls == []
            assert len(panel.objmodel) == 2
            assert len(window.historypanel) == history_count + 1
            result = panel.objmodel[panel.objmodel.get_object_ids()[-1]]
            np.testing.assert_allclose(result.y, gaussian_filter(source, sigma=2.0).y)

            other = create_signal("Other", np.arange(20.0), np.cos(np.arange(20.0)))
            panel.add_object(other)
            panel.objview.select_objects([source, other])
            multi_executor = FakeExecutor()
            window.preview_executor_cache.reset()
            window.preview_executor_cache._executor_factory = lambda: multi_executor

            def accept_multi_preview(dialog):
                dialog.preview.enabled.setChecked(True)
                assert len(multi_executor.requests) == 1
                preview_source = dialog.preview.sources[0]
                preview_result = gaussian_filter(preview_source, dialog.instance)
                multi_executor.requests[0][0].set_result(CompOut(result=preview_result))
                dialog.preview.controller.poll()
                dialog.accept()
                return 1

            monkeypatch.setattr(processingpreview, "exec_dialog", accept_multi_preview)
            panel.processor.compute_1_to_1(
                counted_filter,
                param=GaussianParam.create(sigma=3.0),
                title="Gaussian filter",
                edit=True,
            )

            assert len(nominal_calls) == 2
            assert nominal_calls[0][0] is source
            assert nominal_calls[1][0] is other


def test_special_dialog_preserves_counters_and_validation():
    """The special form keeps its decorations and integer-image guard."""
    from sigima.objects import create_image
    from sigima.proc.signal import replace_special_values

    from datalab.widgets.replacespecialvalues import (
        ReplaceSpecialValuesImageParamDL,
        ReplaceSpecialValuesSignalParamDL,
    )

    with qt_app_context():
        source = create_signal("Source", np.arange(5.0), np.array([1, np.nan, 2, 3, 4]))
        set_number(source, 1)
        param = ReplaceSpecialValuesSignalParamDL()
        param.update_from_obj(source)
        dialog = param.create_dialog()
        dialog.attach_preview(replace_special_values, [source])
        assert "<b>1</b>" in dialog._count_badges["nan"].text()
        assert dialog.preview.editor is dialog.edit_layout
        assert not dialog.preview.enabled.isChecked()
        assert len(dialog._kernel_previews) == 3
        dialog.reject()
        image = create_image("Integer", np.ones((4, 4), dtype=np.uint16))
        set_number(image, 1)
        image_param = ReplaceSpecialValuesImageParamDL()
        image_param.update_from_obj(image)
        blocked = image_param.create_dialog()
        blocked.attach_preview(replace_special_values, [image])
        assert not blocked.preview.enabled.isEnabled()
        blocked.accept()
        assert blocked.result() == 0
        blocked.reject()


def test_processing_tab_debounces_valid_released_editor(monkeypatch):
    """A drag or invalid field never applies, and old editors cannot restart it."""
    from datalab.config import Conf
    from datalab.tests import datalab_test_app_context

    with qt_app_context(), Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context() as window:
            panel = window.signalpanel
            panel.add_object(
                create_signal("Source", np.arange(20.0), np.sin(np.arange(20.0)))
            )
            panel.processor.run_feature(
                "gaussian_filter", GaussianParam.create(sigma=1.0)
            )
            prop = panel.objprop
            editor = prop.processing_param_editor
            applied = []
            editor.SIG_APPLY_BUTTON_CLICKED.disconnect()
            editor.SIG_APPLY_BUTTON_CLICKED.connect(lambda: applied.append(True))
            auto_cb = editor.findChild(QW.QCheckBox, "auto_recompute_on_edit")
            assert auto_cb is not None
            assert not auto_cb.icon().isNull()
            form_layout = editor.edit.layout
            auto_row, auto_column, _row_span, auto_column_span = (
                form_layout.getItemPosition(form_layout.indexOf(auto_cb))
            )
            apply_row, _column, _row_span, _column_span = form_layout.getItemPosition(
                form_layout.indexOf(editor.apply_button)
            )
            field = editor.edit.get_terminal_widgets()[0]
            _row, field_column, _row_span, _column_span = form_layout.getItemPosition(
                form_layout.indexOf(field.group)
            )
            assert auto_row == apply_row + 1
            assert auto_column == field_column
            assert auto_column_span == form_layout.columnCount() - auto_column
            auto_cb.setChecked(True)
            timer = prop._ObjectProp__auto_recompute_timer
            field.edit.setText("2.0")
            assert timer.isActive()
            editor._slider_gesture(True)
            assert not timer.isActive()
            field.edit.setText("3.0")
            assert not timer.isActive()
            editor._slider_gesture(False)
            assert timer.isActive()
            field.edit.setText("-")
            assert not timer.isActive()
            editor.set()
            assert not applied
            field.edit.setText("4.0")
            timer.stop()
            prop._ObjectProp__auto_recompute_trigger()
            assert applied == [True]
            assert editor.dataset.sigma == 4.0
            field.edit.setText("5.0")
            assert timer.isActive()
            panel.objview.select_objects([1])
            assert not timer.isActive()
            editor.change_callback()
            assert not timer.isActive()


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("record_history", [False, True])
def test_processing_tab_updates_result_in_place(automatic, record_history):
    """Real Apply and timer callbacks update the selected result without publishing."""
    from qtpy import QtCore as QC
    from qtpy.QtTest import QTest

    from datalab.gui.processor.base import extract_processing_parameters
    from datalab.objectmodel import get_uuid
    from datalab.tests import datalab_test_app_context

    with qt_app_context():
        with datalab_test_app_context(history=True) as window:
            panel = window.signalpanel
            panel.processor.set_process_isolation_enabled(False)
            history = window.historypanel
            history.toggle_record_mode(record_history)
            source = create_signal("Source", np.arange(20.0), np.sin(np.arange(20.0)))
            source_data = source.xydata.copy()
            panel.add_object(source)
            panel.processor.run_feature(
                "gaussian_filter", GaussianParam.create(sigma=1.0)
            )
            prop = panel.objprop
            result = prop.current_processing_obj
            result_uuid = get_uuid(result)
            result.metadata["user_marker"] = 123
            history_count = len(history)
            action = history.find_action_for_output(result_uuid, "gaussian_filter")
            assert not history.is_edit_mode()
            for sigma in (2.0, 3.0):
                window.set_modified(False)
                editor = prop.processing_param_editor
                editor.findChild(QW.QCheckBox, "auto_recompute_on_edit").setChecked(
                    automatic
                )
                editor.edit.get_terminal_widgets()[0].edit.setText(str(sigma))
                if automatic:
                    for _attempt in range(100):
                        QTest.qWait(20)
                        if extract_processing_parameters(result).param.sigma == sigma:
                            break
                else:
                    editor.apply_button.click()
                assert len(panel.objmodel) == 2
                assert window.is_modified()
                assert panel.objmodel[result_uuid] is result
                assert panel.objview.get_current_object() is result
                assert result.metadata["user_marker"] == 123
                np.testing.assert_allclose(
                    result.y, gaussian_filter(source, sigma=sigma).y
                )
                np.testing.assert_array_equal(source.xydata, source_data)
                assert extract_processing_parameters(result).param.sigma == sigma
                assert len(history) == history_count
                if action is not None:
                    assert action.kwargs["param"].sigma == 1.0
                    assert not action.has_pending_edits
                QW.QApplication.processEvents()
                QW.QApplication.sendPostedEvents(None, QC.QEvent.DeferredDelete)
                assert len(prop.findChildren(type(prop.processing_param_editor))) == 1


@pytest.mark.parametrize(
    "invalid_kind", ["image_1d", "empty", "signal_rows", "coords", "pitch"]
)
def test_recompute_rejects_invalid_output(monkeypatch, invalid_kind):
    """Invalid scientific output must not mutate a saved destination."""
    import copy

    from sigima.objects import create_image
    from sigima.params import ConstantParam

    from datalab.gui.processor.catcher import CompOut
    from datalab.tests import datalab_test_app_context

    with qt_app_context(), datalab_test_app_context() as window:
        is_signal = invalid_kind == "signal_rows"
        panel = window.signalpanel if is_signal else window.imagepanel
        panel.processor.set_process_isolation_enabled(False)
        source = (
            create_signal("Source", np.arange(5.0), np.ones(5))
            if is_signal
            else create_image("Source", np.ones((4, 4)))
        )
        panel.add_object(source)
        panel.processor.run_feature(
            "addition_constant", ConstantParam.create(value=1.0)
        )
        target = panel.objview.get_current_object()
        original_data = target.xydata.copy() if is_signal else target.data.copy()
        original_metadata = copy.deepcopy(target.metadata)
        original_title = target.title
        invalid = target.copy()
        invalid.title = "Invalid result"
        if invalid_kind == "signal_rows":
            invalid.xydata = np.ones((5, 5))
        elif invalid_kind == "image_1d":
            invalid.data = np.ones(5)
        elif invalid_kind == "empty":
            invalid.data = np.empty((0, 4))
        elif invalid_kind == "coords":
            invalid.set_coords(np.arange(3.0), np.arange(4.0))
        else:
            invalid.dx = 0.0
        monkeypatch.setattr(
            panel.processor,
            "recompute_1_to_1",
            lambda *args, **kwargs: CompOut(result=invalid),
        )
        window.set_modified(False)
        report = panel.objprop.apply_processing_parameters(
            param=ConstantParam.create(value=9.0)
        )
        assert not report.success
        assert not window.is_modified()
        assert target.title == original_title
        assert target.metadata == original_metadata
        np.testing.assert_array_equal(
            target.xydata if is_signal else target.data, original_data
        )


def test_recompute_signal_roi_and_selection(monkeypatch):
    """Resampling invalidates ROI masks without restoring a previous selection."""
    from sigima.objects import create_signal_roi
    from sigima.params import Resampling1DParam

    from datalab.objectmodel import get_uuid
    from datalab.tests import datalab_test_app_context

    with qt_app_context(), datalab_test_app_context() as window:
        panel = window.signalpanel
        panel.processor.set_process_isolation_enabled(False)
        source = create_signal(
            "Source", np.linspace(0, 10, 20), np.sin(np.linspace(0, 10, 20))
        )
        panel.add_object(source)
        panel.processor.run_feature(
            "resampling",
            Resampling1DParam.create(mode="nbpts", xmin=0.0, xmax=10.0, nbpts=10),
        )
        target = panel.objview.get_current_object()
        target.roi = create_signal_roi([2.0, 7.0])
        assert target.maskdata.shape == (2, 10)
        report = panel.objprop.apply_processing_parameters(
            param=Resampling1DParam.create(mode="nbpts", xmin=0.0, xmax=10.0, nbpts=30)
        )
        assert report.success
        np.testing.assert_array_equal(target.maskdata, target.roi.to_mask(target))
        filtered = gaussian_filter(target, sigma=2.0)
        np.testing.assert_array_equal(
            filtered.xydata[target.maskdata], target.xydata[target.maskdata]
        )
        QW.QApplication.processEvents()
        original_recompute = panel.processor.recompute_1_to_1

        def recompute_and_switch(*args, **kwargs):
            output = original_recompute(*args, **kwargs)
            panel.objview.select_objects([get_uuid(source)])
            return output

        monkeypatch.setattr(panel.processor, "recompute_1_to_1", recompute_and_switch)
        report = panel.objprop.apply_processing_parameters(
            param=Resampling1DParam.create(mode="nbpts", xmin=0.0, xmax=10.0, nbpts=40)
        )
        QW.QApplication.processEvents()
        assert report.success
        assert panel.objview.get_current_object() is source
        assert not panel.plothandler[get_uuid(target)].isVisible()
        assert (
            panel.plothandler.plot.get_active_item()
            is panel.plothandler[get_uuid(source)]
        )
        panel.objview.select_objects([get_uuid(target)])
        np.testing.assert_array_equal(
            panel.plothandler[get_uuid(target)].get_data()[0], target.x
        )


def test_processing_result_scientific_state():
    """Copy scientific fields without replacing destination-owned properties."""
    from sigima.objects import create_image, create_signal

    from datalab.gui.processor.base import ProcessingParameters, apply_processing_result

    parameters = ProcessingParameters("test", "1-to-1")
    target = create_signal("Old", np.arange(4.0), np.zeros(4))
    target.metadata["user"] = {"value": 42}
    target.annotations = "retained"
    fresh = create_signal("New", np.arange(4.0), np.ones(4))
    fresh.set_xydata(fresh.x, fresh.y, np.ones(4), np.full(4, 2.0))
    fresh.xlabel, fresh.xunit = "Time", "s"
    apply_processing_result(target, fresh, parameters)
    np.testing.assert_array_equal(target.xydata, fresh.xydata)
    assert not np.shares_memory(target.xydata, fresh.xydata)
    assert (target.xlabel, target.xunit) == ("Time", "s")
    assert target.metadata["user"] == {"value": 42}
    assert target.annotations == "retained"

    image = create_image("Old", np.zeros((2, 3)))
    image.metadata["user"] = 42
    fresh_image = create_image("New", np.ones((3, 4)))
    fresh_image.set_coords(np.array([0.0, 1.0, 3.0, 6.0]), np.array([0.0, 2.0, 5.0]))
    apply_processing_result(image, fresh_image, parameters)
    assert not image.is_uniform_coords
    np.testing.assert_array_equal(image.xcoords, fresh_image.xcoords)
    assert not np.shares_memory(image.xcoords, fresh_image.xcoords)
    assert image.metadata["user"] == 42
    snapshot = image.data.copy()
    with pytest.raises(TypeError):
        apply_processing_result(image, fresh, parameters)
    np.testing.assert_array_equal(image.data, snapshot)
    fresh_image.set_uniform_coords(2.0, 3.0, 4.0, 5.0)
    apply_processing_result(image, fresh_image, parameters)
    assert image.is_uniform_coords
    assert image.xcoords.size == image.ycoords.size == 0
    assert (image.dx, image.dy, image.x0, image.y0) == (2.0, 3.0, 4.0, 5.0)


def test_recompute_binning_coordinates():
    """Reprocessing keeps pixel geometry consistent with the freshly computed data."""
    from sigima.objects import create_image
    from sigima.params import BinningParam
    from sigima.proc.image import binning

    from datalab.objectmodel import get_uuid
    from datalab.tests import datalab_test_app_context

    with qt_app_context(), datalab_test_app_context() as window:
        panel = window.imagepanel
        panel.processor.set_process_isolation_enabled(False)
        source = create_image("Source", np.arange(144.0).reshape(12, 12))
        source.set_uniform_coords(0.5, 2.0, 10.0, -3.0)
        panel.add_object(source)
        panel.processor.run_feature(
            "binning", BinningParam.create(sx=2, sy=2, change_pixel_size=True)
        )
        result = panel.objview.get_current_object()
        result_uuid = get_uuid(result)
        param = BinningParam.create(sx=3, sy=4, change_pixel_size=True)
        expected = binning(source, param)
        report = panel.objprop.apply_processing_parameters(param=param)
        assert report.success
        assert panel.objmodel[result_uuid] is result
        np.testing.assert_allclose(result.data, expected.data)
        assert (result.dx, result.dy, result.x0, result.y0) == (
            expected.dx,
            expected.dy,
            expected.x0,
            expected.y0,
        )


def test_preview_preserves_custom_editors_and_backends(monkeypatch):
    """Unknown editors, alternate backends and feature vetoes retain their path."""
    from guidata.dataset import backends

    from datalab.widgets.processingpreview import edit_processing_parameters

    called = []

    class CustomParam(GaussianParam):
        def edit(self, parent=None):
            called.append(self)
            return 0

    custom = CustomParam()
    assert not edit_processing_parameters(custom, gaussian_filter, [], None)
    assert called == [custom]
    param = GaussianParam()
    monkeypatch.setattr(param, "edit", lambda **kwargs: called.append(param) or 1)
    assert edit_processing_parameters(param, gaussian_filter, [], None, False)
    original = backends.get_handler("edit_dataset")
    try:
        backends.set_handler(
            "edit_dataset", lambda instance, **kwargs: called.append(instance) or 1
        )
        other = GaussianParam()
        assert edit_processing_parameters(other, gaussian_filter, [], None)
        assert called[-1] is other
    finally:
        if original is None:
            backends.clear_handler("edit_dataset")
        else:
            backends.set_handler("edit_dataset", original)


def test_live_image_to_signal_preview(tmp_path, monkeypatch):
    """A Qt click drives a real spawn round-trip into a visible PlotPy curve."""
    from qtpy import QtCore as QC
    from qtpy import QtWidgets as QW
    from qtpy.QtTest import QTest
    from sigima.objects import create_image, create_image_roi
    from sigima.proc.image import LineProfileParam, line_profile

    monkeypatch.setattr(
        "guidata.qthelpers.close_widgets_and_quit", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "sigimax.utils.qthelpers.close_widgets_and_quit", lambda **kwargs: None
    )
    with qt_app_context():
        QW.QApplication.processEvents()
        source = create_image("Image", np.arange(120, dtype=np.uint16).reshape(10, 12))
        source.roi = create_image_roi("rectangle", [2, 1, 8, 8], indices=True)
        source.x0, source.y0, source.dx, source.dy = 10.0, -4.0, 0.25, 0.5
        source.xlabel, source.xunit = "Position", "mm"
        source.zlabel, source.zunit = "Intensity", "a.u."
        set_number(source, 1)
        param = LineProfileParam.create(direction="horizontal", row=3)
        expected = line_profile(
            source.copy(), LineProfileParam.create(direction="horizontal", row=3)
        )
        assert expected.x.size == 8
        dialog = ProcessingPreviewDialog(param, line_profile, [source])
        loop = QW.QApplication.instance()
        timeout = QC.QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(loop.quit)
        dialog.preview.controller.SIG_RESULT.connect(loop.quit)
        dialog.preview.controller.SIG_ERROR.connect(loop.quit)
        dialog.show()
        executor = None
        try:
            assert dialog.preview.controller._executor is None
            QTest.mouseClick(
                dialog.preview.enabled,
                QC.Qt.LeftButton,
                pos=QC.QPoint(8, dialog.preview.enabled.height() // 2),
            )
            assert dialog.preview.enabled.isChecked()
            executor = dialog.preview.controller._executor
            assert executor is not None
            timeout.start(30000)
            loop.exec_()
            timeout.stop()
            assert dialog.preview.item is not None, dialog.preview.details.toPlainText()
            assert dialog.preview.plotwidget.isVisible()
            np.testing.assert_allclose(dialog.preview.item.get_data()[0], expected.x)
            np.testing.assert_allclose(dialog.preview.item.get_data()[1], expected.y)
            assert source.data.dtype == np.uint16
            np.testing.assert_array_equal(source.data, np.arange(120).reshape(10, 12))
            assert param.row == 3
            assert dialog.grab().save(str(tmp_path / "processing-preview.png"))
        finally:
            timeout.stop()
            dialog.reject()
            if executor is not None:
                executor.close(wait=True)


@pytest.mark.parametrize("edit_mode", [False, True])
def test_processing_apply_preserves_history_modes(edit_mode):
    """Apply updates in place, changing recorded parameters only in edit mode."""
    from datalab.config import Conf
    from datalab.objectmodel import get_uuid
    from datalab.tests import datalab_test_app_context

    with qt_app_context(), Conf.process_isolation_enabled.context(False):
        with datalab_test_app_context(history=True) as window:
            panel = window.signalpanel
            source = create_signal("Source", np.arange(30.0), np.sin(np.arange(30.0)))
            panel.add_object(source)
            window.historypanel.toggle_record_mode(True)
            panel.processor.run_feature(
                "gaussian_filter", GaussianParam.create(sigma=1.0)
            )
            original = panel.objview.get_current_object()
            original_id = get_uuid(original)
            history_count = len(window.historypanel)
            window.historypanel.toggle_edit_mode(edit_mode)
            editor = panel.objprop.processing_param_editor
            editor.edit.get_terminal_widgets()[0].edit.setText("3.0")
            editor.set()
            assert len(panel.objmodel) == 2
            assert len(window.historypanel) == history_count
            result = panel.objmodel[original_id]
            assert result is original
            assert panel.objview.get_current_object() is original
            action = window.historypanel.find_action_for_output(
                original_id, "gaussian_filter"
            )
            assert action is not None
            assert action.kwargs["param"].sigma == (3.0 if edit_mode else 1.0)
            assert action.has_pending_edits == edit_mode
            np.testing.assert_allclose(result.y, gaussian_filter(source, sigma=3.0).y)
