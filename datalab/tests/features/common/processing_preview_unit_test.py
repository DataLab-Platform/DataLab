"""Speculative requests cannot publish workspace objects or stale parameters."""

from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest
from guidata.qthelpers import qt_app_context
from sigima.objects import create_signal
from sigima.params import GaussianParam
from sigima.proc.signal import gaussian_filter

from datalab.gui.processor.catcher import CompOut
from datalab.gui.processor.preview import PreviewController
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
        controller.close()
        assert executor.closed


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
        field = dialog.edit_layout.get_terminal_widgets()[0]
        field.edit.setText("2.5")
        assert param.sigma == 1.0
        dialog.preview.enabled.setChecked(True)
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
        dialog.reject()
        assert param.sigma == 1.0
        assert executor.closed
        assert not dialog.preview._timer.isActive()
        accepted = ProcessingPreviewDialog(param, gaussian_filter, [source])
        accepted.edit_layout.get_terminal_widgets()[0].edit.setText("3.5")
        accepted.accept()
        assert param.sigma == 3.5
        assert accepted.preview.controller._executor is None


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
            prop._ObjectProp__set_auto_recompute_enabled(True)
            timer = prop._ObjectProp__auto_recompute_timer
            field = editor.edit.get_terminal_widgets()[0]
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
    """Apply creates a result normally, but edits in place in history edit mode."""
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
            assert len(panel.objmodel) == (2 if edit_mode else 3)
            assert len(window.historypanel) == history_count + (0 if edit_mode else 1)
            result = (
                panel.objmodel[original_id]
                if edit_mode
                else panel.objview.get_current_object()
            )
            np.testing.assert_allclose(result.y, gaussian_filter(source, sigma=3.0).y)
