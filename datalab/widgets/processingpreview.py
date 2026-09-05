"""Processing parameter forms with opt-in, unpublished live results."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

from guidata.configtools import get_icon
from guidata.dataset import DataSet, update_dataset
from guidata.dataset.backends import get_handler
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.qthelpers import exec_dialog, win32_fix_title_bar_background
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj
from sigimax.adapters_plotpy.objects.signal import CURVESTYLES

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import _
from datalab.gui.processor.catcher import CompOut
from datalab.gui.processor.preview import PreviewController
from datalab.objectmodel import get_short_id, patch_title_with_ids

__all__ = [
    "ProcessingPreviewDialog",
    "ProcessingPreviewWidget",
    "edit_processing_parameters",
]


def edit_processing_parameters(
    instance, function, sources, parent, preview_enabled=True
) -> bool:
    """Preserve alternate backends and custom editors outside the standard form."""
    from datalab.widgets.replacespecialvalues import (
        ReplaceSpecialValuesImageParamDL,
        ReplaceSpecialValuesSignalParamDL,
    )

    if not preview_enabled or get_handler("edit_dataset") is not None:
        return bool(instance.edit(parent=parent))
    if type(instance) in (
        ReplaceSpecialValuesSignalParamDL,
        ReplaceSpecialValuesImageParamDL,
    ):
        dialog = instance.create_dialog(parent=parent)
        dialog.attach_preview(function, sources)
    elif type(instance).edit is DataSet.edit:
        dialog = ProcessingPreviewDialog(instance, function, sources, parent)
    else:
        return bool(instance.edit(parent=parent))
    try:
        return bool(exec_dialog(dialog))
    finally:
        dialog.preview.close_preview()
        dialog.deleteLater()


class ProcessingPreviewWidget(QW.QWidget):
    """Reusable preview view, independent of form and workspace ownership."""

    def __init__(
        self,
        function: Callable,
        sources: Sequence[SignalObj | ImageObj],
        parent: QW.QWidget | None = None,
        controller_factory: Callable = PreviewController,
    ) -> None:
        super().__init__(parent)
        self.sources = tuple(sources)
        self.editor: DataSetEditLayout | None = None
        self.controller = controller_factory(function, self)
        self._dragging = False
        self._plot_kind = None
        self._plot_signature = None
        self.plotwidget: PlotWidget | None = None
        self.item = None
        self._timer = QC.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._request)
        self._layout = QW.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        controls = QW.QHBoxLayout()
        self.enabled = QW.QCheckBox(_("Preview"))
        self.enabled.setObjectName("preview_enabled")
        self.enabled.setEnabled(bool(sources))
        controls.addWidget(self.enabled)
        self.source_combo = QW.QComboBox()
        self.source_combo.setObjectName("preview_source")
        self.source_combo.setSizeAdjustPolicy(
            QW.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.source_combo.setMinimumContentsLength(16)
        self.source_combo.setToolTip(_("Preview source"))
        for source in sources:
            self.source_combo.addItem(f"{get_short_id(source)}: {source.title}")
        self.source_combo.setVisible(len(sources) > 1)
        self.source_combo.setEnabled(False)
        controls.addWidget(self.source_combo, 1)
        self._layout.addLayout(controls)
        self.status = QW.QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(QC.Qt.PlainText)
        self._layout.addWidget(self.status)
        self.details = QW.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(90)
        self.details.hide()
        self._layout.addWidget(self.details)
        self.enabled.toggled.connect(self._toggle)
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.controller.SIG_RESULT.connect(self._show_result)
        self.controller.SIG_ERROR.connect(self._show_error)

    def _toggle(self, checked: bool) -> None:
        self._timer.stop()
        self.controller.set_enabled(checked)
        self.source_combo.setEnabled(checked)
        self.details.hide()
        if self.plotwidget is not None:
            self.plotwidget.hide()
        self.status.clear()
        if checked:
            self._request()

    def _source_changed(self) -> None:
        self.controller.invalidate()
        self._plot_signature = None
        if self.plotwidget is not None:
            self.plotwidget.hide()
        self.changed()

    def changed(self) -> None:
        """Invalidate immediately, then read the form after its callbacks settle."""
        if self.editor is None or not self.enabled.isChecked():
            return
        self.controller.mark_dirty()
        self.details.hide()
        if not self.editor.check_all_values():
            self.controller.invalidate()
            self._timer.stop()
            self.status.setText(_("Invalid parameters"))
            return
        self.status.setText(_("Updating preview..."))
        if not self._dragging or not self._timer.isActive():
            self._timer.start(200 if self._dragging else 300)

    def slider_gesture(self, pressed: bool) -> None:
        """Throttle drags and request the final value when the handle is released."""
        self._dragging = pressed
        if not pressed and self.enabled.isChecked():
            self._timer.start(0)

    def _request(self) -> None:
        if self.editor is None or not self.enabled.isChecked():
            return
        if not self.editor.check_all_values():
            self.controller.invalidate()
            self.status.setText(_("Invalid parameters"))
            return
        self.editor.accept_changes()
        self.status.setText(_("Computing preview..."))
        self.controller.request(
            self.sources[self.source_combo.currentIndex()], self.editor.instance
        )

    def _show_error(self, message: str) -> None:
        self.status.setText(_("Preview failed"))
        self.details.setPlainText(message)
        self.details.show()

    def _show_result(self, output: CompOut, current: bool) -> None:
        result = output.result
        if not isinstance(result, (SignalObj, ImageObj)):
            self._show_error(_("This result cannot be previewed."))
            return
        try:
            patch_title_with_ids(
                result, [self.sources[self.source_combo.currentIndex()]], get_short_id
            )
            self._render(result)
        except Exception as error:
            self._show_error(str(error))
            return
        self.status.setText(
            _("Preview up to date") if current else _("Updating preview...")
        )
        self.details.setPlainText(output.warning_msg or "")
        self.details.setVisible(bool(output.warning_msg))

    def _render(self, result: SignalObj | ImageObj) -> None:
        kind = "image" if isinstance(result, ImageObj) else "curve"
        if self._plot_kind != kind:
            if self.plotwidget is not None:
                self._layout.removeWidget(self.plotwidget)
                self.plotwidget.hide()
                self.plotwidget.deleteLater()
            self.plotwidget = PlotWidget(self, options=PlotOptions(type=kind))
            self.plotwidget.setMinimumSize(300, 240)
            self._layout.addWidget(self.plotwidget, 1)
            self._plot_kind = kind
            self._plot_signature = None
            self.item = None
        plot = self.plotwidget.plot
        adapter = create_adapter_from_object(result)
        generator = CURVESTYLES.curve_style
        try:
            CURVESTYLES.curve_style = CURVESTYLES.style_generator()
            item = adapter.make_item()
        finally:
            CURVESTYLES.curve_style = generator
        if self.item is not None:
            plot.del_item(self.item)
        self.item = item
        plot.add_item(item)
        plot.set_active_item(item)
        item.unselect()
        plot.set_titles(
            title=result.title,
            xlabel=result.xlabel,
            xunit=result.xunit,
            ylabel=(result.ylabel, getattr(result, "zlabel", "")),
            yunit=(result.yunit, getattr(result, "zunit", "")),
        )
        for axis in ("x", "y"):
            plot.set_axis_scale(
                "bottom" if axis == "x" else "left",
                "log" if getattr(result, f"{axis}scalelog", False) else "lin",
            )
        if kind == "image":
            domain = (result.data.shape, result.x0, result.y0, result.dx, result.dy)
        else:
            domain = (
                (result.x.size, result.x[0], result.x[-1]) if result.x.size else (0,)
            )
        signature = (
            kind,
            domain,
            result.xlabel,
            result.ylabel,
            result.xunit,
            result.yunit,
        )
        if signature != self._plot_signature:
            plot.do_autoscale()
        self._plot_signature = signature
        self.plotwidget.show()
        plot.replot()

    def close_preview(self) -> None:
        """Cancel work before the containing dialog is hidden or destroyed."""
        self._timer.stop()
        self.controller.close()


class ProcessingPreviewDialog(QW.QDialog):
    """Compose a guidata form with a private preview and transactional parameters."""

    def __init__(
        self,
        instance: DataSet,
        function: Callable,
        sources: Sequence[SignalObj | ImageObj],
        parent: QW.QWidget | None = None,
        controller_factory: Callable = PreviewController,
    ) -> None:
        super().__init__(parent)
        win32_fix_title_bar_background(self)
        self.instance = copy.deepcopy(instance)
        self._original = instance
        self.setModal(True)
        self.setWindowTitle(instance.get_title())
        if instance.get_icon():
            self.setWindowIcon(get_icon(instance.get_icon()))
        self.setObjectName(instance.__class__.__name__ + "Dialog")
        layout = QW.QVBoxLayout(self)
        splitter = QW.QSplitter(QC.Qt.Horizontal)
        layout.addWidget(splitter, 1)
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        form = QW.QWidget()
        form_layout = QW.QVBoxLayout(form)
        comment = instance.get_comment()
        if comment:
            label = QW.QLabel(comment)
            label.setWordWrap(True)
            form_layout.addWidget(label)
        grid = QW.QGridLayout()
        grid.setAlignment(QC.Qt.AlignTop)
        form_layout.addLayout(grid)
        form_layout.addStretch()
        scroll.setWidget(form)
        splitter.addWidget(scroll)
        self.preview = ProcessingPreviewWidget(
            function, sources, self, controller_factory
        )
        splitter.addWidget(self.preview)
        self.edit_layout = DataSetEditLayout(
            self,
            self.instance,
            grid,
            change_callback=self.preview.changed,
            auto_sliders=True,
            slider_callback=self.preview.slider_gesture,
        )
        self.preview.editor = self.edit_layout
        self.buttons = QW.QDialogButtonBox(
            QW.QDialogButtonBox.Ok | QW.QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.finished.connect(self.preview.close_preview)
        self.resize(1000, 650)
        screen = self.screen().availableGeometry()
        self.resize(
            min(self.width(), screen.width()), min(self.height(), screen.height())
        )
        splitter.setSizes([450, 550])

    def child_title(self, item) -> str:
        """Supply the usual guidata title for nested editors."""
        title = QW.QApplication.applicationName() or self.windowTitle()
        return f"{title} - {item.label()}"

    def accept(self) -> None:
        """Publish only validated parameter values, never the preview result."""
        if not self.edit_layout.check_all_values():
            self.preview.status.setText(_("Invalid parameters"))
            return
        self.edit_layout.accept_changes()
        update_dataset(self._original, self.instance)
        super().accept()
