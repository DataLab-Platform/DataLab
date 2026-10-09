# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Window showing a plugin instrument: live view and settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from guidata.configtools import get_icon
from guidata.dataset.qtwidgets import DataSetEditLayout
from guidata.qthelpers import win32_fix_title_bar_background
from plotpy.plot import PlotOptions, PlotWidget
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj, SignalObj
from sigimax.adapters_plotpy.objects.signal import CURVESTYLES

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import _
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from datalab.gui.main import DLMainWindow
    from datalab.plugins.instruments import (
        InstrumentAcquisition,
        InstrumentFrame,
        PluginInstrument,
    )

__all__ = ["InstrumentWindow"]


class InstrumentWindow(QW.QDialog):
    """Live view of a plugin instrument, next to its settings.

    The view is refreshed when settings change, and periodically in live mode.
    Each acquisition is added to the workspace in a new group.

    Args:
        main: DataLab main window
        instrument: instrument opened by a plugin tool
        title: window title
    """

    #: Delay between a settings change and the refresh of the view (ms)
    REFRESH_DELAY = 150

    def __init__(
        self, main: DLMainWindow, instrument: PluginInstrument, title: str
    ) -> None:
        super().__init__(main)
        win32_fix_title_bar_background(self)
        self.main = main
        self.instrument = instrument
        self.setWindowTitle(title.rstrip(".…"))
        self.setModal(False)
        self.plotwidget: PlotWidget | None = None
        self.items: list = []
        self._plot_kind: str | None = None
        self._plot_signature: tuple | None = None
        self._value_range: tuple[float, float] | None = None

        splitter = QW.QSplitter(QC.Qt.Horizontal)
        self._stage = QW.QWidget()
        self._stage.setMinimumSize(360, 280)
        self._stage_layout = QW.QVBoxLayout(self._stage)
        self._stage_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_label = QW.QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(QC.Qt.PlainText)
        self._stage_layout.addWidget(self.summary_label)
        splitter.addWidget(self._stage)

        settings_widget = QW.QWidget()
        settings_layout = QW.QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QW.QScrollArea()
        scroll.setWidgetResizable(True)
        form = QW.QWidget()
        form_layout = QW.QVBoxLayout(form)
        grid = QW.QGridLayout()
        grid.setAlignment(QC.Qt.AlignTop)
        form_layout.addLayout(grid)
        form_layout.addStretch()
        scroll.setWidget(form)
        settings_layout.addWidget(scroll, 1)
        self.status_label = QW.QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(QC.Qt.PlainText)
        settings_layout.addWidget(self.status_label)
        commands = QW.QHBoxLayout()
        self.live_button = QW.QPushButton(get_icon("replay.svg"), _("Live"))
        self.live_button.setCheckable(True)
        self.live_button.setToolTip(_("Refresh the view continuously"))
        self.live_button.toggled.connect(self.set_live)
        commands.addWidget(self.live_button)
        commands.addStretch()
        self.acquire_button = QW.QPushButton(get_icon("record.svg"), _("Acquire"))
        self.acquire_button.setToolTip(
            _("Add an acquisition to the workspace, in a new group")
        )
        self.acquire_button.clicked.connect(self.acquire)
        commands.addWidget(self.acquire_button)
        settings_layout.addLayout(commands)
        splitter.addWidget(settings_widget)

        self._refresh_timer = QC.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._live_timer = QC.QTimer(self)
        self._live_timer.setInterval(max(50, int(instrument.live_interval_ms)))
        self._live_timer.timeout.connect(self.refresh)
        self.finished.connect(self.stop)

        self.edit_layout = DataSetEditLayout(
            self,
            instrument.settings,
            grid,
            change_callback=self.settings_changed,
            auto_sliders=True,
        )

        buttons = QW.QDialogButtonBox(QW.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout = QW.QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(buttons)

        self.resize(1100, 650)
        screen = self.screen().availableGeometry()
        self.resize(
            min(self.width(), screen.width()), min(self.height(), screen.height())
        )
        splitter.setSizes([650, 450])
        self._refresh_timer.start(0)

    def child_title(self, item) -> str:
        """Supply the usual guidata title for nested editors."""
        return f"{self.windowTitle()} - {item.label()}"

    def settings_changed(self) -> None:
        """Refresh the view shortly after the settings changed."""
        self._refresh_timer.start(self.REFRESH_DELAY)

    def set_live(self, live: bool) -> None:
        """Start or stop refreshing the view continuously."""
        if live:
            self._live_timer.start()
        else:
            self._live_timer.stop()

    def stop(self) -> None:
        """Stop refreshing the view."""
        self._refresh_timer.stop()
        self._live_timer.stop()

    def _apply_settings(self) -> bool:
        """Write the edited values into the instrument settings."""
        if not self.edit_layout.check_all_values():
            self.status_label.setText(_("Invalid settings"))
            return False
        self.edit_layout.accept_changes()
        return True

    def refresh(self) -> None:
        """Show a live frame for the current settings."""
        if not self._apply_settings():
            return
        try:
            frame = self.instrument.preview()
            self.render(frame)
        except Exception as exc:  # pylint: disable=broad-except
            # Instruments are third-party code: keep the window usable
            self.status_label.setText(str(exc))
            return
        self.status_label.clear()

    def render(self, frame: InstrumentFrame) -> None:
        """Draw a frame in the view."""
        first = frame.objects[0]
        kind = "image" if isinstance(first, ImageObj) else "curve"
        if self._plot_kind != kind:
            if self.plotwidget is not None:
                self._stage_layout.removeWidget(self.plotwidget)
                self.plotwidget.deleteLater()
            self.plotwidget = PlotWidget(self._stage, options=PlotOptions(type=kind))
            self._stage_layout.insertWidget(0, self.plotwidget, 1)
            self._plot_kind = kind
            self._plot_signature = None
            self.items = []
        plot = self.plotwidget.plot
        generator = CURVESTYLES.curve_style
        try:
            CURVESTYLES.curve_style = CURVESTYLES.style_generator()
            items = [
                create_adapter_from_object(obj).make_item() for obj in frame.objects
            ]
        finally:
            CURVESTYLES.curve_style = generator
        if self.items:
            plot.del_items(self.items)
        self.items = items
        for item in items:
            plot.add_item(item)
            item.unselect()
        plot.set_titles(
            title=first.title if kind == "image" else "",
            xlabel=first.xlabel,
            xunit=first.xunit,
            ylabel=(first.ylabel, getattr(first, "zlabel", "")),
            yunit=(first.yunit, getattr(first, "zunit", "")),
        )
        if kind == "image":
            signature = (kind, first.data.shape, first.x0, first.y0, first.dx, first.dy)
            if frame.value_range is not None:
                items[0].set_lut_range(frame.value_range)
                plot.update_colormap_axis(items[0])
        else:
            signature = (
                kind,
                tuple(
                    (obj.x.size, obj.x[0], obj.x[-1]) if obj.x.size else (0,)
                    for obj in frame.objects
                ),
            )
        if signature != self._plot_signature:
            plot.do_autoscale(replot=False)
            if kind == "curve" and frame.value_range is not None:
                plot.set_axis_limits("left", *frame.value_range)
        elif kind == "curve" and frame.value_range != self._value_range:
            # Like a new V/div on a scope: Y follows, the X zoom is kept
            if frame.value_range is None:
                plot.do_autoscale(replot=False, axis_id=plot.get_axis_id("left"))
            else:
                plot.set_axis_limits("left", *frame.value_range)
        self._plot_signature = signature
        self._value_range = frame.value_range
        self.summary_label.setText(frame.summary)
        plot.replot()

    def acquire(self) -> None:
        """Add an acquisition to the workspace, in a new group."""
        if not self._apply_settings():
            return
        live = self._live_timer.isActive()
        self._live_timer.stop()
        try:
            acquisition = self.instrument.acquire()
            self.add_acquisition(acquisition)
        except Exception as exc:  # pylint: disable=broad-except
            # Instruments are third-party code: keep the window usable
            self.status_label.setText(str(exc))
            return
        finally:
            if live:
                self._live_timer.start()
        self.status_label.setText(
            _("%d objects added to group '%s'")
            % (len(acquisition.objects), acquisition.group_title)
        )

    def add_acquisition(self, acquisition: InstrumentAcquisition) -> None:
        """Add acquired objects to their panel, in a new group."""
        objects = acquisition.objects
        panel_name = "signal" if isinstance(objects[0], SignalObj) else "image"
        panel = (
            self.main.signalpanel if panel_name == "signal" else self.main.imagepanel
        )
        with self.main.context_no_refresh():
            group = panel.add_group(acquisition.group_title)
            try:
                # pylint: disable=protected-access
                panel._add_objects(objects, get_uuid(group))
            except Exception:
                panel.objview.remove_item(get_uuid(group), refresh=False)
                panel.objmodel.remove_group(group)
                raise
        self.main.set_current_panel(panel_name)
        panel.objview.select_objects(objects)
