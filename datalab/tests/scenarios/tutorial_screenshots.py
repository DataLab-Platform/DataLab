# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Reproduce the tutorial scenes maintained by ``doc/update_screenshots.py``."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import count
from pathlib import Path
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import sigima.enums
import sigima.objects as sio
import sigima.params as sip
from guidata.dataset.qtwidgets import DataSetEditDialog
from guidata.widgets.collectionseditor import CollectionsEditor
from qtpy import QtCore as QC
from qtpy import QtGui as QG
from qtpy import QtWidgets as QW
from sigima.tests.data import create_paracetamol_signal, get_test_image

from datalab.adapters_metadata import GeometryAdapter, TableAdapter
from datalab.config import Conf, _
from datalab.gui.main import DLMainWindow
from datalab.gui.plugins.config import PluginConfigDialog
from datalab.tests import datalab_test_app_context


def save_screenshot(
    widget: QW.QWidget, filename: Path, overlays: tuple[QW.QMenu, ...] = ()
) -> None:
    """Render a widget and its Qt-owned popup menus, without grabbing the desktop."""
    widget.show()
    QW.QApplication.processEvents()
    pixmap = widget.grab()
    if overlays:
        painter = QG.QPainter(pixmap)
        try:
            for menu in overlays:
                position = widget.mapFromGlobal(menu.pos())
                if not menu.isVisible() or not widget.rect().contains(
                    QC.QRect(position, menu.size())
                ):
                    raise RuntimeError(
                        "Screenshot menu is hidden or outside its window"
                    )
                painter.drawPixmap(position, menu.grab())
        finally:
            painter.end()
    filename.parent.mkdir(parents=True, exist_ok=True)
    if pixmap.isNull() or not pixmap.save(str(filename)):
        raise OSError(f"Could not save screenshot: {filename}")


def _action(menu: QW.QMenu, title: str) -> QW.QAction:
    """Find a visible action by its translated title, ignoring mnemonic markers."""

    def normalized(text: str) -> str:
        return text.replace("&", "").rstrip(".\u2026")

    for action in menu.actions():
        if action.isVisible() and normalized(action.text()) == normalized(title):
            return action
    raise LookupError(f"Action {title!r} not found in {menu.title()!r}")


def capture_menu(
    win: DLMainWindow, filename: Path, menu: QW.QMenu, *titles: str
) -> None:
    """Capture a real menu path over the main window, highlighting its last action."""
    menus = [menu]
    rect = win.menuBar().actionGeometry(menu.menuAction())
    menu.popup(win.menuBar().mapToGlobal(rect.bottomLeft()))
    try:
        for index, title in enumerate(titles):
            action = _action(menu, title)
            if not action.isEnabled():
                raise RuntimeError(f"Screenshot action is disabled: {title}")
            menu.setActiveAction(action)
            if index < len(titles) - 1:
                submenu = action.menu()
                if submenu is None:
                    raise LookupError(f"No submenu for {title!r}")
                pos = menu.actionGeometry(action).topRight() + QC.QPoint(1, 0)
                submenu.popup(menu.mapToGlobal(pos))
                menu = submenu
                menus.append(menu)
        QW.QApplication.processEvents()
        for parent, child in zip(menus, menus[1:]):
            child.move(
                parent.mapToGlobal(
                    parent.actionGeometry(parent.activeAction()).topRight()
                )
                + QC.QPoint(1, 0)
            )
        save_screenshot(win, filename, tuple(menus))
    finally:
        for popup in reversed(menus):
            popup.close()


def _prepare_window(win: DLMainWindow) -> None:
    """Use a stable layout with a sufficiently wide plot and no volatile status text."""
    win.resize(1300, 800)
    win.tabwidget.setMaximumWidth(440)
    win.statusBar().hide()
    win.move(0, 0)


@contextmanager
def screenshot_app_context() -> Iterator[DLMainWindow]:
    """Create a real test window with stable object references."""
    identifiers = count()
    with (
        Conf.show_result_dialog.context(False),
        Conf.process_isolation_enabled.context(False),
        patch(
            "datalab.objectmodel.uuid4",
            side_effect=lambda: uuid5(NAMESPACE_URL, f"screenshot-{next(identifiers)}"),
        ),
        datalab_test_app_context(console=False, exec_loop=False) as win,
    ):
        _prepare_window(win)
        yield win


def capture_fabry_perot(directory: Path) -> None:
    """Capture the annotation editor, metadata and current intensity-profile menu."""
    with screenshot_app_context() as win:
        panel = win.imagepanel
        image = get_test_image("fabry-perot1.jpg")
        image.title = "fabry-perot1.jpg"
        image.set_metadata_option("colormap", "gray")
        image.roi = sio.create_image_roi("circle", [600.0, 559.0, 459.0])
        image.set_graphical_annotations(
            [
                sio.SegmentAnnotation(x0=600, y0=559, x1=922, y1=390, title="Radius"),
                sio.TextAnnotation(x=720, y=410, text="Interesting note", title="Note"),
            ]
        )
        panel.add_object(image)
        panel.processor.run_feature(
            "contour_shape",
            sip.ContourShapeParam.create(shape=sigima.enums.ContourShape.CIRCLE),
        )
        results = list(GeometryAdapter.iterate_from_obj(image))
        assert results and len(results[0].result.coords) == 2
        dialog = panel.open_separate_view(edit_annotations=True)
        assert dialog is not None
        try:
            dialog.resize(1000, 750)
            save_screenshot(dialog, directory / "13.png")
        finally:
            dialog.reject()

        assert image.get_annotations()
        assert "_ann_" not in image.metadata
        assert "annotations" not in image.metadata
        editor = CollectionsEditor(win)
        editor.setup(image.metadata, readonly=True, width=900)
        try:
            editor.resize(900, 600)
            save_screenshot(editor, directory / "16.png")
        finally:
            editor.close()

        second = get_test_image("fabry-perot2.jpg")
        second.title = "fabry-perot2.jpg"
        second.set_metadata_option("colormap", "gray")
        second.roi = image.roi.copy()
        panel.add_object(second)
        panel.processor.run_feature(
            "contour_shape",
            sip.ContourShapeParam.create(shape=sigima.enums.ContourShape.CIRCLE),
        )
        capture_menu(
            win,
            directory / "24.png",
            win.analysis_menu,
            _("Intensity profiles"),
            _("Line profile"),
        )


def capture_spectrum(directory: Path) -> None:
    """Follow the spectrum tutorial through filtering, fitting and XY peak markers."""
    with screenshot_app_context() as win:
        panel = win.signalpanel
        panel.add_object(create_paracetamol_signal())
        panel.processor.run_feature("wiener")
        filtered = panel.objview.get_sel_objects()[0]
        capture_menu(win, directory / "06.png", win.roi_menu, _("Extract"))

        roi = sio.create_signal_roi([37.2, 40.1])
        panel.processor.compute_roi_extraction(roi)
        capture_menu(
            win,
            directory / "09.png",
            win.processing_menu,
            _("Fitting"),
            panel.processor.get_feature("gaussian_fit").action_title,
        )
        panel.processor.run_feature("gaussian_fit")

        panel.objview.select_objects([filtered])
        filtered.roi = sio.create_signal_roi(
            [
                [5.1, 12.8],
                [21.9, 23.0],
                [24.9, 26.1],
                [35.3, 37.6],
                [39.7, 41.8],
                [51.6, 53.0],
            ]
        )
        panel.processor.run_feature("linear_fit")
        baseline = panel.objview.get_sel_objects()[0]
        filtered.roi = None
        panel.objview.select_objects([filtered])
        panel.processor.run_feature("difference", obj2=baseline)
        spectrum = panel.objview.get_sel_objects()[0]
        count = len(panel.objmodel)
        panel.processor.compute_peak_detection(
            sip.PeakDetectionParam.create(threshold=8, min_dist=1)
        )
        tables = list(TableAdapter.iterate_from_obj(spectrum))
        assert len(tables) == 1
        assert tables[0].result.kind == sio.TableKind.XY_MARKERS
        assert len(tables[0].result.data) > 0
        assert len(panel.objmodel) == count, "Peak detection must not create a signal"
        save_screenshot(win, directory / "20.png")
        capture_menu(win, directory / "21.png", win.file_menu, _("Save to HDF5 file"))


def capture_blobs(directory: Path) -> None:
    """Use the example plugin's image generator and its actual detection parameters."""
    example = (
        Path(__file__).resolve().parents[3]
        / "plugins"
        / "examples"
        / "datalab_example_imageproc.py"
    )
    module_name = "datalab_screenshot_imageproc"
    spec = importlib.util.spec_from_file_location(module_name, example)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        with screenshot_app_context() as win:
            spec.loader.exec_module(module)
            win.set_current_panel(win.imagepanel)
            plugin = module.ExtractBlobs()
            plugin.__class__.__plugin_filepath__ = str(example)
            plugin.register(win)
            try:
                # The Advanced tab shows illustrative paths, never the temporary home.
                with patch.object(
                    PluginConfigDialog,
                    "_collect_fixed_search_paths",
                    return_value=[
                        (r"C:\Users\user\.DataLab_v1\plugins", False),
                        (r"C:\DataLab\datalab\plugins\builtin", False),
                    ],
                ):
                    dialog = PluginConfigDialog(win)
                assert dialog.tabs is not None
                assert dialog.load_info_label is not None
                dialog.load_info_label.hide()
                dialog.resize(1000, 700)
                try:
                    dialog.tabs.setCurrentIndex(3)
                    save_screenshot(dialog, directory / "01.png")
                    dialog.tabs.setCurrentIndex(0)
                    save_screenshot(dialog, directory / "02.png")
                finally:
                    dialog.reject()

                param = sio.NewImageParam.create(
                    title="Test image", height=2048, width=2048
                )
                state = np.random.get_state()
                try:
                    np.random.seed(520)
                    with patch.object(
                        plugin, "edit_new_image_parameters", return_value=param
                    ):
                        plugin.generate_test_image()
                finally:
                    np.random.set_state(state)
                panel = win.imagepanel
                capture_menu(
                    win,
                    directory / "05.png",
                    win.processing_menu,
                    panel.processor.get_feature("binning").action_title,
                )
                plugin.preprocess()

                # Intercept only the dialog input; the plugin supplies the parameters.
                with patch.object(panel.processor, "run_feature") as run_feature:
                    plugin.detect_blobs()
                detection = run_feature.call_args.args[1]
                assert not detection.filter_by_color
                dialog = DataSetEditDialog(detection, parent=win)
                try:
                    save_screenshot(dialog, directory / "11.png")
                finally:
                    dialog.reject()
                panel.processor.run_feature("blob_opencv", detection)
                image = panel.objview.get_sel_objects()[0]
                assert list(GeometryAdapter.iterate_from_obj(image))
            finally:
                plugin.unregister()
    finally:
        sys.modules.pop(module_name)
