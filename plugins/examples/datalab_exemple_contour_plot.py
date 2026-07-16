# -*- coding: utf-8 -*-

"""
Contour plot plugin for DataLab.

This plugin adds a simple user-facing contour-plot view for the currently
selected image. It opens a separate PlotPy dialog and overlays automatically
generated isocontours on top of the image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np
from datalab.adapters_plotpy.objects.image import get_obj_coords
from plotpy.config import CONF
from plotpy.items import AnnotatedPolygon
from plotpy.items.contour import compute_contours
from plotpy.plot import PlotDialog
from plotpy.styles import AnnotationParam

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.config import _
from datalab.objectmodel import get_uuid
from datalab.plugins import PluginBase, PluginInfo

if TYPE_CHECKING:
    from sigima.objects import ImageObj


class _ContourAnnotatedPolygon(AnnotatedPolygon):
    """AnnotatedPolygon with label placed on the contour perimeter."""

    LABEL_ANCHOR = "C"

    def __init__(self, *args, label_fraction: float = 0.0, **kwargs) -> None:
        self._label_fraction = label_fraction
        super().__init__(*args, **kwargs)

    def set_label_position(self) -> None:
        """Place the label on the contour perimeter at *_label_fraction*."""
        pts = self.shape.get_points()
        if pts is None or len(pts) < 2:
            super().set_label_position()
            return
        # Pick the vertex at the given fraction of the perimeter
        idx = int(self._label_fraction * len(pts)) % len(pts)
        x, y = float(pts[idx, 0]), float(pts[idx, 1])
        self.label.set_pos(x, y)


def _nice_step(data_range: float, target_count: int = 10) -> float:
    """Compute a human-friendly step size for *data_range* / *target_count*.

    Returns a value from the set {1, 2, 5} × 10^n that yields
    approximately *target_count* intervals.
    """
    if data_range <= 0 or target_count < 1:
        return 1.0
    raw = data_range / target_count
    magnitude = 10 ** np.floor(np.log10(raw))
    residual = raw / magnitude
    if residual <= 1.0:
        nice = 1.0
    elif residual <= 2.0:
        nice = 2.0
    elif residual <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * magnitude)


class ContourPlotParam(gds.DataSet):
    """Parameters for contour-plot visualization."""

    minimum = gds.FloatItem(_("Minimum value"), default=0.0)
    maximum = gds.FloatItem(_("Maximum value"), default=1.0)
    step = gds.FloatItem(_("Step between levels"), default=1.0, min=1e-12)
    show_image = gds.BoolItem(_("Overlay image"), default=True)
    show_labels = gds.BoolItem(_("Show level labels"), default=True)


class ContourPlotPlugin(PluginBase):
    """DataLab plugin exposing PlotPy contour plots for images."""

    PLUGIN_INFO = PluginInfo(
        name=_("Contour isoline plot"),
        version="1.0.0",
        description=_(
            "Display isolines (contour lines) overlaid on the selected image, "
            "with configurable level range, step, and optional value labels"
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._dialogs: list[PlotDialog] = []

    def _get_selected_image(self) -> ImageObj | None:
        """Return the single selected image object."""
        objects = self.imagepanel.objview.get_sel_objects(include_groups=False)
        if len(objects) != 1:
            return None
        return objects[0]

    @staticmethod
    def _get_finite_range(obj: ImageObj) -> tuple[np.ndarray, float, float]:
        """Return cleaned image data and its finite value range."""
        data = np.asarray(obj.data.real, dtype=np.float64)
        mask = obj.maskdata
        if mask is not None:
            finite = np.asarray(np.ma.array(data, mask=mask).compressed(), dtype=float)
        else:
            finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ValueError("Image has no finite data")
        min_value = float(finite.min())
        max_value = float(finite.max())
        cleaned = np.nan_to_num(
            data,
            nan=min_value,
            posinf=max_value,
            neginf=min_value,
        )
        if mask is not None:
            cleaned = np.where(mask, min_value, cleaned)
        return cleaned, min_value, max_value

    @staticmethod
    def _build_levels(param: ContourPlotParam) -> np.ndarray:
        """Create contour levels from plugin parameters."""
        start = param.minimum + param.step
        stop = param.maximum
        if start >= stop:
            return np.array([param.minimum + (param.maximum - param.minimum) / 2])
        levels = np.arange(start, stop, param.step, dtype=np.float64)
        # Round to avoid floating-point drift (e.g. 30.000000000000004)
        decimals = max(0, -int(np.floor(np.log10(param.step))) + 10)
        return np.round(levels, decimals)

    @staticmethod
    def _make_contour_items(
        contour_data: np.ndarray,
        levels: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        show_labels: bool,
    ) -> list[AnnotatedPolygon]:
        """Create annotated contour items with embedded level labels."""
        clines = compute_contours(contour_data, levels, grid_x, grid_y)
        items: list[AnnotatedPolygon] = []
        # Use the golden angle to spread labels around the perimeter
        golden_angle = (np.sqrt(5) - 1) / 2  # ≈ 0.618
        for idx, cline in enumerate(clines):
            level_text = f"{cline.level:g}"

            def _info_cb(ann: AnnotatedPolygon, text: str = level_text) -> str:
                return f"Z = {text}"

            ann_param = AnnotationParam(_("Annotation"), icon="annotation.png")
            ann_param.read_config(CONF, "plot", "shape/label")
            fraction = (idx * golden_angle) % 1.0
            item = _ContourAnnotatedPolygon(
                points=cline.vertices,
                closed=True,
                annotationparam=ann_param,
                info_callback=_info_cb,
                label_fraction=fraction,
            )
            item.set_style("plot", "shape/contour")
            item.setTitle(_("Contour") + f" Z={level_text}")
            item.set_label_visible(show_labels)
            items.append(item)
        return items

    def _release_dialog(self, dialog: PlotDialog) -> None:
        """Release a closed dialog reference."""
        if dialog in self._dialogs:
            self._dialogs.remove(dialog)
        dialog.deleteLater()

    def show_contour_plot(self) -> None:
        """Open a separate contour-plot view for the selected image."""
        obj = self._get_selected_image()
        if obj is None:
            self.show_warning(_("Select a single image first."))
            return

        try:
            contour_data, data_min, data_max = self._get_finite_range(obj)
        except ValueError:
            self.show_warning(_("Selected image does not contain finite data."))
            return

        if np.isclose(data_min, data_max):
            self.show_warning(_("Selected image is constant: no contour can be drawn."))
            return

        current_item = self.imagepanel.plothandler.get(get_uuid(obj))
        default_min, default_max = data_min, data_max
        if current_item is not None:
            lut_min, lut_max = current_item.get_lut_range()
            if np.isfinite(lut_min) and np.isfinite(lut_max) and lut_min < lut_max:
                default_min, default_max = lut_min, lut_max

        param = ContourPlotParam(_("Contour plot parameters"))
        param.minimum = default_min
        param.maximum = default_max
        param.step = _nice_step(default_max - default_min)
        if not param.edit(self.main):
            return

        if param.maximum <= param.minimum:
            self.show_warning(_("Maximum value must be strictly greater than minimum."))
            return

        levels = self._build_levels(param)
        xcoords, ycoords = get_obj_coords(obj)
        grid_x, grid_y = np.meshgrid(xcoords, ycoords)
        contour_items = self._make_contour_items(
            contour_data, levels, grid_x, grid_y, param.show_labels
        )
        if not contour_items:
            self.show_warning(_("No contour was found for the selected level range."))
            return

        dlg = self.imagepanel.create_new_dialog(
            title=f"{obj.title} - {self.PLUGIN_INFO.name}",
            edit=False,
            name=f"{obj.PREFIX}_contour_plot",
            options={
                "show_contrast": True,
                "show_itemlist": True,
                "lock_aspect_ratio": True,
                "curve_antialiasing": False,
            },
        )
        if dlg is None:
            return

        plot = dlg.get_plot()
        if param.show_image:
            adapter = create_adapter_from_object(obj)
            image_item = adapter.make_item(update_from=current_item)
            plot.add_item(image_item, z=0)
            plot.set_active_item(image_item)
            image_item.unselect()

        for item in contour_items:
            plot.add_item(item)

        plot.replot()
        self._dialogs.append(dlg)
        dlg.finished.connect(lambda _result, dialog=dlg: self._release_dialog(dialog))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def create_actions(self) -> None:
        """Create plugin menu actions."""
        acth = self.imagepanel.acthandler
        with acth.new_menu(self.PLUGIN_INFO.name):
            acth.new_action(
                _("Show contour plot..."),
                triggered=self.show_contour_plot,
                select_condition="exactly_one",
            )
