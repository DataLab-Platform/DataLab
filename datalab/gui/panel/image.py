# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Image panel (see parent package :mod:`datalab.gui.panel`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from typing import TYPE_CHECKING, Type
from weakref import ReferenceType, ref

import guidata.dataset as gds
import guidata.dataset.qtwidgets as gdq
import numpy as np
from guidata.dataset import update_dataset
from guidata.qthelpers import exec_dialog
from plotpy.builder import make
from plotpy.interfaces import IVoiImageItemType
from plotpy.plot import PlotDialog, PlotOptions
from plotpy.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    LabelTool,
)
from qtpy import QtWidgets as QW
from sigima.io import (
    ImageExportOptionSpec,
    ImageExportParam,
    get_image_export_capabilities,
    get_supported_export_dtypes,
    prepare_image_export_preview,
    write_image,
)
from sigima.io.image import ImageIORegistry
from sigima.objects import ImageDatatypes, ImageObj, ImageROI, NewImageParam

from datalab.config import Conf, _
from datalab.gui import roieditor
from datalab.gui.actionhandler import ImageActionHandler
from datalab.gui.newobject import create_image_gui
from datalab.gui.panel.base import BaseDataPanel
from datalab.gui.plothandler import ImagePlotHandler
from datalab.gui.processor.image import ImageProcessor
from datalab.objectmodel import get_uuid

if TYPE_CHECKING:
    from plotpy.plot import BasePlot

    from datalab.gui.docks import DockablePlotWidget


IMAGE_EXPORT_OPTION_LABELS = {
    "compress_level": _("Compression level"),
    "compression": _("Compression"),
    "compression_level": _("Compression level"),
    "do_compression": _("MAT compression"),
    "quality": _("Quality"),
    "subsampling": _("Subsampling"),
    "progressive": _("Progressive"),
    "optimize": _("Optimize"),
    "smooth": _("Smoothing"),
    "quality_mode": _("Quality mode"),
    "quality_layers": _("Quality layers"),
    "irreversible": _("Irreversible"),
    "progression": _("Progression order"),
    "num_resolutions": _("Number of resolutions"),
    "tile_size": _("Tile size"),
    "plt": _("PLT markers"),
    "predictor": _("Predictor"),
    "rows_per_strip": _("Rows per strip"),
    "resolution": _("Resolution"),
    "resolution_unit": _("Resolution unit"),
    "photometric": _("Photometric interpretation"),
    "delimiter": _("Delimiter"),
    "precision": _("Precision"),
}

IMAGE_EXPORT_CHOICE_LABELS = {
    "none": _("None"),
    "rates": _("Rates"),
    "lzw": "LZW",
    "deflate": _("Deflate"),
    "zstd": _("Zstandard"),
    "jpeg": "JPEG",
    "whitespace": _("Whitespace"),
    "tab": _("Tab"),
    "comma": _("Comma"),
    "semicolon": _("Semicolon"),
    "inch": _("Inch"),
    "centimeter": _("Centimeter"),
    "horizontal": _("Horizontal"),
    "floatingpoint": _("Floating-point"),
    "minisblack": _("Black is zero"),
    "miniswhite": _("White is zero"),
}


def get_image_export_dtype_choices(
    param: ImageExportGUIParam, _item: gds.ChoiceItem, _value: str
) -> list[tuple[str, str, None]]:
    """Return target dtype choices supported by the selected image format."""
    return [
        (dtype, _("Automatic") if dtype == "auto" else dtype, None)
        for dtype in param.supported_dtypes
    ]


class ImageExportGUIParam(ImageExportParam, title=_("Image export")):
    """Image export parameters restricted to the selected file format."""

    supported_dtypes: tuple[str, ...] = ("auto",)
    format_extension = ""
    format_option_specs: tuple[ImageExportOptionSpec, ...] = ()
    target_dtype = gds.ChoiceItem(
        _("Target data type"), get_image_export_dtype_choices, default="auto"
    )
    gamma = gds.FloatItem(
        _("Gamma"), default=None, min=0.01, check=False, allow_none=True
    ).set_prop(
        "display",
        active=gds.FuncProp(
            ImageExportParam.normalization_prop, lambda value: value != "none"
        ),
    )
    invert = gds.BoolItem(_("Invert"), default=False).set_prop(
        "display",
        active=gds.FuncProp(
            ImageExportParam.normalization_prop, lambda value: value != "none"
        ),
    )

    def get_format_options(self) -> dict[str, object]:
        """Return writer options exposed by this format parameter class."""
        options = {}
        for spec in self.format_option_specs:
            value = getattr(self, spec.key)
            if spec.value_kind in ("int_pair", "float_pair", "float_list"):
                value = serialize_image_export_sequence(value, spec)
            options[spec.key] = value
        return options


def get_image_export_choice_label(value: object) -> str:
    """Return a translated human-readable label for an option choice."""
    return IMAGE_EXPORT_CHOICE_LABELS.get(value, str(value))


def serialize_image_export_sequence(
    value: object, spec: ImageExportOptionSpec
) -> tuple[int, int] | tuple[float, float] | list[float] | None:
    """Serialize a pair or list option from its editable GUI representation."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if spec.allow_none:
            return None
        label = IMAGE_EXPORT_OPTION_LABELS[spec.key]
        raise ValueError(_("%s may not be empty") % label)
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    else:
        values = list(value)
    if spec.value_kind in ("int_pair", "float_pair") and len(values) != 2:
        label = IMAGE_EXPORT_OPTION_LABELS[spec.key]
        raise ValueError(_("%s must contain 2 values") % label)
    try:
        if spec.value_kind == "int_pair":
            return tuple(int(item) for item in values)
        converted = [float(item) for item in values]
    except (TypeError, ValueError) as exc:
        label = IMAGE_EXPORT_OPTION_LABELS[spec.key]
        raise ValueError(_("%s contains invalid numbers") % label) from exc
    if spec.value_kind == "float_pair":
        return tuple(converted)
    return converted


def create_image_export_option_item(spec: ImageExportOptionSpec) -> gds.DataItem:
    """Create a guidata item for an image export capability option."""
    label = IMAGE_EXPORT_OPTION_LABELS[spec.key]
    if spec.value_kind == "bool":
        return gds.BoolItem(label, default=spec.default, allow_none=spec.allow_none)
    if spec.value_kind == "int":
        return gds.IntItem(
            label,
            default=spec.default,
            min=spec.minimum,
            max=spec.maximum,
            nonzero=spec.minimum == 0 and not spec.minimum_inclusive,
            allow_none=spec.allow_none,
        )
    if spec.value_kind == "float":
        return gds.FloatItem(
            label,
            default=spec.default,
            min=spec.minimum,
            max=spec.maximum,
            nonzero=spec.minimum == 0 and not spec.minimum_inclusive,
            allow_none=spec.allow_none,
        )
    if spec.value_kind == "choice":
        choices = [
            (choice, get_image_export_choice_label(choice)) for choice in spec.choices
        ]
        return gds.ChoiceItem(
            label, choices, default=spec.default, allow_none=spec.allow_none
        )
    if spec.value_kind in ("int_pair", "float_pair", "float_list"):
        default = (
            ""
            if spec.default is None
            else ", ".join(str(value) for value in spec.default)
        )
        return gds.StringItem(label, default=default)
    if spec.value_kind == "string":
        return gds.StringItem(label, default=spec.default, allow_none=spec.allow_none)
    raise AssertionError(f"Unknown image export option kind: {spec.value_kind!r}")


IMAGE_EXPORT_PARAM_CLASSES: dict[str, type[ImageExportGUIParam]] = {}


def get_image_export_param_class(filename: str) -> type[ImageExportGUIParam]:
    """Return the cached GUI parameter class for an export format."""
    capabilities = get_image_export_capabilities(filename)
    canonical_extension = capabilities.canonical_extension
    paramclass = IMAGE_EXPORT_PARAM_CLASSES.get(canonical_extension)
    if paramclass is None:
        class_name = "".join(
            character
            for character in capabilities.format_name.title()
            if character.isalnum()
        )
        class_attributes = {
            "__doc__": f"{capabilities.format_name} export parameters.",
            "format_option_specs": capabilities.option_specs,
        }
        class_attributes.update(
            {
                spec.key: create_image_export_option_item(spec)
                for spec in capabilities.option_specs
            }
        )
        paramclass = type(
            f"{class_name}ImageExportGUIParam",
            (ImageExportGUIParam,),
            class_attributes,
        )
        IMAGE_EXPORT_PARAM_CLASSES[canonical_extension] = paramclass
    return paramclass


def create_image_export_gui_param(obj: ImageObj, filename: str) -> ImageExportGUIParam:
    """Create format-aware image export parameters with practical defaults.

    Args:
        obj: Source image
        filename: Destination file name

    Returns:
        Image export GUI parameters
    """
    capabilities = get_image_export_capabilities(filename)
    paramclass = get_image_export_param_class(filename)
    param = paramclass()
    param.format_extension = capabilities.canonical_extension
    param.supported_dtypes = get_supported_export_dtypes(filename)
    source_dtype = np.dtype(obj.data.dtype)
    concrete_dtypes = param.supported_dtypes[1:]
    source_is_supported = source_dtype.name in concrete_dtypes
    forces_integer_conversion = concrete_dtypes and all(
        np.issubdtype(np.dtype(dtype), np.integer) for dtype in concrete_dtypes
    )
    if (
        not capabilities.raw_preserving
        and not source_is_supported
        and forces_integer_conversion
    ):
        param.normalization = "minmax"
        param.behavior = "rescale"
        param.target_dtype = "auto"
    return param


class ImageExportDialog(PlotDialog):
    """Format-aware image export dialog with an in-memory graphical preview."""

    def __init__(
        self,
        parent: QW.QWidget | None,
        obj: ImageObj,
        filename: str,
        param: ImageExportGUIParam,
    ) -> None:
        self.obj = obj
        self.filename = filename
        self.preview_item = None
        self.param_widget = gdq.DataSetEditGroupBox(
            _("Export options"),
            param.__class__,
            button_text=_("Preview"),
            button_icon="reload.svg",
        )
        update_dataset(self.param_widget.dataset, param)
        self.param_widget.dataset.supported_dtypes = param.supported_dtypes
        self.param_widget.dataset.format_extension = param.format_extension
        super().__init__(
            parent=parent,
            title=_("Export image"),
            icon="export.svg",
            toolbar=False,
            edit=True,
            options=PlotOptions(type="image", show_contrast=True),
            size=(900, 600),
        )
        self.param_widget.SIG_APPLY_BUTTON_CLICKED.connect(self.update_preview)
        self.param_widget.set_apply_button_state(True)
        self.update_preview()

    def setup_layout(self) -> None:
        """Add export controls beside the graphical preview."""
        super().setup_layout()
        self.plot_layout.addWidget(self.param_widget, 0, 1)
        self.plot_layout.setColumnStretch(0, 2)
        self.plot_layout.setColumnStretch(1, 1)

    def get_export_param(self) -> ImageExportParam:
        """Return backend parameters from the current GUI values."""
        guiparam = self.param_widget.dataset
        param = ImageExportParam()
        update_dataset(param, guiparam)
        param.format_options = guiparam.get_format_options()
        return param

    def update_preview(self) -> bool:
        """Render current export preparation directly in the preview plot."""
        try:
            data = prepare_image_export_preview(
                self.obj.data, self.filename, self.get_export_param()
            )
        except (TypeError, ValueError, OSError) as exc:
            QW.QMessageBox.warning(self, _("Export image"), str(exc))
            return False
        plot = self.manager.get_plot()
        if self.preview_item is None:
            self.preview_item = make.image(data, colormap="gray")
            plot.add_item(self.preview_item)
            plot.set_active_item(self.preview_item)
            plot.do_autoscale()
        else:
            self.preview_item.set_data(data)
            plot.replot()
        return True

    def accept(self) -> None:
        """Validate controls and preview preparation before accepting."""
        if not self.param_widget.edit.check_all_values():
            QW.QMessageBox.warning(
                self,
                _("Export image"),
                _("Some required entries are incorrect."),
            )
            return
        self.param_widget.edit.accept_changes()
        if self.update_preview():
            QW.QDialog.accept(self)


class ImagePanel(BaseDataPanel[ImageObj, ImageROI, roieditor.ImageROIEditor]):
    """Object handling the item list, the selected item properties and plot,
    specialized for Image objects"""

    PANEL_STR = _("Image Panel")
    PANEL_STR_ID = "image"
    PARAMCLASS = ImageObj
    MINDIALOGSIZE = (800, 800)

    # The following tools are used to create annotations on images. The annotation
    # items are created using PlotPy's default settings. Those appearance settings
    # may be modified in the configuration (see `datalab.config`).
    ANNOTATION_TOOLS = (
        AnnotatedCircleTool,
        AnnotatedSegmentTool,
        AnnotatedRectangleTool,
        AnnotatedPointTool,
        AnnotatedEllipseTool,
        LabelTool,
    )

    IO_REGISTRY = ImageIORegistry
    H5_PREFIX = "DataLab_Ima"

    # pylint: disable=duplicate-code

    @staticmethod
    def get_roi_class() -> Type[ImageROI]:
        """Return ROI class"""
        return ImageROI

    @staticmethod
    def get_roieditor_class() -> Type[roieditor.ImageROIEditor]:
        """Return ROI editor class"""
        return roieditor.ImageROIEditor

    def edit_export_parameters(
        self, obj: ImageObj, filename: str
    ) -> tuple[bool, ImageExportParam | None]:
        """Edit image export parameters for the selected file format.

        Args:
            obj: Image to export
            filename: Destination file name

        Returns:
            A tuple containing the confirmation state and export parameters
        """
        try:
            guiparam = create_image_export_gui_param(obj, filename)
        except ValueError:
            return True, None
        dialog = ImageExportDialog(self.parentWidget(), obj, filename, guiparam)
        if not exec_dialog(dialog):
            return False, None
        return True, dialog.get_export_param()

    def write_object_to_file(
        self,
        obj: ImageObj,
        filename: str,
        param: ImageExportParam | None = None,
    ) -> None:
        """Write an image with optional format-aware export parameters.

        Args:
            obj: Image to export
            filename: Destination file name
            param: Optional image export parameters
        """
        write_image(filename, obj, param)

    def __init__(
        self,
        parent: QW.QWidget,
        dockableplotwidget: DockablePlotWidget,
        panel_toolbar: QW.QToolBar,
    ) -> None:
        super().__init__(parent)
        self._contrast_sync_in_progress = False
        self._contrast_editors: dict[
            str, list[ReferenceType[roieditor.ImageROIEditor]]
        ] = {}
        self.plothandler = ImagePlotHandler(self, dockableplotwidget.plotwidget)
        self.processor = ImageProcessor(self, dockableplotwidget.plotwidget)
        view_toolbar = dockableplotwidget.toolbar
        self.acthandler = ImageActionHandler(self, panel_toolbar, view_toolbar)

    def register_contrast_editor(
        self, obj: ImageObj, editor: roieditor.ImageROIEditor
    ) -> None:
        """Register an image ROI editor for contrast synchronization."""
        obj_uuid = get_uuid(obj)
        editors = self._contrast_editors.setdefault(obj_uuid, [])
        for editor_ref in list(editors):
            current_editor = editor_ref()
            if current_editor is None:
                editors.remove(editor_ref)
            elif current_editor is editor:
                return
        editors.append(ref(editor))
        item = self.plothandler.get(obj_uuid)
        if item is not None:
            zmin, zmax = item.get_lut_range()
            editor.apply_shared_contrast(zmin, zmax)

    def _update_contrast_panel_range(self, zmin: float, zmax: float) -> None:
        """Update contrast panel range without re-emitting LUT signals."""
        contrast = self.plothandler.plotwidget.manager.get_contrast_panel()
        if contrast is None:
            return
        contrast.histogram.range.set_range(zmin, zmax, dosignal=False)
        contrast.histogram.replot()

    def _update_object_contrast_state(
        self, obj: ImageObj, zmin: float, zmax: float, update_panel: bool = False
    ) -> None:
        """Update object and current panel state for a contrast change."""
        obj.zscalemin, obj.zscalemax = zmin, zmax
        if obj is self.objview.get_current_object():
            self.objprop.update_properties_from(obj)
            if update_panel:
                self._update_contrast_panel_range(zmin, zmax)

    def apply_shared_contrast(
        self,
        obj: ImageObj,
        zmin: float,
        zmax: float,
        source: roieditor.ImageROIEditor | None = None,
    ) -> None:
        """Apply a contrast change coming from another view."""
        del source  # unused: kept for API symmetry with _notify_contrast_editors
        self._update_object_contrast_state(obj, zmin, zmax, update_panel=True)
        item = self.plothandler.get(get_uuid(obj))
        if item is None:
            return
        if item.get_lut_range() == (zmin, zmax):
            return
        self._contrast_sync_in_progress = True
        try:
            item.set_lut_range((zmin, zmax))
            plot = self.plothandler.plot
            plot.update_colormap_axis(item)
            plot.notify_colormap_changed()
        finally:
            self._contrast_sync_in_progress = False

    def _notify_contrast_editors(
        self,
        obj: ImageObj,
        zmin: float,
        zmax: float,
        source: roieditor.ImageROIEditor | None = None,
    ) -> None:
        """Propagate a contrast change to all ROI editors of an image."""
        obj_uuid = get_uuid(obj)
        editors = self._contrast_editors.get(obj_uuid)
        if not editors:
            return
        alive_editors: list[ReferenceType[roieditor.ImageROIEditor]] = []
        for editor_ref in editors:
            editor = editor_ref()
            if editor is None:
                continue
            alive_editors.append(editor_ref)
            if editor is not source:
                editor.apply_shared_contrast(zmin, zmax)
        if alive_editors:
            self._contrast_editors[obj_uuid] = alive_editors
        else:
            self._contrast_editors.pop(obj_uuid, None)

    def _get_lut_changed_objects(
        self, plot: BasePlot
    ) -> list[tuple[ImageObj, float, float]]:
        """Return image objects affected by a LUT change on the plot."""
        changed_objects: list[tuple[ImageObj, float, float]] = []
        items = plot.get_selected_items(item_type=IVoiImageItemType)
        if not items:
            active_item = plot.get_last_active_item(IVoiImageItemType)
            items = [] if active_item is None else [active_item]
        for item in items:
            obj = self.plothandler.get_obj_from_item(item)
            if not isinstance(obj, ImageObj):
                continue
            zmin, zmax = item.get_lut_range()
            changed_objects.append((obj, zmin, zmax))
        return changed_objects

    # ------Refreshing GUI--------------------------------------------------------------
    def plot_lut_changed(self, plot: BasePlot) -> None:
        """The LUT of the plot has changed: updating image objects accordingly

        Args:
            plot: Plot object
        """
        for obj, zmin, zmax in self._get_lut_changed_objects(plot):
            self._update_object_contrast_state(obj, zmin, zmax, update_panel=True)
            if not self._contrast_sync_in_progress:
                self._notify_contrast_editors(obj, zmin, zmax)

    # ------Creating, adding, removing objects------------------------------------------
    def get_newparam_from_current(
        self, newparam: NewImageParam | None = None, title: str | None = None
    ) -> NewImageParam | None:
        """Get new object parameters from the current object.

        Args:
            newparam (guidata.dataset.DataSet): new object parameters.
             If None, create a new one.
            title: new object title. If None, use the current object title, or the
             default title.

        Returns:
            New object parameters
        """
        curobj: ImageObj = self.objview.get_current_object()
        if newparam is None:
            newparam = NewImageParam()
        if title is not None:
            newparam.title = title
        if curobj is not None and Conf.use_image_dims.get(True):
            # Use current image dimensions for new image:
            newparam.height, newparam.width = curobj.data.shape
            newparam.dtype = ImageDatatypes.from_numpy_dtype(curobj.data.dtype)
        return newparam

    def new_object(
        self,
        param: NewImageParam | None = None,
        edit: bool = False,
        add_to_panel: bool = True,
    ) -> ImageObj | None:
        """Create a new object (image).

        Args:
            param (guidata.dataset.DataSet): new object parameters
            edit (bool): Open a dialog box to edit parameters (default: False).
             When False, the object is created with default parameters and creation
             parameters are stored in metadata for interactive editing.
            add_to_panel (bool): Add the object to the panel (default: True)

        Returns:
            New object
        """
        if not self.mainwindow.confirm_memory_state():
            return None
        param = self.get_newparam_from_current(param)
        image = create_image_gui(param, edit=edit, parent=self.parentWidget())
        if image is None:
            return None
        action = self.mainwindow.historypanel.add_ui_entry(
            _("New image"),
            target="imagepanel",
            method_name="new_object",
            save_state=False,
            param=param,
            add_to_panel=add_to_panel,
        )
        if add_to_panel:
            self.add_object(image)
            if action is not None:
                self.mainwindow.historypanel.register_action_outputs(
                    action, [get_uuid(image)]
                )
        return image

    def toggle_show_contrast(self, state: bool) -> None:
        """Toggle show contrast option"""
        Conf.show_contrast.set(state)
        self.refresh_plot("selected", True, False)
