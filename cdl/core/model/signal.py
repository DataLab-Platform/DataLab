# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal object and related classes
---------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator
from uuid import uuid4

import guidata.dataset as gds
import numpy as np
import scipy.signal as sps
from guidata.configtools import get_icon
from guidata.dataset import restore_dataset, update_dataset
from guidata.qthelpers import exec_dialog
from numpy import ma
from plotpy.builder import make
from plotpy.tools import EditPointTool
from qtpy import QtWidgets as QW

from cdl.algorithms.signal import GaussianModel, LorentzianModel, VoigtModel
from cdl.config import Conf, _
from cdl.core.model import base

if TYPE_CHECKING:
    from plotpy.items import CurveItem
    from plotpy.plot import PlotDialog
    from plotpy.styles import CurveParam


class CurveStyles:
    """Object to manage curve styles"""

    #: Curve colors
    COLORS = (
        "#1f77b4",  # muted blue
        "#ff7f0e",  # safety orange
        "#2ca02c",  # cooked asparagus green
        "#d62728",  # brick red
        "#9467bd",  # muted purple
        "#8c564b",  # chestnut brown
        "#e377c2",  # raspberry yogurt pink
        "#7f7f7f",  # gray
        "#bcbd22",  # curry yellow-green
        "#17becf",  # blue-teal
    )
    #: Curve line styles
    LINESTYLES = ("SolidLine", "DashLine", "DashDotLine", "DashDotDotLine")

    def __init__(self) -> None:
        self.__suspend = False
        self.curve_style = self.style_generator()

    @staticmethod
    def style_generator() -> Generator[tuple[str, str], None, None]:
        """Cycling through curve styles"""
        while True:
            for linestyle in CurveStyles.LINESTYLES:
                for color in CurveStyles.COLORS:
                    yield (color, linestyle)

    def apply_style(self, param: CurveParam) -> None:
        """Apply style to curve"""
        if self.__suspend:
            # Suspend mode: always apply the first style
            color, linestyle = CurveStyles.COLORS[0], CurveStyles.LINESTYLES[0]
        else:
            color, linestyle = next(self.curve_style)
        param.line.color = color
        param.line.style = linestyle
        param.symbol.marker = "NoSymbol"

    def reset_styles(self) -> None:
        """Reset styles"""
        self.curve_style = self.style_generator()

    @contextmanager
    def alternative(
        self, other_style_generator: Generator[tuple[str, str], None, None]
    ) -> Generator[None, None, None]:
        """Use an alternative style generator"""
        old_style_generator = self.curve_style
        self.curve_style = other_style_generator
        yield
        self.curve_style = old_style_generator

    @contextmanager
    def suspend(self) -> Generator[None, None, None]:
        """Suspend style generator"""
        self.__suspend = True
        yield
        self.__suspend = False


CURVESTYLES = CurveStyles()  # This is the unique instance of the CurveStyles class


class ROI1DParam(gds.DataSet):
    """Signal ROI parameters"""

    xmin = gds.FloatItem(_("First point coordinate"))
    xmax = gds.FloatItem(_("Last point coordinate"))

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        imin, imax = np.searchsorted(obj.x, [self.xmin, self.xmax])
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])


def apply_downsampling(item: CurveItem, do_not_update: bool = False) -> None:
    """Apply downsampling to curve item

    Args:
        item: curve item
        do_not_update: if True, do not update the item even if the downsampling
         parameters have changed
    """
    old_use_dsamp = item.param.use_dsamp
    item.param.use_dsamp = False
    if Conf.view.sig_autodownsampling.get():
        nbpoints = item.get_data()[0].size
        maxpoints = Conf.view.sig_autodownsampling_maxpoints.get()
        if nbpoints > 5 * maxpoints:
            item.param.use_dsamp = True
            item.param.dsamp_factor = nbpoints // maxpoints
    if not do_not_update and old_use_dsamp != item.param.use_dsamp:
        item.update_data()


class SignalObj(gds.DataSet, base.BaseObj):
    """Signal object"""

    PREFIX = "s"
    CONF_FMT = Conf.view.sig_format
    DEFAULT_FMT = "g"
    VALID_DTYPES = (np.float32, np.float64, np.complex128)

    uuid = gds.StringItem("UUID").set_prop("display", hide=True)

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data and metadata"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    xydata = gds.FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    metadata = gds.DictItem(_("Metadata"), default={})
    _e_datag = gds.EndGroup(_("Data and metadata"))

    _unitsg = gds.BeginGroup(_("Titles and units"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    _tabs_u = gds.BeginTabGroup("units")
    _unitsx = gds.BeginGroup(_("X-axis"))
    xlabel = gds.StringItem(_("Title"), default="")
    xunit = gds.StringItem(_("Unit"), default="")
    _e_unitsx = gds.EndGroup(_("X-axis"))
    _unitsy = gds.BeginGroup(_("Y-axis"))
    ylabel = gds.StringItem(_("Title"), default="")
    yunit = gds.StringItem(_("Unit"), default="")
    _e_unitsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_u = gds.EndTabGroup("units")
    _e_unitsg = gds.EndGroup(_("Titles and units"))

    _scalesg = gds.BeginGroup(_("Scales"))
    _prop_autoscale = gds.GetAttrProp("autoscale")
    autoscale = gds.BoolItem(_("Auto scale"), default=True).set_prop(
        "display", store=_prop_autoscale
    )
    _tabs_b = gds.BeginTabGroup("bounds")
    _boundsx = gds.BeginGroup(_("X-axis"))
    xscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    xscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    xscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsx = gds.EndGroup(_("X-axis"))
    _boundsy = gds.BeginGroup(_("Y-axis"))
    yscalelog = gds.BoolItem(_("Logarithmic scale"), default=False)
    yscalemin = gds.FloatItem(_("Lower bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    yscalemax = gds.FloatItem(_("Upper bound"), check=False).set_prop(
        "display", active=gds.NotProp(_prop_autoscale)
    )
    _e_boundsy = gds.EndGroup(_("Y-axis"))
    _e_tabs_b = gds.EndTabGroup("bounds")
    _e_scalesg = gds.EndGroup(_("Scales"))

    _e_tabs = gds.EndTabGroup("all")

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title: title
            comment: comment
            icon: icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)
        self.regenerate_uuid()
        self._maskdata_cache = None

    def regenerate_uuid(self):
        """Regenerate UUID

        This method is used to regenerate UUID after loading the object from a file.
        This is required to avoid UUID conflicts when loading objects from file
        without clearing the workspace first.
        """
        self.uuid = str(uuid4())

    def copy(
        self, title: str | None = None, dtype: np.dtype | None = None
    ) -> SignalObj:
        """Copy object.

        Args:
            title: title
            dtype: data type

        Returns:
            Copied object
        """
        title = self.title if title is None else title
        obj = SignalObj(title=title)
        obj.title = title
        obj.xlabel = self.xlabel
        obj.xunit = self.xunit
        obj.yunit = self.yunit
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        obj.metadata = base.deepcopy_metadata(self.metadata)
        obj.xydata = np.array(self.xydata, copy=True, dtype=dtype)
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:  # pylint: disable=unused-argument
        """Change data type.

        Args:
            Data type
        """
        raise RuntimeError("Setting data type is not support for signals")

    def set_xydata(
        self,
        x: np.ndarray | list,
        y: np.ndarray | list,
        dx: np.ndarray | list | None = None,
        dy: np.ndarray | list | None = None,
    ) -> None:
        """Set xy data

        Args:
            x: x data
            y: y data
            dx: dx data (optional: error bars)
            dy: dy data (optional: error bars)
        """
        if x is not None:
            x = np.array(x)
        if y is not None:
            y = np.array(y)
        if dx is not None:
            dx = np.array(dx)
        if dy is not None:
            dy = np.array(dy)
        if dx is None and dy is None:
            self.xydata = np.vstack([x, y])
        else:
            if dx is None:
                dx = np.zeros_like(dy)
            if dy is None:
                dy = np.zeros_like(dx)
            self.xydata = np.vstack((x, y, dx, dy))

    def __get_x(self) -> np.ndarray | None:
        """Get x data"""
        if self.xydata is not None:
            return self.xydata[0]
        return None

    def __set_x(self, data) -> None:
        """Set x data"""
        self.xydata[0] = np.array(data)

    def __get_y(self) -> np.ndarray | None:
        """Get y data"""
        if self.xydata is not None:
            return self.xydata[1]
        return None

    def __set_y(self, data) -> None:
        """Set y data"""
        self.xydata[1] = np.array(data)

    def __get_dx(self) -> np.ndarray | None:
        """Get dx data"""
        if self.xydata is not None and len(self.xydata) > 2:
            return self.xydata[2]
        return None

    def __set_dx(self, data) -> None:
        """Set dx data"""
        if self.xydata is not None and len(self.xydata) > 2:
            self.xydata[2] = np.array(data)
        else:
            raise ValueError("dx data not available")

    def __get_dy(self) -> np.ndarray | None:
        """Get dy data"""
        if self.xydata is not None and len(self.xydata) > 3:
            return self.xydata[3]
        return None

    def __set_dy(self, data) -> None:
        """Set dy data"""
        if self.xydata is not None and len(self.xydata) > 3:
            self.xydata[3] = np.array(data)
        else:
            raise ValueError("dy data not available")

    x = property(__get_x, __set_x)
    y = data = property(__get_y, __set_y)
    dx = property(__get_dx, __set_dx)
    dy = property(__get_dy, __set_dy)

    def get_data(self, roi_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index: ROI index

        Returns:
            Data
        """
        if self.roi is None or roi_index is None:
            return self.x, self.y
        i1, i2 = self.roi[roi_index, :]
        return self.x[i1:i2], self.y[i1:i2]

    def update_plot_item_parameters(self, item: CurveItem) -> None:
        """Update plot item parameters from object data/metadata

        Takes into account a subset of plot item parameters. Those parameters may
        have been overriden by object metadata entries or other object data. The goal
        is to update the plot item accordingly.

        This is *almost* the inverse operation of `update_metadata_from_plot_item`.

        Args:
            item: plot item
        """
        update_dataset(item.param.line, self.metadata)
        update_dataset(item.param.symbol, self.metadata)
        super().update_plot_item_parameters(item)

    def update_metadata_from_plot_item(self, item: CurveItem) -> None:
        """Update metadata from plot item.

        Takes into account a subset of plot item parameters. Those parameters may
        have been modified by the user through the plot item GUI. The goal is to
        update the metadata accordingly.

        This is *almost* the inverse operation of `update_plot_item_parameters`.

        Args:
            item: plot item
        """
        super().update_metadata_from_plot_item(item)
        restore_dataset(item.param.line, self.metadata)
        restore_dataset(item.param.symbol, self.metadata)

    def make_item(self, update_from: CurveItem = None) -> CurveItem:
        """Make plot item from data.

        Args:
            update_from: plot item to update from

        Returns:
            Plot item
        """
        if len(self.xydata) in (2, 3, 4):
            if len(self.xydata) == 2:  # x, y signal
                x, y = self.xydata
                item = make.mcurve(x.real, y.real, label=self.title)
            elif len(self.xydata) == 3:  # x, y, dy error bar signal
                x, y, dy = self.xydata
                item = make.merror(x.real, y.real, dy.real, label=self.title)
            elif len(self.xydata) == 4:  # x, y, dx, dy error bar signal
                x, y, dx, dy = self.xydata
                item = make.merror(x.real, y.real, dx.real, dy.real, label=self.title)
            CURVESTYLES.apply_style(item.param)
            apply_downsampling(item, do_not_update=True)
        else:
            raise RuntimeError("data not supported")
        if update_from is None:
            self.update_plot_item_parameters(item)
        else:
            update_dataset(item.param, update_from.param)
            item.update_params()
        return item

    def update_item(self, item: CurveItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item: plot item
            data_changed: if True, data has changed
        """
        if data_changed:
            if len(self.xydata) == 2:  # x, y signal
                x, y = self.xydata
                item.set_data(x.real, y.real)
            elif len(self.xydata) == 3:  # x, y, dy error bar signal
                x, y, dy = self.xydata
                item.set_data(x.real, y.real, dy=dy.real)
            elif len(self.xydata) == 4:  # x, y, dx, dy error bar signal
                x, y, dx, dy = self.xydata
                item.set_data(x.real, y.real, dx.real, dy.real)
        item.param.label = self.title
        apply_downsampling(item)
        self.update_plot_item_parameters(item)

    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes.

        Args:
            coords: coordinates

        Returns:
            Indexes
        """
        indexes = np.array(coords, int)
        for row in range(indexes.shape[0]):
            for col in range(indexes.shape[1]):
                x0 = coords[row][col]
                indexes[row, col] = np.abs(self.x - x0).argmin()
        return indexes

    def get_roi_param(self, title: str, *defaults: int) -> ROI1DParam:
        """Return ROI parameters dataset (converting ROI point indexes to coordinates)

        Args:
            title: title
            *defaults: default values (first, last point indexes)

        Returns:
            ROI parameters dataset (containing the ROI coordinates: first and last X)
        """
        i0, i1 = defaults
        param = ROI1DParam(title)
        param.xmin = self.x[i0]
        param.xmax = self.x[i1]
        param.set_global_prop("data", unit=self.xunit)
        return param

    def params_to_roidata(self, params: gds.DataSetGroup) -> np.ndarray:
        """Convert ROI dataset group to ROI array data.

        Args:
            params: ROI dataset group

        Returns:
            ROI array data
        """
        roilist = []
        for roiparam in params.datasets:
            roiparam: ROI1DParam
            idx1 = np.searchsorted(self.x, roiparam.xmin)
            idx2 = np.searchsorted(self.x, roiparam.xmax)
            roilist.append([idx1, idx2])
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(self, fmt: str, lbl: bool, editable: bool):
        """Return a new ROI item from scratch

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable
        """
        # We take the real part of the x data to avoid `ComplexWarning` warnings
        # when creating and manipulating the `XRangeSelection` shape (`plotpy`)
        xmin, xmax = self.x.real.min(), self.x.real.max()
        xdelta = (xmax - xmin) * 0.2
        coords = xmin + xdelta, xmax - xdelta
        return base.make_roi_item(
            lambda x, y, _title: make.range(x, y),
            coords,
            "ROI",
            fmt,
            lbl,
            editable,
            option=self.PREFIX,
        )

    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True):
        """Make plot item representing a Region of Interest.

        Args:
            fmt: format string
            lbl: if True, add label
            editable: if True, ROI is editable

        Yields:
            Plot item
        """
        if self.roi is not None:
            # We take the real part of the x data to avoid `ComplexWarning` warnings
            # when creating and manipulating the `XRangeSelection` shape (`plotpy`)
            for index, coords in enumerate(self.x.real[self.roi]):
                yield base.make_roi_item(
                    lambda x, y, _title: make.range(x, y),
                    coords,
                    f"ROI{index:02d}",
                    fmt,
                    lbl,
                    editable,
                    option=self.PREFIX,
                )

    @property
    def maskdata(self) -> np.ndarray:
        """Return masked data (areas outside defined regions of interest)

        Returns:
            Masked data
        """
        roi_changed = self.roi_has_changed()
        if self.roi is None:
            if roi_changed:
                self._maskdata_cache = None
        elif roi_changed or self._maskdata_cache is None:
            mask = np.ones_like(self.xydata, dtype=bool)
            for roirow in self.roi:
                mask[:, roirow[0] : roirow[1]] = False
            self._maskdata_cache = mask
        return self._maskdata_cache

    def get_masked_view(self) -> ma.MaskedArray:
        """Return masked view for data

        Returns:
            Masked view
        """
        self.data: np.ndarray
        view = self.data.view(ma.MaskedArray)
        view.mask = self.maskdata
        return view

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title: title (if None, use signal title)
        """
        title = self.title if title is None else title
        if title:
            label = make.label(title, "TL", (0, 0), "TL")
            self.add_annotations_from_items([label])


def create_signal(
    title: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create a new Signal object.

    Args:
        title: signal title
        x: X data
        y: Y data
        dx: dX data (optional: error bars)
        dy: dY data (optional: error bars)
        metadata: signal metadata
        units: X, Y units (tuple of strings)
        labels: X, Y labels (tuple of strings)

    Returns:
        Signal object
    """
    assert isinstance(title, str)
    signal = SignalObj(title=title)
    signal.title = title
    signal.set_xydata(x, y, dx=dx, dy=dy)
    if units is not None:
        signal.xunit, signal.yunit = units
    if labels is not None:
        signal.xlabel, signal.ylabel = labels
    if metadata is not None:
        signal.metadata.update(metadata)
    return signal


class SignalTypes(base.Choices):
    """Signal types"""

    #: Signal filled with zeros
    ZEROS = _("zeros")
    #: Gaussian function
    GAUSS = _("gaussian")
    #: Lorentzian function
    LORENTZ = _("lorentzian")
    #: Voigt function
    VOIGT = "Voigt"
    #: Random signal (uniform law)
    UNIFORMRANDOM = _("random (uniform law)")
    #: Random signal (normal law)
    NORMALRANDOM = _("random (normal law)")
    #: Sinusoid
    SINUS = _("sinus")
    #: Cosinusoid
    COSINUS = _("cosinus")
    #: Sawtooth function
    SAWTOOTH = _("sawtooth")
    #: Triangle function
    TRIANGLE = _("triangle")
    #: Square function
    SQUARE = _("square")
    #: Cardinal sine
    SINC = _("cardinal sine")
    #: Step function
    STEP = _("step")
    #: Exponential function
    EXPONENTIAL = _("exponential")
    #: Pulse function
    PULSE = _("pulse")
    #: Polynomial function
    POLYNOMIAL = _("polynomial")
    #: Experimental function
    EXPERIMENTAL = _("experimental")


class GaussLorentzVoigtParam(gds.DataSet):
    """Parameters for Gaussian and Lorentzian functions"""

    a = gds.FloatItem("A", default=1.0)
    ymin = gds.FloatItem("Ymin", default=0.0).set_pos(col=1)
    sigma = gds.FloatItem("σ", default=1.0)
    mu = gds.FloatItem("μ", default=0.0).set_pos(col=1)


class FreqUnits(base.Choices):
    """Frequency units"""

    HZ = "Hz"
    KHZ = "kHz"
    MHZ = "MHz"
    GHZ = "GHz"

    @classmethod
    def convert_in_hz(cls, value, unit):
        """Convert value in Hz"""
        factor = {cls.HZ: 1, cls.KHZ: 1e3, cls.MHZ: 1e6, cls.GHZ: 1e9}.get(unit)
        if factor is None:
            raise ValueError(f"Unknown unit: {unit}")
        return value * factor


class PeriodicParam(gds.DataSet):
    """Parameters for periodic functions"""

    def get_frequency_in_hz(self):
        """Return frequency in Hz"""
        return FreqUnits.convert_in_hz(self.freq, self.freq_unit)

    a = gds.FloatItem("A", default=1.0)
    ymin = gds.FloatItem("Ymin", default=0.0).set_pos(col=1)
    freq = gds.FloatItem(_("Frequency"), default=1.0)
    freq_unit = gds.ChoiceItem(
        _("Unit"), FreqUnits.get_choices(), default=FreqUnits.HZ
    ).set_pos(col=1)
    phase = gds.FloatItem(_("Phase"), default=0.0, unit="°").set_pos(col=1)


class StepParam(gds.DataSet):
    """Parameters for step function"""

    a1 = gds.FloatItem("A1", default=0.0)
    a2 = gds.FloatItem("A2", default=1.0).set_pos(col=1)
    x0 = gds.FloatItem("X0", default=0.0)


class ExponentialParam(gds.DataSet):
    """Parameters for exponential function"""

    a = gds.FloatItem("A", default=1.0)
    offset = gds.FloatItem(_("Offset"), default=0.0)
    exponent = gds.FloatItem(_("Exponent"), default=1.0)


class PulseParam(gds.DataSet):
    """Parameters for pulse function"""

    amp = gds.FloatItem("Amplitude", default=1.0)
    start = gds.FloatItem(_("Start"), default=0.0).set_pos(col=1)
    offset = gds.FloatItem(_("Offset"), default=0.0)
    stop = gds.FloatItem(_("End"), default=0.0).set_pos(col=1)


class PolyParam(gds.DataSet):
    """Parameters for polynomial function"""

    a0 = gds.FloatItem("a0", default=1.0)
    a3 = gds.FloatItem("a3", default=0.0).set_pos(col=1)
    a1 = gds.FloatItem("a1", default=1.0)
    a4 = gds.FloatItem("a4", default=0.0).set_pos(col=1)
    a2 = gds.FloatItem("a2", default=0.0)
    a5 = gds.FloatItem("a5", default=0.0).set_pos(col=1)


class ExperSignalParam(gds.DataSet):
    """Parameters for experimental signal"""

    size = gds.IntItem("Size", default=5).set_prop("display", hide=True)
    xyarray = gds.FloatArrayItem(
        "XY Values",
        format="%g",
    )
    xmin = gds.FloatItem("Min", default=0).set_prop("display", hide=True)
    xmax = gds.FloatItem("Max", default=1).set_prop("display", hide=True)

    def edit_curve(self, *args) -> None:  # pylint: disable=unused-argument
        """Edit experimental curve"""
        win: PlotDialog = make.dialog(
            wintitle=_("Select one point then press OK to accept"),
            edit=True,
            type="curve",
        )
        edit_tool = win.manager.add_tool(
            EditPointTool, title=_("Edit experimental curve")
        )
        edit_tool.activate()
        plot = win.manager.get_plot()
        x, y = self.xyarray[:, 0], self.xyarray[:, 1]
        curve = make.mcurve(x, y, "-+")
        plot.add_item(curve)
        plot.set_active_item(curve)

        insert_btn = QW.QPushButton(_("Insert point"), win)
        insert_btn.clicked.connect(edit_tool.trigger_insert_point_at_selection)
        win.button_layout.insertWidget(0, insert_btn)

        exec_dialog(win)

        new_x, new_y = curve.get_data()
        self.xmax = new_x.max()
        self.xmin = new_x.min()
        self.size = new_x.size
        self.xyarray = np.vstack((new_x, new_y)).T

    btn_curve_edit = gds.ButtonItem(
        "Edit curve", callback=edit_curve, icon="signal.svg"
    )

    def setup_array(
        self,
        size: int | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
    ) -> None:
        """Setup the xyarray from size, xmin and xmax (use the current values is not
        provided)

        Args:
            size: xyarray size (default: None)
            xmin: X min (default: None)
            xmax: X max (default: None)
        """
        self.size = size or self.size
        self.xmin = xmin or self.xmin
        self.xmax = xmax or self.xmax
        x_arr = np.linspace(self.xmin, self.xmax, self.size)  # type: ignore
        self.xyarray = np.vstack((x_arr, x_arr)).T


class NewSignalParam(gds.DataSet):
    """New signal dataset"""

    hide_signal_type = False

    title = gds.StringItem(_("Title"))
    xmin = gds.FloatItem("Xmin", default=-10.0)
    xmax = gds.FloatItem("Xmax", default=10.0)
    size = gds.IntItem(
        _("Size"), help=_("Signal size (total number of points)"), min=1, default=500
    )
    stype = gds.ChoiceItem(_("Type"), SignalTypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_signal_type")
    )


DEFAULT_TITLE = _("Untitled signal")


def new_signal_param(
    title: str | None = None,
    stype: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    size: int | None = None,
) -> NewSignalParam:
    """Create a new Signal dataset instance.

    Args:
        title: dataset title (default: None, uses default title)
        stype: signal type (default: None, uses default type)
        xmin: X min (default: None, uses default value)
        xmax: X max (default: None, uses default value)
        size: signal size (default: None, uses default value)

    Returns:
        NewSignalParam: new signal dataset instance
    """
    title = DEFAULT_TITLE if title is None else title
    param = NewSignalParam(title=title, icon=get_icon("new_signal.svg"))
    param.title = title
    if xmin is not None:
        param.xmin = xmin
    if xmax is not None:
        param.xmax = xmax
    if size is not None:
        param.size = size
    if stype is not None:
        param.stype = stype
    return param


SIG_NB = 0


def triangle_func(xarr: np.ndarray) -> np.ndarray:
    """Triangle function

    Args:
        xarr: x data
    """
    return sps.sawtooth(xarr, width=0.5)


def create_signal_from_param(
    newparam: NewSignalParam,
    addparam: gds.DataSet | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> SignalObj | None:
    """Create a new Signal object from a dialog box.

    Args:
        newparam: new signal parameters
        addparam: additional parameters
        edit: Open a dialog box to edit parameters (default: False)
        parent: parent widget

    Returns:
        Signal object or None if canceled
    """
    global SIG_NB  # pylint: disable=global-statement
    if newparam is None:
        newparam = new_signal_param()
    incr_sig_nb = not newparam.title
    if incr_sig_nb:
        newparam.title = f"{newparam.title} {SIG_NB + 1:d}"
    if not edit or addparam is not None or newparam.edit(parent=parent):
        prefix = newparam.stype.name.lower()
        if incr_sig_nb:
            SIG_NB += 1
        signal = create_signal(newparam.title)
        xarr = np.linspace(newparam.xmin, newparam.xmax, newparam.size)
        p = addparam
        if newparam.stype == SignalTypes.ZEROS:
            signal.set_xydata(xarr, np.zeros(newparam.size))
        elif newparam.stype in (SignalTypes.UNIFORMRANDOM, SignalTypes.NORMALRANDOM):
            pclass = {
                SignalTypes.UNIFORMRANDOM: base.UniformRandomParam,
                SignalTypes.NORMALRANDOM: base.NormalRandomParam,
            }[newparam.stype]
            if p is None:
                p = pclass(_("Signal") + " - " + prefix)
            if edit and not p.edit(parent=parent):
                return None
            rng = np.random.default_rng(p.seed)
            if newparam.stype == SignalTypes.UNIFORMRANDOM:
                yarr = rng.random((newparam.size,)) * (p.vmax - p.vmin) + p.vmin
                if signal.title == DEFAULT_TITLE:
                    signal.title = f"{prefix}(vmin={p.vmin:.3g},vmax={p.vmax:.3g})"
            elif newparam.stype == SignalTypes.NORMALRANDOM:
                yarr = rng.normal(p.mu, p.sigma, size=(newparam.size,))
                if signal.title == DEFAULT_TITLE:
                    signal.title = f"{prefix}(mu={p.mu:.3g},sigma={p.sigma:.3g})"
            else:
                raise NotImplementedError(f"New param type: {prefix}")
            signal.set_xydata(xarr, yarr)
        elif newparam.stype in (
            SignalTypes.GAUSS,
            SignalTypes.LORENTZ,
            SignalTypes.VOIGT,
        ):
            func, title = {
                SignalTypes.GAUSS: (GaussianModel.func, _("Gaussian")),
                SignalTypes.LORENTZ: (LorentzianModel.func, _("Lorentzian")),
                SignalTypes.VOIGT: (VoigtModel.func, "Voigt"),
            }[newparam.stype]
            if p is None:
                p = GaussLorentzVoigtParam(title)
            if edit and not p.edit(parent=parent):
                return None
            yarr = func(xarr, p.a, p.sigma, p.mu, p.ymin)
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = (
                    f"{prefix}(a={p.a:.3g},sigma={p.sigma:.3g},"
                    f"mu={p.mu:.3g},ymin={p.ymin:.3g})"
                )
        elif newparam.stype in (
            SignalTypes.SINUS,
            SignalTypes.COSINUS,
            SignalTypes.SAWTOOTH,
            SignalTypes.TRIANGLE,
            SignalTypes.SQUARE,
            SignalTypes.SINC,
        ):
            func, title = {
                SignalTypes.SINUS: (np.sin, _("Sinusoid")),
                SignalTypes.COSINUS: (np.cos, _("Sinusoid")),
                SignalTypes.SAWTOOTH: (sps.sawtooth, _("Sawtooth function")),
                SignalTypes.TRIANGLE: (triangle_func, _("Triangle function")),
                SignalTypes.SQUARE: (sps.square, _("Square function")),
                SignalTypes.SINC: (np.sinc, _("Cardinal sine")),
            }[newparam.stype]
            if p is None:
                p = PeriodicParam(title)
            if edit and not p.edit(parent=parent):
                return None
            freq = p.get_frequency_in_hz()
            yarr = p.a * func(2 * np.pi * freq * xarr + np.deg2rad(p.phase)) + p.ymin
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = (
                    f"{prefix}(f={p.freq:.3g} {p.freq_unit.value}),"
                    f"a={p.a:.3g},ymin={p.ymin:.3g},phase={p.phase:.3g}°)"
                )
        elif newparam.stype == SignalTypes.STEP:
            if p is None:
                p = StepParam(
                    _("Step function"), comment="y(x) = a1 if x <= x0 else a2"
                )
            if edit and not p.edit(parent=parent):
                return None
            yarr = np.ones_like(xarr) * p.a1
            yarr[xarr > p.x0] = p.a2
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = f"{prefix}(x0={p.x0:.3g},a1={p.a1:.3g},a2={p.a2:.3g})"
        elif newparam.stype is SignalTypes.EXPONENTIAL:
            if p is None:
                p = ExponentialParam(
                    _("Exponential function"),
                    comment="y(x) = a.e<sup>exponent.x</sup> + offset",
                )
            if edit and not p.edit(parent=parent):
                return None
            yarr = p.a * np.exp(p.exponent * xarr) + p.offset
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = (
                    f"{prefix}(a={p.a:.3g},exponent={p.exponent:.3g},"
                    f"offset={p.offset:.3g})"
                )
        elif newparam.stype is SignalTypes.PULSE:
            if p is None:
                p = PulseParam(
                    _("Pulse function"),
                    comment="y(x) = offset + amp if start <= x <= stop else offset",
                )
            if edit and not p.edit(parent=parent):
                return None
            yarr = np.full_like(xarr, p.offset)
            yarr[(xarr >= p.start) & (xarr <= p.stop)] += p.amp
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = (
                    f"{prefix}(start={p.start:.3g},stop={p.stop:.3g}"
                    f",offset={p.offset:.3g})"
                )
        elif newparam.stype is SignalTypes.POLYNOMIAL:
            if p is None:
                p = PolyParam(
                    _("Polynomial function"),
                    comment=(
                        "y(x) = a<sub>0</sub> + a<sub>1</sub>.x + "
                        "a<sub>2</sub>.x<sup>2</sup> + a<sub>3</sub>.x<sup>3</sup>"
                        " + a<sub>4</sub>.x<sup>4</sup> + a<sub>5</sub>.x<sup>5</sup>"
                    ),
                )
            if edit and not p.edit(parent=parent):
                return None
            yarr = np.polyval([p.a5, p.a4, p.a3, p.a2, p.a1, p.a0], xarr)
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = (
                    f"{prefix}(a0={p.a0:2g},a1={p.a1:2g},a2={p.a2:2g},"
                    f"a3={p.a3:2g},a4={p.a4:2g},a5={p.a5:2g})"
                )
        elif newparam.stype is SignalTypes.EXPERIMENTAL:
            p2 = ExperSignalParam(_("Experimental points"))
            p2.setup_array(size=newparam.size, xmin=newparam.xmin, xmax=newparam.xmax)
            if edit and not p2.edit(parent=parent):
                return None
            signal.xydata = p2.xyarray.T
            if signal.title == DEFAULT_TITLE:
                signal.title = f"{prefix}(npts={p2.size})"
        return signal
    return None
