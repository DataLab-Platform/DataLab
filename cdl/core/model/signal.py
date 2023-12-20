# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Signal object and related classes
---------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from uuid import uuid4

import guidata.dataset as gds
import numpy as np
import scipy.signal as sps
from guidata.configtools import get_icon
from guidata.dataset import restore_dataset, update_dataset
from plotpy.builder import make

from cdl.algorithms import fit
from cdl.config import Conf, _
from cdl.core.model import base
from cdl.env import execenv

if TYPE_CHECKING:  # pragma: no cover
    from plotpy.items import CurveItem
    from plotpy.styles import CurveParam
    from qtpy import QtWidgets as QW


class CurveStyles:
    """Object to manage curve styles"""

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
    LINESTYLES = ("SolidLine", "DashLine", "DashDotLine", "DashDotDotLine")

    def style_generator():  # pylint: disable=no-method-argument
        """Cycling through curve styles"""
        while True:
            for linestyle in CurveStyles.LINESTYLES:
                for color in CurveStyles.COLORS:
                    yield (color, linestyle)

    CURVE_STYLE = style_generator()

    @classmethod
    def apply_style(cls, param: CurveParam):
        """Apply style to curve"""
        color, linestyle = next(cls.CURVE_STYLE)
        param.line.color = color
        param.line.style = linestyle
        param.symbol.marker = "NoSymbol"


class ROIParam(gds.DataSet):
    """Signal ROI parameters"""

    col1 = gds.IntItem(_("First point index"))
    col2 = gds.IntItem(_("Last point index"))


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

    _e_tabs = gds.EndTabGroup("all")

    def __init__(self, title=None, comment=None, icon=""):
        """Constructor

        Args:
            title (str): title
            comment (str): comment
            icon (str): icon
        """
        gds.DataSet.__init__(self, title, comment, icon)
        base.BaseObj.__init__(self)
        self.uuid = str(uuid4())

    def copy(
        self, title: str | None = None, dtype: np.dtype | None = None
    ) -> SignalObj:
        """Copy object.

        Args:
            title (str): title
            dtype (numpy.dtype): data type

        Returns:
            SignalObj: copied object
        """
        title = self.title if title is None else title
        obj = SignalObj(title=title)
        obj.title = title
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        obj.metadata = deepcopy(self.metadata)
        obj.xydata = np.array(self.xydata, copy=True, dtype=dtype)
        return obj

    def set_data_type(self, dtype: np.dtype) -> None:  # pylint: disable=unused-argument
        """Change data type.

        Args:
            dtype (numpy.dtype): data type
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
            x (numpy.ndarray): x data
            y (numpy.ndarray): y data
            dx (numpy.ndarray): dx data (optional: error bars)
            dy (numpy.ndarray): dy data (optional: error bars)
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

    def get_data(self, roi_index: int | None = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).

        Args:
            roi_index (int): ROI index

        Returns:
            numpy.ndarray: data
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
            update_from (CurveItem): plot item to update from

        Returns:
            CurveItem: plot item
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
            CurveStyles.apply_style(item.param)
        else:
            raise RuntimeError("data not supported")
        if update_from is None:
            if execenv.demo_mode:
                item.param.line.width = 3
            self.update_plot_item_parameters(item)
        else:
            update_dataset(item.param, update_from.param)
            item.update_params()
        return item

    def update_item(self, item: CurveItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item (CurveItem): plot item
            data_changed (bool): if True, data has changed
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
        self.update_plot_item_parameters(item)

    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes.

        Args:
            coords (list): coordinates

        Returns:
            numpy.ndarray: indexes
        """
        indexes = np.array(coords, int)
        for row in range(indexes.shape[0]):
            for col in range(indexes.shape[1]):
                x0 = coords[row][col]
                indexes[row, col] = np.abs(self.x - x0).argmin()
        return indexes

    def get_roi_param(self, title: str, *defaults) -> gds.DataSet:
        """Return ROI parameters dataset.

        Args:
            title (str): title
            *defaults: default values
        """
        imax = len(self.x) - 1
        i0, i1 = defaults
        param = ROIParam(title)
        param.col1 = i0
        param.col2 = i1
        param.set_global_prop("data", min=-1, max=imax)
        return param

    @staticmethod
    def params_to_roidata(params: gds.DataSetGroup) -> np.ndarray:
        """Convert ROI dataset group to ROI array data.

        Args:
            params (DataSetGroup): ROI dataset group

        Returns:
            numpy.ndarray: ROI array data
        """
        roilist = []
        for roiparam in params.datasets:
            roiparam: ROIParam
            roilist.append([roiparam.col1, roiparam.col2])
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(self, fmt: str, lbl: bool, editable: bool):
        """Return a new ROI item from scratch

        Args:
            fmt (str): format string
            lbl (bool): if True, add label
            editable (bool): if True, ROI is editable
        """
        coords = self.x.min(), self.x.max()
        return base.make_roi_item(
            lambda x, y, _title: make.range(x, y),
            coords,
            "ROI",
            fmt,
            lbl,
            editable,
            option="shape/drag",
        )

    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True):
        """Make plot item representing a Region of Interest.

        Args:
            fmt (str): format string
            lbl (bool): if True, add label
            editable (bool): if True, ROI is editable

        Yields:
            PlotItem: plot item
        """
        if self.roi is not None:
            for index, coords in enumerate(self.x[self.roi]):
                yield base.make_roi_item(
                    lambda x, y, _title: make.range(x, y),
                    coords,
                    f"ROI{index:02d}",
                    fmt,
                    lbl,
                    editable,
                    option="shape/drag",
                )

    def add_label_with_title(self, title: str | None = None) -> None:
        """Add label with title annotation

        Args:
            title (str): title (if None, use signal title)
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
    units: tuple | None = None,
    labels: tuple | None = None,
) -> SignalObj:
    """Create a new Signal object.

    Args:
        title (str): signal title
        x (numpy.ndarray): X data
        y (numpy.ndarray): Y data
        dx (numpy.ndarray): dX data (optional: error bars)
        dy (numpy.ndarray): dY data (optional: error bars)
        metadata (dict): signal metadata
        units (tuple): X, Y units (tuple of strings)
        labels (tuple): X, Y labels (tuple of strings)

    Returns:
        SignalObj: signal object
    """
    assert isinstance(title, str)
    signal = SignalObj()
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

    ZEROS = _("zeros")
    GAUSS = _("gaussian")
    LORENTZ = _("lorentzian")
    VOIGT = "Voigt"
    UNIFORMRANDOM = _("random (uniform law)")
    NORMALRANDOM = _("random (normal law)")
    SINUS = _("sinus")
    COSINUS = _("cosinus")
    SAWTOOTH = _("sawtooth")
    TRIANGLE = _("triangle")
    SQUARE = _("square")
    SINC = _("cardinal sine")
    STEP = _("step")


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
        title (str): dataset title (default: None, uses default title)
        stype (str): signal type (default: None, uses default type)
        xmin (float): X min (default: None, uses default value)
        xmax (float): X max (default: None, uses default value)
        size (int): signal size (default: None, uses default value)

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
        xarr (numpy.ndarray): x data
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
        newparam (NewSignalParam): new signal parameters
        addparam (guidata.dataset.DataSet): additional parameters
        edit (bool): Open a dialog box to edit parameters (default: False)
        parent (QWidget): parent widget

    Returns:
        SignalObj: signal object or None if canceled
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
                SignalTypes.GAUSS: (fit.GaussianModel.func, _("Gaussian")),
                SignalTypes.LORENTZ: (fit.LorentzianModel.func, _("Lorentzian")),
                SignalTypes.VOIGT: (fit.VoigtModel.func, "Voigt"),
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
                p = StepParam(_("Step function"))
            if edit and not p.edit(parent=parent):
                return None
            yarr = np.ones_like(xarr) * p.a1
            yarr[xarr > p.x0] = p.a2
            signal.set_xydata(xarr, yarr)
            if signal.title == DEFAULT_TITLE:
                signal.title = f"{prefix}(x0={p.x0:.3g},a1={p.a1:.3g},a2={p.a2:.3g})"
        return signal
    return None
