# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab Datasets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
import scipy.signal as sps
from guidata.configtools import get_icon
from guidata.utils import update_dataset
from guiqwt.builder import make
from guiqwt.curve import CurveItem
from guiqwt.styles import update_style_attr

from cdl.config import Conf, _
from cdl.core.computation import fit
from cdl.core.model import base
from cdl.env import execenv

if TYPE_CHECKING:
    from qtpy import QtWidgets as QW


class SignalParam(gdt.DataSet, base.ObjectItf):
    """Signal dataset"""

    PREFIX = "s"
    CONF_FMT = Conf.view.sig_format
    DEFAULT_FMT = ".3f"
    VALID_DTYPES = (np.float32, np.float64, np.complex128)

    _tabs = gdt.BeginTabGroup("all")

    _datag = gdt.BeginGroup(_("Data and metadata"))
    title = gdi.StringItem(_("Signal title"), default=_("Untitled"))
    xydata = gdi.FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    metadata = base.MetadataItem(_("Metadata"), default={})
    _e_datag = gdt.EndGroup(_("Data and metadata"))

    _unitsg = gdt.BeginGroup(_("Titles and units"))
    title = gdi.StringItem(_("Signal title"), default=_("Untitled"))
    _tabs_u = gdt.BeginTabGroup("units")
    _unitsx = gdt.BeginGroup(_("X-axis"))
    xlabel = gdi.StringItem(_("Title"), default="")
    xunit = gdi.StringItem(_("Unit"), default="")
    _e_unitsx = gdt.EndGroup(_("X-axis"))
    _unitsy = gdt.BeginGroup(_("Y-axis"))
    ylabel = gdi.StringItem(_("Title"), default="")
    yunit = gdi.StringItem(_("Unit"), default="")
    _e_unitsy = gdt.EndGroup(_("Y-axis"))
    _e_tabs_u = gdt.EndTabGroup("units")
    _e_unitsg = gdt.EndGroup(_("Titles and units"))

    _e_tabs = gdt.EndTabGroup("all")

    def __init__(self, title=None, comment=None, icon=""):
        gdt.DataSet.__init__(self, title, comment, icon)
        base.ObjectItf.__init__(self)

    def copy_data_from(self, other, dtype=None):
        """Copy data from other dataset instance.

        Args:
            other (ObjectItf): other dataset instance
            dtype (numpy.dtype): data type
        """
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        self.metadata = deepcopy(other.metadata)
        self.xydata = np.array(other.xydata, copy=True, dtype=dtype)

    def set_data_type(self, dtype):  # pylint: disable=unused-argument,no-self-use
        """Change data type.

        Args:
            dtype (numpy.dtype): data type
        """
        raise RuntimeError("Setting data type is not support for signals")

    def set_xydata(self, x, y, dx=None, dy=None):
        """Set xy data"""
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
            else:
                dy = np.zeros_like(dx)
            self.xydata = np.vstack((x, y, dx, dy))

    def __get_x(self):
        """Get x data"""
        if self.xydata is not None:
            return self.xydata[0]
        return None

    def __set_x(self, data):
        """Set x data"""
        self.xydata[0] = np.array(data)

    def __get_y(self):
        """Get y data"""
        if self.xydata is not None:
            return self.xydata[1]
        return None

    def __set_y(self, data):
        """Set y data"""
        self.xydata[1] = np.array(data)

    x = property(__get_x, __set_x)
    y = data = property(__get_y, __set_y)

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

    def make_item(self, update_from=None):
        """Make plot item from data.

        Args:
            update_from (ObjectItf): update

        Returns:
            PlotItem: plot item
        """
        if len(self.xydata) == 2:  # x, y signal
            x, y = self.xydata
            item = make.mcurve(x.real, y.real, label=self.title)
        elif len(self.xydata) == 3:  # x, y, dy error bar signal
            x, y, dy = self.xydata
            item = make.merror(x.real, y.real, dy.real, label=self.title)
        elif len(self.xydata) == 4:  # x, y, dx, dy error bar signal
            x, y, dx, dy = self.xydata
            item = make.merror(x.real, y.real, dx.real, dy.real, label=self.title)
        else:
            raise RuntimeError("data not supported")
        if update_from is not None:
            update_dataset(item.curveparam, update_from.curveparam)
        return item

    def update_item(self, item: CurveItem, data_changed: bool = True) -> None:
        """Update plot item from data.

        Args:
            item (PlotItem): plot item
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
        item.curveparam.label = self.title
        if execenv.demo_mode:
            item.curveparam.line.width = 3
        update_style_attr(next(make.style), item.curveparam)
        update_dataset(item.curveparam, self.metadata)
        item.update_params()

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

    def get_roi_param(self, title, *defaults):
        """Return ROI parameters dataset.

        Args:
            title (str): title
            *defaults: default values
        """
        imax = len(self.x) - 1
        i0, i1 = defaults

        class ROIParam(gdt.DataSet):
            """Signal ROI parameters"""

            col1 = gdi.IntItem(_("First point index"), default=i0, min=-1, max=imax)
            col2 = gdi.IntItem(_("Last point index"), default=i1, min=-1, max=imax)

        return ROIParam(title)

    @staticmethod
    def params_to_roidata(params: gdt.DataSetGroup) -> np.ndarray:
        """Convert ROI dataset group to ROI array data.

        Args:
            params (DataSetGroup): ROI dataset group

        Returns:
            numpy.ndarray: ROI array data
        """
        roilist = []
        for roiparam in params.datasets:
            roilist.append([roiparam.col1, roiparam.col2])
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(self, fmt, lbl, editable):
        """Return a new ROI item from scratch"""
        coords = self.x.min(), self.x.max()
        return base.make_roi_item(
            lambda x, y, _title: make.range(x, y), coords, "ROI", fmt, lbl, editable
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
                )


def create_signal(
    title: str,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
    metadata: dict | None = None,
    units: tuple | None = None,
    labels: tuple | None = None,
) -> SignalParam:
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
        SignalParam: signal object
    """
    assert isinstance(title, str)
    signal = SignalParam()
    signal.title = title
    signal.set_xydata(x, y, dx=dx, dy=dy)
    if units is not None:
        signal.xunit, signal.yunit = units
    if labels is not None:
        signal.xlabel, signal.ylabel = labels
    if metadata is None:
        signal.reset_metadata_to_defaults()
    else:
        signal.metadata = metadata
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
    SINC = _("sinc")
    STEP = _("step")


class GaussLorentzVoigtParam(gdt.DataSet):
    """Parameters for Gaussian and Lorentzian functions"""

    a = gdi.FloatItem("A", default=1.0)
    ymin = gdi.FloatItem("Ymin", default=0.0).set_pos(col=1)
    sigma = gdi.FloatItem("σ", default=1.0)
    mu = gdi.FloatItem("μ", default=0.0).set_pos(col=1)


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


class PeriodicParam(gdt.DataSet):
    """Parameters for periodic functions"""

    def get_frequency_in_hz(self):
        """Return frequency in Hz"""
        return FreqUnits.convert_in_hz(self.freq, self.freq_unit)

    a = gdi.FloatItem("A", default=1.0)
    ymin = gdi.FloatItem("Ymin", default=0.0).set_pos(col=1)
    freq = gdi.FloatItem("Frequency", default=1.0)
    freq_unit = gdi.ChoiceItem(
        "Unit", FreqUnits.get_choices(), default=FreqUnits.HZ
    ).set_pos(col=1)
    phase = gdi.FloatItem("Phase", default=0.0, unit="°").set_pos(col=1)


class StepParam(gdt.DataSet):
    """Parameters for step function"""

    a1 = gdi.FloatItem("A1", default=0.0)
    a2 = gdi.FloatItem("A2", default=1.0).set_pos(col=1)
    x0 = gdi.FloatItem("X0", default=0.0)


class SignalParamNew(gdt.DataSet):
    """New signal dataset"""

    title = gdi.StringItem(_("Title"), default=_("Untitled"))
    xmin = gdi.FloatItem("Xmin", default=-10.0)
    xmax = gdi.FloatItem("Xmax", default=10.0)
    size = gdi.IntItem(
        _("Size"), help=_("Signal size (total number of points)"), min=1, default=500
    )
    type = gdi.ChoiceItem(_("Type"), SignalTypes.get_choices())


def new_signal_param(
    title: str | None = None,
    stype: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    size: int | None = None,
) -> SignalParamNew:
    """Create a new Signal dataset instance.

    Args:
        title (str): dataset title (default: None, uses default title)
        stype (str): signal type (default: None, uses default type)
        xmin (float): X min (default: None, uses default value)
        xmax (float): X max (default: None, uses default value)
        size (int): signal size (default: None, uses default value)

    Returns:
        SignalParamNew: new signal dataset instance
    """
    if title is None:
        title = _("Untitled signal")
    param = SignalParamNew(title=title, icon=get_icon("new_signal.svg"))
    param.title = title
    if xmin is not None:
        param.xmin = xmin
    if xmax is not None:
        param.xmax = xmax
    if size is not None:
        param.size = size
    if stype is not None:
        param.type = stype
    return param


SIG_NB = 0


def triangle_func(xarr: np.ndarray) -> np.ndarray:
    """Triangle function"""
    return sps.sawtooth(xarr, width=0.5)


def create_signal_from_param(
    newparam: SignalParamNew,
    addparam: gdt.DataSet | None = None,
    edit: bool = False,
    parent: QW.QWidget | None = None,
) -> SignalParam:
    """Create a new Signal object from a dialog box.

    Args:
        newparam (SignalParamNew): new signal parameters
        addparam (DataSet): additional parameters
        edit (bool): Open a dialog box to edit parameters (default: False)
        parent (QWidget): parent widget

    Returns:
        SignalParam: signal object
    """
    global SIG_NB  # pylint: disable=global-statement
    if newparam is None:
        newparam = new_signal_param()
    incr_sig_nb = not newparam.title
    if incr_sig_nb:
        newparam.title = f"{newparam.title} {SIG_NB + 1:d}"
    if not edit or addparam is not None or newparam.edit(parent=parent):
        prefix = newparam.type.name.lower()
        if incr_sig_nb:
            SIG_NB += 1
        signal = create_signal(newparam.title)
        xarr = np.linspace(newparam.xmin, newparam.xmax, newparam.size)
        p = addparam
        if newparam.type == SignalTypes.ZEROS:
            signal.set_xydata(xarr, np.zeros(newparam.size))
        elif newparam.type in (SignalTypes.UNIFORMRANDOM, SignalTypes.NORMALRANDOM):
            pclass = {
                SignalTypes.UNIFORMRANDOM: base.UniformRandomParam,
                SignalTypes.NORMALRANDOM: base.NormalRandomParam,
            }[newparam.type]
            if p is None:
                p = pclass(_("Signal") + " - " + prefix)
            if edit and not p.edit(parent=parent):
                return None
            rng = np.random.default_rng(p.seed)
            if newparam.type == SignalTypes.UNIFORMRANDOM:
                yarr = rng.random((newparam.size,)) * (p.vmax - p.vmin) + p.vmin
                signal.title = f"{prefix}(vmin={p.vmin:.3g},vmax={p.vmax:.3g})"
            elif newparam.type == SignalTypes.NORMALRANDOM:
                yarr = rng.normal(p.mu, p.sigma, size=(newparam.size,))
                signal.title = f"{prefix}(mu={p.mu:.3g},sigma={p.sigma:.3g})"
            else:
                raise NotImplementedError(f"New param type: {prefix}")
            signal.set_xydata(xarr, yarr)
        elif newparam.type in (
            SignalTypes.GAUSS,
            SignalTypes.LORENTZ,
            SignalTypes.VOIGT,
        ):
            func, title = {
                SignalTypes.GAUSS: (fit.GaussianModel.func, _("Gaussian")),
                SignalTypes.LORENTZ: (fit.LorentzianModel.func, _("Lorentzian")),
                SignalTypes.VOIGT: (fit.VoigtModel.func, _("Voigt")),
            }[newparam.type]
            if p is None:
                p = GaussLorentzVoigtParam(title)
            if edit and not p.edit(parent=parent):
                return None
            yarr = func(xarr, p.a, p.sigma, p.mu, p.ymin)
            signal.set_xydata(xarr, yarr)
            signal.title = (
                f"{prefix}(a={p.a:.3g},sigma={p.sigma:.3g},"
                f"mu={p.mu:.3g},ymin={p.ymin:.3g})"
            )
        elif newparam.type in (
            SignalTypes.SINUS,
            SignalTypes.COSINUS,
            SignalTypes.SAWTOOTH,
            SignalTypes.TRIANGLE,
            SignalTypes.SQUARE,
            SignalTypes.SINC,
        ):
            func, title = {
                SignalTypes.SINUS: (np.sin, _("New sinusoidal function")),
                SignalTypes.COSINUS: (np.cos, _("New cosinusoidal function")),
                SignalTypes.SAWTOOTH: (sps.sawtooth, _("New sawtooth function")),
                SignalTypes.TRIANGLE: (triangle_func, _("New triangle function")),
                SignalTypes.SQUARE: (sps.square, _("New square function")),
                SignalTypes.SINC: (np.sinc, _("New sinc function")),
            }[newparam.type]
            if p is None:
                p = PeriodicParam(title)
            if edit and not p.edit(parent=parent):
                return None
            freq = p.get_frequency_in_hz()
            yarr = p.a * func(2 * np.pi * freq * xarr + np.deg2rad(p.phase)) + p.ymin
            signal.set_xydata(xarr, yarr)
            signal.title = (
                f"{prefix}(f={p.freq:.3g} {p.freq_unit.name}),"
                f"a={p.a:.3g},ymin={p.ymin:.3g},phase={p.phase:.3g}°)"
            )
        elif newparam.type == SignalTypes.STEP:
            if p is None:
                p = StepParam(_("New step function"))
            if edit and not p.edit(parent=parent):
                return None
            yarr = np.ones_like(xarr) * p.a1
            yarr[xarr > p.x0] = p.a2
            signal.set_xydata(xarr, yarr)
            signal.title = f"{prefix}(x0={p.x0:.3g},a1={p.a1:.3g},a2={p.a2:.3g})"
        return signal
    return None
