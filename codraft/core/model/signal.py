# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT Datasets
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

import guidata.dataset.dataitems as gdi
import guidata.dataset.datatypes as gdt
import numpy as np
from guidata.configtools import get_icon
from guidata.utils import update_dataset
from guiqwt.builder import make
from guiqwt.curve import CurveItem
from guiqwt.styles import update_style_attr

from codraft.config import Conf, _
from codraft.core.computation import fit
from codraft.core.model import base
from codraft.env import execenv


class SignalParam(gdt.DataSet, base.ObjectItf):
    """Signal dataset"""

    CONF_FMT = Conf.view.sig_format
    DEFAULT_FMT = ".3f"

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
    xlabel = gdi.StringItem(_("Title"))
    xunit = gdi.StringItem(_("Unit"))
    _e_unitsx = gdt.EndGroup(_("X-axis"))
    _unitsy = gdt.BeginGroup(_("Y-axis"))
    ylabel = gdi.StringItem(_("Title"))
    yunit = gdi.StringItem(_("Unit"))
    _e_unitsy = gdt.EndGroup(_("Y-axis"))
    _e_tabs_u = gdt.EndTabGroup("units")
    _e_unitsg = gdt.EndGroup(_("Titles and units"))

    _e_tabs = gdt.EndTabGroup("all")

    def copy_data_from(self, other, dtype=None):
        """Copy data from other dataset instance"""
        if dtype not in (None, float, complex, np.complex128):
            raise RuntimeError("Signal data only supports float64/complex128 dtype")
        self.metadata = other.metadata.copy()
        self.xydata = np.array(other.xydata, copy=True, dtype=dtype)

    def set_data_type(self, dtype):  # pylint: disable=unused-argument,no-self-use
        """Change data type"""
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

    def get_data(self, roi_index: int = None) -> np.ndarray:
        """
        Return original data (if ROI is not defined or `roi_index` is None),
        or ROI data (if both ROI and `roi_index` are defined).
        """
        if self.roi is None or roi_index is None:
            return self.x, self.y
        i1, i2 = self.roi[roi_index, :]
        return self.x[i1:i2], self.y[i1:i2]

    def make_item(self, update_from=None):
        """Make plot item from data"""
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

    def update_item(self, item: CurveItem):
        """Update plot item from data"""
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

    def roi_indexes_to_coords(self) -> np.ndarray:
        """Convert ROI indexes to coordinates"""
        return self.x[self.roi]

    def roi_coords_to_indexes(self, coords: list) -> np.ndarray:
        """Convert ROI coordinates to indexes"""
        indexes = np.array(coords, int)
        for row in range(indexes.shape[0]):
            for col in range(indexes.shape[1]):
                x0 = coords[row][col]
                indexes[row, col] = np.abs(self.x - x0).argmin()
        return indexes

    def get_roi_param(self, title, *defaults):
        """Return ROI parameters dataset"""
        imax = len(self.x) - 1
        i0, i1 = defaults

        class ROIParam(gdt.DataSet):
            """Signal ROI parameters"""

            col1 = gdi.IntItem(_("First point index"), default=i0, min=-1, max=imax)
            col2 = gdi.IntItem(_("Last point index"), default=i1, min=-1, max=imax)

        return ROIParam(title)

    @staticmethod
    def params_to_roidata(params: gdt.DataSetGroup) -> np.ndarray:
        """Convert list of dataset parameters to ROI data"""
        roilist = []
        for roiparam in params.datasets:
            roilist.append([roiparam.col1, roiparam.col2])
        if len(roilist) == 0:
            return None
        return np.array(roilist, int)

    def new_roi_item(self, fmt, lbl, editable):
        """Return a new ROI item from scratch"""
        coords = self.x.min(), self.x.max()
        return self.make_roi_item(
            lambda x, y, _title: make.range(x, y), coords, "ROI", fmt, lbl, editable
        )

    def iterate_roi_items(self, fmt: str, lbl: bool, editable: bool = True):
        """Make plot item representing a Region of Interest"""
        if self.roi is None:
            yield self.new_roi_item(fmt, lbl, editable)
        else:
            for index, coords in enumerate(self.roi_indexes_to_coords()):
                yield self.make_roi_item(
                    lambda x, y, _title: make.range(x, y),
                    coords,
                    f"ROI{index:02d}",
                    fmt,
                    lbl,
                    editable,
                )


def create_signal(
    title: str,
    x: np.ndarray = None,
    y: np.ndarray = None,
    dx: np.ndarray = None,
    dy: np.ndarray = None,
    metadata: dict = None,
    units: tuple = None,
    labels: tuple = None,
) -> SignalParam:
    """Create a new Signal object

    :param str title: signal title
    :param numpy.ndarray x: X data
    :param numpy.ndarray y: Y data
    :param numpy.ndarray dx: dX data (optional: error bars)
    :param numpy.ndarray dy: dY data (optional: error bars)
    :param dict metadata: signal metadata
    :param tuple units: X, Y units (tuple of strings)
    :param tuple labels: X, Y labels (tuple of strings)
    """
    assert isinstance(title, str)
    signal = SignalParam()
    signal.title = title
    signal.set_xydata(x, y, dx=dx, dy=dy)
    if units is not None:
        signal.xunit, signal.yunit = units
    if labels is not None:
        signal.xlabel, signal.ylabel = labels
    signal.metadata = {} if metadata is None else metadata
    return signal


class SignalTypes(base.Choices):
    """Signal types"""

    ZEROS = _("zeros")
    GAUSS = _("gaussian")
    LORENTZ = _("lorentzian")
    VOIGT = "Voigt"
    UNIFORMRANDOM = _("random (uniform law)")
    NORMALRANDOM = _("random (normal law)")


class GaussLorentzVoigtParam(gdt.DataSet):
    """Parameters for Gaussian and Lorentzian functions"""

    a = gdi.FloatItem("A", default=1.0)
    ymin = gdi.FloatItem("Ymin", default=0.0).set_pos(col=1)
    sigma = gdi.FloatItem("σ", default=1.0)
    mu = gdi.FloatItem("μ", default=0.0).set_pos(col=1)


class SignalParamNew(gdt.DataSet):
    """New signal dataset"""

    title = gdi.StringItem(_("Title"), default=_("Untitled"))
    xmin = gdi.FloatItem("Xmin", default=-10.0)
    xmax = gdi.FloatItem("Xmax", default=10.0)
    size = gdi.IntItem(
        _("Size"), help=_("Signal size (total number of points)"), min=1, default=500
    )
    type = gdi.ChoiceItem(_("Type"), SignalTypes.get_choices())


def new_signal_param(title=None, stype=None, xmin=None, xmax=None, size=None):
    """Create a new Signal dataset instance.

    :param str title: dataset title (default: None, uses default title)"""
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


def create_signal_from_param(newparam, addparam=None, edit=False, parent=None):
    """Create a new Signal object from a dialog box.

    :param SignalParamNew param: new signal parameters
    :param guidata.dataset.datatypes.DataSet addparam: additional parameters
    :param bool edit: Open a dialog box to edit parameters (default: False)
    :param QWidget parent: parent widget
    """
    global SIG_NB  # pylint: disable=global-statement
    if newparam is None:
        newparam = new_signal_param()
    incr_sig_nb = not newparam.title
    if incr_sig_nb:
        newparam.title = f"{newparam.title} {SIG_NB + 1:d}"
    if not edit or addparam is not None or newparam.edit(parent=parent):
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
                p = pclass(_("Signal") + " - " + newparam.type.value)
            if edit and not p.edit(parent=parent):
                return None
            rng = np.random.default_rng(p.seed)
            if newparam.type == SignalTypes.UNIFORMRANDOM:
                yarr = rng.random((newparam.size,)) * (p.vmax - p.vmin) + p.vmin
            elif newparam.type == SignalTypes.NORMALRANDOM:
                yarr = rng.normal(p.mu, p.sigma, size=(newparam.size,))
            else:
                raise NotImplementedError(f"New param type: {newparam.type.value}")
            signal.set_xydata(xarr, yarr)
        elif newparam.type == SignalTypes.GAUSS:
            if p is None:
                p = GaussLorentzVoigtParam(_("New gaussian function"))
            if edit and not p.edit(parent=parent):
                return None
            yarr = fit.GaussianModel.func(xarr, p.a, p.sigma, p.mu, p.ymin)
            signal.set_xydata(xarr, yarr)
        elif newparam.type == SignalTypes.LORENTZ:
            if p is None:
                p = GaussLorentzVoigtParam(_("New lorentzian function"))
            if edit and not p.edit(parent=parent):
                return None
            yarr = fit.LorentzianModel.func(xarr, p.a, p.sigma, p.mu, p.ymin)
            signal.set_xydata(xarr, yarr)
        elif newparam.type == SignalTypes.VOIGT:
            if p is None:
                p = GaussLorentzVoigtParam(_("New Voigt function"))
            if edit and not p.edit(parent=parent):
                return None
            yarr = fit.VoigtModel.func(xarr, p.a, p.sigma, p.mu, p.ymin)
            signal.set_xydata(xarr, yarr)
        return signal
    return None
