# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Signal object and related classes
---------------------------------

"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import Type

import guidata.dataset as gds
import numpy as np
import scipy.signal as sps

from cdl.config import _
from sigima_.algorithms.signal import GaussianModel, LorentzianModel, VoigtModel
from sigima_.model import base


class ROI1DParam(base.BaseROIParam["SignalObj", "SegmentROI"]):
    """Signal ROI parameters"""

    # Note: in this class, the ROI parameters are stored as X coordinates

    xmin = gds.FloatItem(_("First point coordinate"))
    xmax = gds.FloatItem(_("Last point coordinate"))

    def to_single_roi(self, obj: SignalObj, title: str = "") -> SegmentROI:
        """Convert parameters to single ROI

        Args:
            obj: signal object
            title: ROI title

        Returns:
            Single ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        return SegmentROI([self.xmin, self.xmax], False, title=title)

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        assert isinstance(self.xmin, float) and isinstance(self.xmax, float)
        imin, imax = np.searchsorted(obj.x, [self.xmin, self.xmax])
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])


class SegmentROI(base.BaseSingleROI["SignalObj", ROI1DParam]):
    """Segment ROI

    Args:
        coords: ROI coordinates (xmin, xmax)
        title: ROI title
    """

    # Note: in this class, the ROI parameters are stored as X indices

    def check_coords(self) -> None:
        """Check if coords are valid

        Raises:
            ValueError: invalid coords
        """
        if len(self.coords) != 2:
            raise ValueError("Invalid ROI segment coords (2 values expected)")
        if self.coords[0] >= self.coords[1]:
            raise ValueError("Invalid ROI segment coords (xmin >= xmax)")

    def get_data(self, obj: SignalObj) -> np.ndarray:
        """Get signal data in ROI

        Args:
            obj: signal object

        Returns:
            Data in ROI
        """
        imin, imax = self.get_indices_coords(obj)
        return np.array([obj.x[imin:imax], obj.y[imin:imax]])

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """

        mask = np.ones_like(obj.xydata, dtype=bool)
        imin, imax = self.get_indices_coords(obj)
        mask[:, imin:imax] = False
        return mask

    # pylint: disable=unused-argument
    def to_param(self, obj: SignalObj, title: str | None = None) -> ROI1DParam:
        """Convert ROI to parameters

        Args:
            obj: object (signal), for physical-indices coordinates conversion
            title: ROI title
        """
        title = title or self.title
        param = ROI1DParam(title)
        param.xmin, param.xmax = self.get_physical_coords(obj)
        return param


class SignalROI(base.BaseROI["SignalObj", SegmentROI, ROI1DParam]):
    """Signal Regions of Interest

    Args:
        singleobj: if True, when extracting data defined by ROIs, only one object
         is created (default to True). If False, one object is created per single ROI.
         If None, the value is get from the user configuration
        inverse: if True, ROI is outside the region
    """

    PREFIX = "s"

    @staticmethod
    def get_compatible_single_roi_classes() -> list[Type[SegmentROI]]:
        """Return compatible single ROI classes"""
        return [SegmentROI]

    def to_mask(self, obj: SignalObj) -> np.ndarray:
        """Create mask from ROI

        Args:
            obj: signal object

        Returns:
            Mask (boolean array where True values are inside the ROI)
        """
        mask = np.ones_like(obj.xydata, dtype=bool)
        for roi in self.single_rois:
            mask &= roi.to_mask(obj)
        return mask


def create_signal_roi(
    coords: np.ndarray | list[float] | list[list[float]],
    indices: bool = False,
    singleobj: bool | None = None,
    inverse: bool = False,
    title: str = "",
) -> SignalROI:
    """Create Signal Regions of Interest (ROI) object.
    More ROIs can be added to the object after creation, using the `add_roi` method.

    Args:
        coords: single ROI coordinates `[xmin, xmax]`, or multiple ROIs coordinates
         `[[xmin1, xmax1], [xmin2, xmax2], ...]` (lists or NumPy arrays)
        indices: if True, coordinates are indices, if False, they are physical values
         (default to False for signals)
        singleobj: if True, when extracting data defined by ROIs, only one object
         is created (default to True). If False, one object is created per single ROI.
         If None, the value is get from the user configuration
        inverse: if True, ROI is outside the region
        title: title

    Returns:
        Regions of Interest (ROI) object

    Raises:
        ValueError: if the number of coordinates is not even
    """
    coords = np.array(coords, float)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    roi = SignalROI(singleobj, inverse)
    for row in coords:
        roi.add_roi(SegmentROI(row, indices=indices, title=title))
    return roi


class SignalObj(gds.DataSet, base.BaseObj[SignalROI]):
    """Signal object"""

    PREFIX = "s"
    VALID_DTYPES = (np.float32, np.float64, np.complex128)

    _tabs = gds.BeginTabGroup("all")

    _datag = gds.BeginGroup(_("Data and metadata"))
    title = gds.StringItem(_("Signal title"), default=_("Untitled"))
    xydata = gds.FloatArrayItem(_("Data"), transpose=True, minmax="rows")
    metadata = gds.DictItem(_("Metadata"), default={})  # type: ignore[assignment]
    annotations = gds.StringItem(_("Annotations"), default="").set_prop(
        "display",
        hide=True,
    )  # Annotations as a serialized JSON string  # type: ignore[assignment]
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

    @staticmethod
    def get_roi_class() -> Type[SignalROI]:
        """Return ROI class"""
        return SignalROI

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
        obj.annotations = self.annotations
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
        x: np.ndarray | list | None,
        y: np.ndarray | list | None,
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
        if x is None and y is None:
            # Using empty arrays (this allows initialization of the object without data)
            x = np.array([], dtype=np.float64)
            y = np.array([], dtype=np.float64)
        if x is None and y is not None:
            # If x is None, we create a default x array based on the length of y
            assert isinstance(y, (list, np.ndarray))
            x = np.arange(len(y), dtype=np.float64)
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
            assert x is not None and y is not None
            self.xydata = np.vstack((x, y, dx, dy))

    def __get_x(self) -> np.ndarray | None:
        """Get x data"""
        if self.xydata is not None:
            return self.xydata[0]
        return None

    def __set_x(self, data) -> None:
        """Set x data"""
        assert isinstance(self.xydata, np.ndarray)
        self.xydata[0] = np.array(data)

    def __get_y(self) -> np.ndarray | None:
        """Get y data"""
        if self.xydata is not None:
            return self.xydata[1]
        return None

    def __set_y(self, data) -> None:
        """Set y data"""
        assert isinstance(self.xydata, np.ndarray)
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
            roi_index: ROI index

        Returns:
            Data
        """
        if self.roi is None or roi_index is None:
            assert isinstance(self.xydata, np.ndarray)
            return self.xydata
        single_roi = self.roi.get_single_roi(roi_index)
        return single_roi.get_data(self)

    def physical_to_indices(self, coords: list[float]) -> list[int]:
        """Convert coordinates from physical (real world) to indices (pixel)

        Args:
            coords: coordinates

        Returns:
            Indices
        """
        assert isinstance(self.x, np.ndarray)
        return [int(np.abs(self.x - x).argmin()) for x in coords]

    def indices_to_physical(self, indices: list[int]) -> list[float]:
        """Convert coordinates from indices to physical (real world)

        Args:
            indices: indices

        Returns:
            Coordinates
        """
        # We take the real part of the x data to avoid `ComplexWarning` warnings
        # when creating and manipulating the `XRangeSelection` shape (`plotpy`)
        return self.x.real[indices].tolist()


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


DEFAULT_TITLE = _("Untitled signal")


class NewSignalParam(gds.DataSet):
    """New signal dataset"""

    hide_signal_type = False

    title = gds.StringItem(_("Title"), default=DEFAULT_TITLE)
    xmin = gds.FloatItem("Xmin", default=-10.0)
    xmax = gds.FloatItem("Xmax", default=10.0)
    size = gds.IntItem(
        _("Size"), help=_("Signal size (total number of points)"), min=1, default=500
    )
    stype = gds.ChoiceItem(_("Type"), SignalTypes.get_choices()).set_prop(
        "display", hide=gds.GetAttrProp("hide_signal_type")
    )


def triangle_func(xarr: np.ndarray) -> np.ndarray:
    """Triangle function

    Args:
        xarr: x data
    """
    # ignore warning, as type hint is not handled properly in upstream library
    return sps.sawtooth(xarr, width=0.5)  # type: ignore[no-untyped-def]


SIG_NB = 0


def get_next_signal_number() -> int:
    """Get the next signal number.

    This function is used to keep track of the number of signals created.
    It is typically used to generate unique titles for new signals.

    Returns:
        int: new signal number
    """
    global SIG_NB  # pylint: disable=global-statement
    SIG_NB += 1
    return SIG_NB


def create_signal_from_param(
    base_param: NewSignalParam,
    extra_param: gds.DataSet | None = None,
) -> SignalObj:
    """Create a new Signal object from parameters.

    Args:
        base_param: new signal parameters
        extra_param: additional parameters (optional)

    Returns:
        Signal object

    Raises:
        ValueError: if `extra_param` is required but not provided
        NotImplementedError: if the signal type is not supported
    """
    incr_sig_nb = not base_param.title
    prefix = base_param.stype.name.lower()
    if incr_sig_nb:
        base_param.title = f"{base_param.title} {get_next_signal_number():d}"

    ep = extra_param
    signal = create_signal(base_param.title)
    xarr = np.linspace(base_param.xmin, base_param.xmax, base_param.size)
    title = base_param.title or DEFAULT_TITLE

    if base_param.stype == SignalTypes.ZEROS:
        yarr = np.zeros(base_param.size)

    elif base_param.stype == SignalTypes.UNIFORMRANDOM:
        if ep is None:
            raise ValueError("extra_param (UniformRandomParam) required.")
        assert isinstance(ep, base.UniformRandomParam)
        rng = np.random.default_rng(ep.seed)
        yarr = rng.random((base_param.size,)) * (ep.vmax - ep.vmin) + ep.vmin
        title = f"{prefix}(vmin={ep.vmin:.3g},vmax={ep.vmax:.3g})"

    elif base_param.stype == SignalTypes.NORMALRANDOM:
        if ep is None:
            raise ValueError("extra_param (NormalRandomParam) required.")
        assert isinstance(ep, base.NormalRandomParam)
        rng = np.random.default_rng(ep.seed)
        yarr = rng.normal(ep.mu, ep.sigma, size=(base_param.size,))
        title = f"{prefix}(mu={ep.mu:.3g},sigma={ep.sigma:.3g})"

    elif base_param.stype in (
        SignalTypes.GAUSS,
        SignalTypes.LORENTZ,
        SignalTypes.VOIGT,
    ):
        if ep is None:
            raise ValueError("extra_param (GaussLorentzVoigtParam) required.")
        assert isinstance(ep, GaussLorentzVoigtParam)
        func = {
            SignalTypes.GAUSS: GaussianModel.func,
            SignalTypes.LORENTZ: LorentzianModel.func,
            SignalTypes.VOIGT: VoigtModel.func,
        }[base_param.stype]
        yarr = func(xarr, ep.a, ep.sigma, ep.mu, ep.ymin)
        title = (
            f"{prefix}(a={ep.a:.3g},sigma={ep.sigma:.3g},"
            f"mu={ep.mu:.3g},ymin={ep.ymin:.3g})"
        )

    elif base_param.stype in (
        SignalTypes.SINUS,
        SignalTypes.COSINUS,
        SignalTypes.SAWTOOTH,
        SignalTypes.TRIANGLE,
        SignalTypes.SQUARE,
        SignalTypes.SINC,
    ):
        if ep is None:
            raise ValueError("extra_param (PeriodicParam) required.")
        assert isinstance(ep, PeriodicParam)
        func = {
            SignalTypes.SINUS: np.sin,
            SignalTypes.COSINUS: np.cos,
            SignalTypes.SAWTOOTH: sps.sawtooth,
            SignalTypes.TRIANGLE: triangle_func,
            SignalTypes.SQUARE: sps.square,
            SignalTypes.SINC: np.sinc,
        }[base_param.stype]
        freq = ep.get_frequency_in_hz()
        yarr = ep.a * func(2 * np.pi * freq * xarr + np.deg2rad(ep.phase)) + ep.ymin
        title = (
            f"{prefix}(f={ep.freq:.3g} {ep.freq_unit.value}),"
            f"a={ep.a:.3g},ymin={ep.ymin:.3g},phase={ep.phase:.3g}°)"
        )

    elif base_param.stype == SignalTypes.STEP:
        if ep is None:
            raise ValueError("extra_param (StepParam) required.")
        assert isinstance(ep, StepParam)
        yarr = np.ones_like(xarr) * ep.a1
        yarr[xarr > ep.x0] = ep.a2
        title = f"{prefix}(a1={ep.a1:.3g},a2={ep.a2:.3g},x0={ep.x0:.3g})"

    elif base_param.stype == SignalTypes.EXPONENTIAL:
        if ep is None:
            raise ValueError("extra_param (ExponentialParam) required.")
        assert isinstance(ep, ExponentialParam)
        yarr = ep.a * np.exp(ep.exponent * xarr) + ep.offset
        title = f"{prefix}(a={ep.a:.3g},k={ep.exponent:.3g},y0={ep.offset:.3g})"

    elif base_param.stype == SignalTypes.PULSE:
        if ep is None:
            raise ValueError("extra_param (PulseParam) required.")
        assert isinstance(ep, PulseParam)
        yarr = np.full_like(xarr, ep.offset)
        yarr[(xarr >= ep.start) & (xarr <= ep.stop)] += ep.amp
        title = (
            f"{prefix}(start={ep.start:.3g},stop={ep.stop:.3g},offset={ep.offset:.3g})"
        )

    elif base_param.stype == SignalTypes.POLYNOMIAL:
        if ep is None:
            raise ValueError("extra_param (PolyParam) required.")
        assert isinstance(ep, PolyParam)
        yarr = np.polyval([ep.a5, ep.a4, ep.a3, ep.a2, ep.a1, ep.a0], xarr)
        title = (
            f"{prefix}(a0={ep.a0:.3g},a1={ep.a1:.3g},a2={ep.a2:.3g},"
            f"a3={ep.a3:.3g},a4={ep.a4:.3g},a5={ep.a5:.3g})"
        )

    else:
        raise NotImplementedError(
            f"Signal type '{base_param.stype}' is not implemented."
        )

    signal.set_xydata(xarr, yarr)
    if signal.title == DEFAULT_TITLE:
        signal.title = title
    return signal
