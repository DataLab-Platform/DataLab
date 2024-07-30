# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Common computation objects (see parent package :mod:`cdl.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.computation.param module.

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np

from cdl.config import _
from cdl.obj import ResultProperties, create_signal

if TYPE_CHECKING:
    from typing import Callable

    from cdl.obj import ImageObj, SignalObj


class ArithmeticParam(gds.DataSet):
    """Arithmetic parameters"""

    def get_operation(self) -> str:
        """Return the operation string"""
        o, a, b = self.operator, self.factor, self.constant
        b_added = False
        if a == 0.0:
            if o in ("+", "-"):
                txt = "obj3 = obj1"
            elif b == 0.0:
                txt = "obj3 = 0"
            else:
                txt = f"obj3 = {b}"
                b_added = True
        elif a == 1.0:
            txt = f"obj3 = obj1 {o} obj2"
        else:
            txt = f"obj3 = (obj1 {o} obj2) × {a}"
        if b != 0.0 and not b_added:
            txt += f" + {b}"
        return txt

    def update_operation(self, _item, _value):  # pylint: disable=unused-argument
        """Update the operation item"""
        self.operation = self.get_operation()

    operators = ("+", "-", "×", "/")
    operator = gds.ChoiceItem(_("Operator"), list(zip(operators, operators))).set_prop(
        "display", callback=update_operation
    )
    factor = (
        gds.FloatItem(_("Factor"), default=1.0)
        .set_pos(col=1)
        .set_prop("display", callback=update_operation)
    )
    constant = (
        gds.FloatItem(_("Constant"), default=0.0)
        .set_pos(col=1)
        .set_prop("display", callback=update_operation)
    )
    operation = gds.StringItem(_("Operation"), default="").set_prop(
        "display", active=False
    )
    restore_dtype = gds.BoolItem(
        _("Convert to `obj1` data type"), label=_("Result"), default=True
    )


class GaussianParam(gds.DataSet):
    """Gaussian filter parameters"""

    sigma = gds.FloatItem("σ", default=1.0)


HELP_MODE = _("""Mode of the filter:
- 'reflect': Reflect the data at the boundary
- 'constant': Pad with a constant value
- 'nearest': Pad with the nearest value
- 'mirror': Reflect the data at the boundary with the data itself
- 'wrap': Circular boundary""")


class MovingAverageParam(gds.DataSet):
    """Moving average parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1)
    modes = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(modes, modes)), default="reflect", help=HELP_MODE
    )


class MovingMedianParam(gds.DataSet):
    """Moving median parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1, even=False)
    modes = ("reflect", "constant", "nearest", "mirror", "wrap")
    mode = gds.ChoiceItem(
        _("Mode"), list(zip(modes, modes)), default="nearest", help=HELP_MODE
    )


class ClipParam(gds.DataSet):
    """Data clipping parameters"""

    lower = gds.FloatItem(_("Lower clipping value"), check=False)
    upper = gds.FloatItem(_("Upper clipping value"), check=False)


class NormalizeParam(gds.DataSet):
    """Normalize parameters"""

    methods = (
        ("maximum", _("Maximum")),
        ("amplitude", _("Amplitude")),
        ("area", _("Area")),
        ("energy", _("Energy")),
        ("rms", _("RMS")),
    )
    method = gds.ChoiceItem(_("Normalize with respect to"), methods)


class HistogramParam(gds.DataSet):
    """Histogram parameters"""

    def get_suffix(self, data: np.ndarray) -> str:
        """Return suffix for the histogram computation

        Args:
            data: data array
        """
        suffix = f"bins={self.bins:d}"
        if self.lower is not None:
            suffix += f", ymin={self.lower:.3f}"
        else:
            self.lower = np.min(data)
        if self.upper is not None:
            suffix += f", ymax={self.upper:.3f}"
        else:
            self.upper = np.max(data)

    bins = gds.IntItem(_("Number of bins"), default=256, min=1)
    lower = gds.FloatItem(_("Lower limit"), default=None, check=False)
    upper = gds.FloatItem(_("Upper limit"), default=None, check=False)


class ROIDataParam(gds.DataSet):
    """ROI Editor data"""

    roidata = gds.FloatArrayItem(
        _("ROI data"),
        help=_(
            "For convenience, this item accepts a 2D NumPy array, a list of list "
            "of numbers, or None. In the end, the data is converted to a 2D NumPy "
            "array of integers (if not None)."
        ),
    )
    singleobj = gds.BoolItem(
        _("Single object"),
        help=_("Whether to extract the ROI as a single object or not."),
    )

    @property
    def is_empty(self) -> bool:
        """Return True if there is no ROI"""
        return self.roidata is None or np.array(self.roidata).size == 0


class FFTParam(gds.DataSet):
    """FFT parameters"""

    shift = gds.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))


class SpectrumParam(gds.DataSet):
    """Spectrum parameters"""

    log = gds.BoolItem(_("Logarithmic scale"), default=False)


class ConstantParam(gds.DataSet):
    """Parameter used to set a constant value to used in operations"""

    value = gds.FloatItem(_("Constant value"))


# MARK: Helper functions for creating result objects -----------------------------------


def dst_11(
    src: SignalObj | ImageObj, name: str, suffix: str | None = None
) -> SignalObj | ImageObj:
    """Create a result object, as returned by the callback function of the
    :func:`cdl.core.gui.processor.base.BaseProcessor.compute_11` method

    Args:
        src: source signal or image object
        name: name of the function. If provided, the title of the result object
         will be `{name}({src.short_id})|{suffix})`, unless the name is a single
         character, in which case the title will be `{src.short_id}{name}{suffix}`
         where `name` is an operator and `suffix` is the other term of the operation.
        suffix: suffix to add to the title. Optional.

    Returns:
        Result signal or image object
    """
    if len(name) == 1:  # This is an operator
        title = f"{src.short_id}{name}"
    else:
        title = f"{name}({src.short_id})"
        if suffix is not None:
            title += "|"
    if suffix is not None:
        title += suffix
    return src.copy(title=title)


def dst_n1n(
    src1: SignalObj | ImageObj,
    src2: SignalObj | ImageObj,
    name: str,
    suffix: str | None = None,
) -> SignalObj | ImageObj:
    """Create a result  object, as returned by the callback function of the
    :func:`cdl.core.gui.processor.base.BaseProcessor.compute_n1n` method

    Args:
        src1: input signal or image object
        src2: input signal or image object
        name: name of the processing function

    Returns:
        Output signal or image object
    """
    if len(name) == 1:  # This is an operator
        title = f"{src1.short_id}{name}{src2.short_id}"
    else:
        title = f"{name}({src1.short_id}, {src2.short_id})"
    if suffix is not None:
        title += "|" + suffix
    return src1.copy(title=title)


def new_signal_result(
    src: SignalObj | ImageObj,
    name: str,
    suffix: str | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create new signal object as a result of a compute_11 function

    As opposed to the `dst_11` functions, this function creates a new signal object
    without copying the original object metadata, except for the "source" entry.

    Args:
        src: input signal or image object
        name: name of the processing function
        suffix: suffix to add to the title
        units: units of the output signal
        labels: labels of the output signal

    Returns:
        Output signal object
    """
    title = f"{name}({src.short_id})"
    dst = create_signal(title=title, units=units, labels=labels)
    if suffix is not None:
        dst.title += "|" + suffix
    if "source" in src.metadata:
        dst.metadata["source"] = src.metadata["source"]  # Keep track of the source
    return dst


def calc_resultproperties(
    title: str, obj: SignalObj | ImageObj, labeledfuncs: dict[str, Callable]
) -> ResultProperties:
    """Calculate result properties by executing a computation function
    on a signal/image object.

    Args:
        title: title of the result properties
        obj: signal or image object
        labeledfuncs: dictionary of labeled computation functions. The keys are
         the labels of the computation functions and the values are the functions
         themselves (each function must take a single argument - which is the data
         of the ROI or the whole signal/image - and return a float)

    Returns:
        Result properties object
    """
    if not all(isinstance(k, str) for k in labeledfuncs.keys()):
        raise ValueError("Keys of labeledfuncs must be strings")
    if not all(callable(v) for v in labeledfuncs.values()):
        raise ValueError("Values of labeledfuncs must be functions")

    res = []
    roi_nb = 0 if obj.roi is None else obj.roi.shape[0]
    for i_roi in [None] + list(range(roi_nb)):
        data_roi = obj.get_data(i_roi)
        val_roi = -1 if i_roi is None else i_roi
        res.append([val_roi] + [fn(data_roi) for fn in labeledfuncs.values()])
    return ResultProperties(title, np.array(res), list(labeledfuncs.keys()))
