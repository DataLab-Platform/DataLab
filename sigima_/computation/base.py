# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Common computation objects (see parent package :mod:`sigima_.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the sigima_.param module.

from __future__ import annotations

from typing import TYPE_CHECKING

import guidata.dataset as gds
import numpy as np

from sigima_ import ImageObj, ResultProperties, SignalObj, create_signal
from sigima_.config import _, options

if TYPE_CHECKING:
    from typing import Callable


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


class FFTParam(gds.DataSet):
    """FFT parameters"""

    shift = gds.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift = options.fft_shift_enabled.get()


class SpectrumParam(gds.DataSet):
    """Spectrum parameters"""

    log = gds.BoolItem(_("Logarithmic scale"), default=False)


class ConstantParam(gds.DataSet):
    """Parameter used to set a constant value to used in operations"""

    value = gds.FloatItem(_("Constant value"))


# MARK: Helper functions for creating result objects -----------------------------------


def dst_1_to_1(
    src: SignalObj | ImageObj, name: str, suffix: str | None = None
) -> SignalObj | ImageObj:
    """Create a result object, as returned by the callback function of the
    :func:`cdl.gui.processor.base.BaseProcessor.compute_1_to_1` method.

    .. note::

        Data of the result object is copied from the source object (`src`).
        This initial data is usually replaced by the processing function, but it may
        also be used to initialize the result object as part of the processing function.

    Args:
        src: source signal or image object
        name: name of the function. If provided, the title of the result object
         will be `{name}({{0}})|{suffix})`, unless the name is a single
         character, in which case the title will be `{{0}}{name}{suffix}`
         where `name` is an operator and `suffix` is the other term of the operation.
        suffix: suffix to add to the title. Optional.

    Returns:
        Result signal or image object
    """
    if len(name) == 1:  # This is an operator
        title = f"{{0}}{name}"
    else:
        title = f"{name}({{0}})"
        if suffix:  # suffix may be None or an empty string
            title += "|"
    if suffix:  # suffix may be None or an empty string
        title += suffix
    dst = src.copy(title=title)
    if not options.keep_results.get():
        dst.delete_results()  # Remove any previous results
    return dst


def dst_n_to_1(
    src_list: list[SignalObj | ImageObj], name: str, suffix: str | None = None
) -> SignalObj | ImageObj:
    """Create a result object, as returned by the callback function of the
    :func:`cdl.gui.processor.base.BaseProcessor.compute_n_to_1` method

    .. note::

        Data of the result object is copied from the first source object
        (`src_list[0]`). This initial data is usually replaced by the processing
        function, but it may also be used to initialize the result object as part
        of the processing function.

    Args:
        src_list: list of input signal or image objects
        name: name of the processing function
        suffix: suffix to add to the title

    Returns:
        Result signal or image object
    """
    if not isinstance(src_list, list) or len(src_list) <= 1:
        raise ValueError("src_list must be a list of at least 2 objects")
    all_sigs = all(isinstance(obj, SignalObj) for obj in src_list)
    all_imgs = all(isinstance(obj, ImageObj) for obj in src_list)
    if not (all_sigs or all_imgs):
        raise ValueError("src_list must be a list of SignalObj or ImageObj objects")
    title = f"{name}(" + ", ".join(f"{{{i}}}" for i in range(len(src_list))) + ")"
    if suffix:  # suffix may be None or an empty string
        title += "|" + suffix
    if any(np.issubdtype(obj.data.dtype, complex) for obj in src_list):
        dst_dtype = complex
    else:
        dst_dtype = float
    dst = src_list[0].copy(title=title, dtype=dst_dtype)
    dst.roi = None
    if not options.keep_results.get():
        dst.delete_results()  # Remove any previous results
    for src_obj in src_list:
        if options.keep_results.get():
            dst.update_resultshapes_from(src_obj)
        if src_obj.roi is not None:
            if dst.roi is None:
                dst.roi = src_obj.roi.copy()
            else:
                roi = dst.roi
                roi.add_roi(src_obj.roi)
                dst.roi = roi
    return dst


def dst_2_to_1(
    src1: SignalObj | ImageObj,
    src2: SignalObj | ImageObj,
    name: str,
    suffix: str | None = None,
) -> SignalObj | ImageObj:
    """Create a result  object, as returned by the callback function of the
    :func:`cdl.gui.processor.base.BaseProcessor.compute_2_to_1` method

    .. note::

        Data of the result object is copied from the first source object (`src1`).
        This initial data is usually replaced by the processing function, but it may
        also be used to initialize the result object as part of the processing function.

    Args:
        src1: input signal or image object
        src2: input signal or image object
        name: name of the processing function

    Returns:
        Output signal or image object
    """
    if len(name) == 1:  # This is an operator
        title = f"{{0}}{name}{{1}}"
    else:
        title = f"{name}({{0}}, {{1}})"
    if suffix is not None:
        title += "|" + suffix
    dst = src1.copy(title=title)
    if not options.keep_results.get():
        dst.delete_results()  # Remove any previous results
    return dst


def new_signal_result(
    src: SignalObj | ImageObj,
    name: str,
    suffix: str | None = None,
    units: tuple[str, str] | None = None,
    labels: tuple[str, str] | None = None,
) -> SignalObj:
    """Create new signal object as a result of a `compute_1_to_1` function

    As opposed to the `dst_1_to_1` functions, this function creates a new signal object
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
    title = f"{name}({{0}})"
    dst = create_signal(title=title, units=units, labels=labels)
    if suffix is not None:
        dst.title += "|" + suffix
    try:
        source = src.get_metadata_option("source")
        dst.set_metadata_option("source", source)  # Keep track of the source
    except ValueError:
        # No source to keep track of
        pass
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
    roi_indices = list(obj.iterate_roi_indices())
    if roi_indices[0] is not None:
        roi_indices.insert(0, None)
    for i_roi in roi_indices:
        data_roi = obj.get_data(i_roi)
        val_roi = -1 if i_roi is None else i_roi
        res.append([val_roi] + [fn(data_roi) for fn in labeledfuncs.values()])
    return ResultProperties(title, np.array(res), list(labeledfuncs.keys()))
