# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
.. Common computation objects (see parent package :mod:`cdl.core.computation`)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

# Note:
# ----
# All dataset classes must also be imported in the cdl.core.computation.param module.

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

import guidata.dataset as gds
import numpy as np
import scipy.signal as sps

from cdl.config import _
from cdl.core.model.signal import create_signal

if TYPE_CHECKING:
    from cdl.core.model.image import ImageObj
    from cdl.core.model.signal import SignalObj


class GaussianParam(gds.DataSet):
    """Gaussian filter parameters"""

    sigma = gds.FloatItem("σ", default=1.0)


class MovingAverageParam(gds.DataSet):
    """Moving average parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1)


class MovingMedianParam(gds.DataSet):
    """Moving median parameters"""

    n = gds.IntItem(_("Size of the moving window"), default=3, min=1, even=False)


class ThresholdParam(gds.DataSet):
    """Threshold parameters"""

    value = gds.FloatItem(_("Threshold"))


class ClipParam(gds.DataSet):
    """Data clipping parameters"""

    value = gds.FloatItem(_("Clipping value"))


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


class BaseFilterParam(gds.DataSet):
    # Dict of filter methods name that point to a tuple og their corresponding functions
    # to be used in the filter as well as a function to get the parameters of the filter
    FILTER_METHODS: dict[str, tuple[Callable, Callable]] = {}

    def method_choices(self, *args) -> list[tuple[str, str, Callable]]:
        """Return the filter function and the parameter function for the method"""
        choices = []
        for method in self.FILTER_METHODS:
            choices.append((method, method, None))
        return choices

    # Must be overwriten by the child class
    method = gds.ChoiceItem(
        _("Filter method"),
        choices=method_choices,
    )
    order = gds.IntItem(_("Filter order"), default=4, min=1)

    def get_filter(self) -> Callable:
        """Return the filter function corresponding to the method"""
        return self.FILTER_METHODS[self.method][0]  # type: ignore

    def get_param(self) -> tuple[float | str, ...]:
        return self.FILTER_METHODS[self.method][1](self)  # type: ignore

    type_: str = ""


class LowPassFilterParam(BaseFilterParam):
    f_cut = gds.FloatItem(_("Low cutoff frequency"), default=10)
    fe = gds.FloatItem(_("Sampling frequency"), default=100).set_prop(
        "display", hide=True
    )
    type_ = "low"

    def get_bessel_params(self) -> tuple[float | str, ...]:
        args: list[int | str] = [self.order]  # type: ignore
        args.extend((2 * self.f_cut / self.fe, self.type_))  # type: ignore
        print(args)
        return tuple(args)

    def get_butter_params(self) -> tuple[float | str, ...]:
        return self.get_bessel_params()

    def get_filter_params(self) -> tuple[float | str, float | str]:
        return self.get_filter()(*self.get_param())

    FILTER_METHODS = {
        "bessel": (sps.bessel, get_bessel_params),
        "butter": (sps.butter, get_butter_params),
        #     "cheby1": (sps.cheby1, sps.cheby1),
        #     "cheby2": (sps.cheby2, sps.cheby2),
    }

    def set_fe_from_xdata(self, x: np.ndarray) -> None:
        self.fe = (x.size - 1) / (x[-1] - x[0])


class HighPassFilterParam(LowPassFilterParam):
    type_ = "high"


class FFTParam(gds.DataSet):
    """FFT parameters"""

    shift = gds.BoolItem(_("Shift"), help=_("Shift zero frequency to center"))


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
