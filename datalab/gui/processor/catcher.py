# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab warning and error catcher utility for processors
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import dataclasses
import traceback
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from datalab.env import execenv

if TYPE_CHECKING:
    from sigima.obj import ImageObj, ResultProperties, ResultShape, SignalObj


@dataclasses.dataclass
class CompOut:
    """Class for representing computation output

    Attributes:
        result: computation result
        error_msg: error message
        warning_msg: warning message
    """

    result: SignalObj | ImageObj | ResultShape | ResultProperties | None = None
    error_msg: str | None = None
    warning_msg: str | None = None


def wng_err_func(func: Callable, args: tuple[Any]) -> CompOut:
    """Wrapper function to catch errors and warnings during computation

    Args:
        func: function to call
        args: function arguments

    Returns:
        Computation output object containing the result, error message,
         or warning message.
    """
    with warnings.catch_warnings(record=True) as wngs:
        try:
            result = func(*args)
            if wngs:
                wng = wngs[-1]
                warning_msg = warnings.formatwarning(
                    message=wng.message,
                    category=wng.category,
                    filename=wng.filename,
                    lineno=wng.lineno,
                    line=wng.line,
                )
                return CompOut(result=result, warning_msg=warning_msg)
            return CompOut(result=result)
        except Exception:  # pylint: disable=broad-except
            if execenv.unattended and not execenv.catcher_test:
                #  In unattended mode (test cases), we want to raise the exception
                #  because test cases are supposed to work without any error. In real
                #  life, we want to avoid raising the exception because it would stop
                #  the application, and exceptions could be related to non-critical
                #  errors due to external libraries.
                #  When testing the catcher, on the other hand, we don't want to
                #  raise the exception because it would stop the unattended test
                #  execution.
                raise
            return CompOut(error_msg=traceback.format_exc())
