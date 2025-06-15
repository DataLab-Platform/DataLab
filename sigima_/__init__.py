# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
sigima_
======

Scientific computing engine for 1D signals and 2D images,
part of the DataLab open-source platform.
"""

# TODO: Annotations: currently, the `items_to_json` and `json_to_items` functions rely
#       on PlotPy's serialization functions, which are not compatible with `sigima`
#       annotations model. So, we have to implement the functions to convert `sigima`
#       annotations to PlotPy items and vice versa: `items_to_sigima_annotations` and
#       `sigima_annotations_to_items`.
#       Here are the locations where the functions will be used:
#       - `BaseObjPlotPyAdapter.add_annotations_from_items`
#       - `BaseDataPanel.__separate_view_finished`
#       - `RemoteClient.add_annotations_from_items`

# TODO: Rename "sigima_" (temporary name until the package is fully migrated)
#       to "sigima_" when the migration is complete.
# TODO: Move functions below to a separate module?
# TODO: Remove all dependencies on `cdl` package on all modules
# TODO: Remove all references for `Conf` (no need for it in `sigima_` package?)
# TODO: Add local translations for the `sigima_` package
# TODO: Add `pytest` infrastructure for the `sigima_` package
# TODO: Implement a I/O plugin system similar to the `cdl.plugins` module
# TODO: Implement a computation plugin system similar to the `cdl.plugins` module
# TODO: Handle the NumPy minimum requirement to v1.21 to use advanced type hints?
# TODO: Should we keep `PREFIX` attribute in `BaseObj`?

import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
from types import ModuleType
from typing import Callable, Optional, TypeVar

from typing_extensions import ParamSpec

# pylint:disable=unused-import
# flake8: noqa

from sigima_.io import (
    read_images,
    read_signals,
    write_image,
    write_signal,
    read_image,
    read_signal,
)
from sigima_.model import (
    CircularROI,
    ExponentialParam,
    Gauss2DParam,
    GaussLorentzVoigtParam,
    ImageDatatypes,
    ImageROI,
    ImageObj,
    ImageTypes,
    NewImageParam,
    NewSignalParam,
    NormalRandomParam,
    PeriodicParam,
    PolygonalROI,
    RectangularROI,
    ResultProperties,
    ResultShape,
    ROI1DParam,
    ROI2DParam,
    SegmentROI,
    ShapeTypes,
    SignalObj,
    SignalROI,
    SignalTypes,
    StepParam,
    TypeObj,
    TypeROI,
    UniformRandomParam,
    create_image,
    create_image_from_param,
    create_image_roi,
    create_signal,
    create_signal_from_param,
    create_signal_roi,
)

__version__ = "0.0.1"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/sigima_/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""

# Marker attribute used by @computation_function and introspection
COMPUTATION_METADATA_ATTR = "__computation_function_metadata"

P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass(frozen=True)
class ComputationMetadata:
    """Metadata for a computation function.

    Attributes:
        name: The name of the computation function.
        description: A description or docstring for the computation function.
    """

    name: str
    description: str


def computation_function(
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to mark a function as a DataLab computation function.

    Args:
        name: Optional name to override the function name.
        description: Optional docstring override or additional description.

    Returns:
        The wrapped function, tagged with a marker attribute.
    """

    def decorator(f: Callable[P, R]) -> Callable[P, R]:
        """Decorator to mark a function as a DataLab computation function.
        This decorator adds a marker attribute to the function, allowing
        it to be identified as a computation function.
        It also allows for optional name and description overrides.
        The function can be used as a decorator or as a standalone function.
        """

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return f(*args, **kwargs)

        metadata = ComputationMetadata(
            name=name or f.__name__, description=description or f.__doc__
        )
        setattr(wrapper, COMPUTATION_METADATA_ATTR, metadata)
        return wrapper

    return decorator


def is_computation_function(function: Callable) -> bool:
    """Check if a function is a DataLab computation function.

    Args:
        function: The function to check.

    Returns:
        True if the function is a DataLab computation function, False otherwise.
    """
    return getattr(function, COMPUTATION_METADATA_ATTR, None) is not None


def get_computation_metadata(function: Callable) -> ComputationMetadata:
    """Get the metadata of a DataLab computation function.

    Args:
        function: The function to get metadata from.

    Returns:
        Computation function metadata.

    Raises:
        ValueError: If the function is not a DataLab computation function.
    """
    metadata = getattr(function, COMPUTATION_METADATA_ATTR, None)
    if not isinstance(metadata, ComputationMetadata):
        raise ValueError(
            f"The function {function.__name__} is not a DataLab computation function."
        )
    return metadata


def find_computation_functions(
    module: ModuleType | None = None,
) -> list[tuple[str, Callable]]:
    """Find all computation functions in the `sigima_.computation` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima_.computation` package and its subpackages.

    Args:
        module: Optional module to search in. If None, the current module is used.

    Returns:
        A list of tuples, each containing the function name and the function object.
    """
    functions = []
    if module is None:
        path = [osp.dirname(__file__)]
    else:
        path = module.__path__
    for _, modname, _ in pkgutil.walk_packages(path=path, prefix=__name__ + "."):
        try:
            module = importlib.import_module(modname)
        except Exception:  # pylint: disable=broad-except
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if is_computation_function(obj):
                functions.append((modname, name, obj.__doc__))
    return functions
