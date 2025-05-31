# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
sigima
======

Scientific computing engine for 1D signals and 2D images,
part of the DataLab open-source platform.
"""

# TODO: Move functions below to a separate module?
# TODO: Remove all dependencies on `cdl` package on all modules
# TODO: Reduce need for `Conf` to a minimum, then create a local `sigima.conf` module
# TODO: Add local translations for the `sigima` package
# TODO: Add `pytest` infrastructure for the `sigima` package
# TODO: Migrate the `cdl.core.model.image` module
# TODO: In `cdl` package: rename `BaseObj`, `ImageObj`, `SignalObj` to `PlotBaseObj`,
#       `PlotImageObj`, `PlotSignalObj`, then add the GUI-related methods and properties
# TODO: Migrate the I/O functions from `cdl` to `sigima.io` module
# TODO: Implement a I/O plugin system similar to the `cdl.plugins` module
# TODO: Implement a computation plugin system similar to the `cdl.plugins` module

import functools
import importlib
import inspect
import os.path as osp
import pkgutil
from types import ModuleType
from typing import Callable

__version__ = "0.0.1"
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/sigima/issues/new/choose"

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""

# Marker attribute used by @computation_function and introspection
COMPUTATION_MARKER = "_is_datalab_computation"


def computation_function(
    func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable:
    """Decorator to mark a function as a DataLab computation function.

    Args:
        name: Optional name to override the function name.
        description: Optional docstring override or additional description.

    Returns:
        The wrapped function, tagged with a marker attribute.
    """

    def decorator(f: Callable) -> Callable:
        """Decorator to mark a function as a DataLab computation function.
        This decorator adds a marker attribute to the function, allowing
        it to be identified as a computation function.
        It also allows for optional name and description overrides.
        The function can be used as a decorator or as a standalone function.
        """

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        setattr(wrapper, COMPUTATION_MARKER, True)
        # pylint: disable=protected-access
        wrapper._computation_name = name or f.__name__
        wrapper._computation_description = description or f.__doc__
        return wrapper

    return decorator if func is None else decorator(func)


def is_computation_function(function: Callable) -> bool:
    """Check if a function is a DataLab computation function.

    Args:
        function: The function to check.

    Returns:
        True if the function is a DataLab computation function, False otherwise.
    """
    return getattr(function, COMPUTATION_MARKER, False)


def find_computation_functions(
    module: ModuleType | None = None,
) -> list[tuple[str, Callable]]:
    """Find all computation functions in the `sigima.computation` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `sigima.computation` package and its subpackages.

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
