"""
Computation (:mod:`sigima_.computation`)
----------------------------------------

This package contains the computation functions used by the DataLab project.
Those functions operate directly on DataLab objects (i.e. :class:`sigima_.obj.SignalObj`
and :class:`sigima_.obj.ImageObj`) and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`sigima_.computation` package is the main entry point for the DataLab
    computation functions when manipulating DataLab objects.
    See the :mod:`sigima_.algorithms` package for algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.Dataset` subclasses).
The computation functions takes a DataLab object (e.g. :class:`sigima_.obj.SignalObj`)
and a parameter object (e.g. :py:class:`sigima_.param.MovingAverageParam`) as input
and return a DataLab object as output (the result of the computation). The parameter
object is used to configure the computation function (e.g. the size of the moving
average window).

In DataLab overall architecture, the purpose of this package is to provide the
computation functions that are used by the :mod:`sigima_.core.gui.processor` module,
based on the algorithms defined in the :mod:`sigima_.algorithms` module and on the
data model defined in the :mod:`sigima_.obj` (or :mod:`sigima_.core.model`) module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`sigima_.computation.base`: Common processing features
- :mod:`sigima_.computation.signal`: Signal processing features
- :mod:`sigima_.computation.image`: Image processing features

Common processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.computation.base
   :members:

Signal processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sigima_.computation.signal
   :members:

Image processing features
^^^^^^^^^^^^^^^^^^^^^^^^^

Base image processing features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image
   :members:

Threshold features
~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.threshold
    :members:

Exposure correction features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.exposure
    :members:

Restoration features
~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.restoration
    :members:

Morphological features
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.morphology
    :members:

Edge detection features
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.edges

Detection features
~~~~~~~~~~~~~~~~~~

.. automodule:: sigima_.computation.image.detection
    :members:
"""

import dataclasses
import functools
import importlib
import inspect
import os.path as osp
import pkgutil
import sys
from types import ModuleType
from typing import Callable, Optional, TypeVar

if sys.version_info >= (3, 10):
    # Use ParamSpec from typing module in Python 3.10+
    from typing import ParamSpec
else:
    # Use ParamSpec from typing_extensions module in Python < 3.10
    from typing_extensions import ParamSpec

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
