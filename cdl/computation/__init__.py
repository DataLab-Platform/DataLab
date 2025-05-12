"""
Computation (:mod:`cdl.computation`)
-----------------------------------------

This package contains the computation functions used by the DataLab project.
Those functions operate directly on DataLab objects (i.e. :class:`cdl.obj.SignalObj`
and :class:`cdl.obj.ImageObj`) and are designed to be used in the DataLab pipeline,
but can be used independently as well.

.. seealso::

    The :mod:`cdl.computation` package is the main entry point for the DataLab
    computation functions when manipulating DataLab objects.
    See the :mod:`cdl.algorithms` package for algorithms that operate directly on
    NumPy arrays.

Each computation module defines a set of computation objects, that is, functions
that implement processing features and classes that implement the corresponding
parameters (in the form of :py:class:`guidata.dataset.datatypes.Dataset` subclasses).
The computation functions takes a DataLab object (e.g. :class:`cdl.obj.SignalObj`)
and a parameter object (e.g. :py:class:`cdl.param.MovingAverageParam`) as input
and return a DataLab object as output (the result of the computation). The parameter
object is used to configure the computation function (e.g. the size of the moving
average window).

In DataLab overall architecture, the purpose of this package is to provide the
computation functions that are used by the :mod:`cdl.core.gui.processor` module,
based on the algorithms defined in the :mod:`cdl.algorithms` module and on the
data model defined in the :mod:`cdl.obj` (or :mod:`cdl.core.model`) module.

The computation modules are organized in subpackages according to their purpose.
The following subpackages are available:

- :mod:`cdl.computation.base`: Common processing features
- :mod:`cdl.computation.signal`: Signal processing features
- :mod:`cdl.computation.image`: Image processing features

Common processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.computation.base
   :members:

Signal processing features
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.computation.signal
   :members:

Image processing features
^^^^^^^^^^^^^^^^^^^^^^^^^

Base image processing features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image
   :members:

Threshold features
~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.threshold
    :members:

Exposure correction features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.exposure
    :members:

Restoration features
~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.restoration
    :members:

Morphological features
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.morphology
    :members:

Edge detection features
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.edges

Detection features
~~~~~~~~~~~~~~~~~~

.. automodule:: cdl.computation.image.detection
    :members:
"""

from __future__ import annotations

import functools
import importlib
import inspect
import os.path as osp
import pkgutil
from types import ModuleType
from typing import Callable

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
        wrapper._computation_name = name or f.__name__
        wrapper._computation_description = description or f.__doc__
        return wrapper

    return decorator if func is None else decorator(func)


def find_computation_functions(
    module: ModuleType | None = None,
) -> list[tuple[str, Callable]]:
    """Find all computation functions in the `cdl.computation` package.

    This function uses introspection to locate all functions decorated with
    `@computation_function` in the `cdl.computation` package and its subpackages.

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
        except Exception:
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if getattr(obj, COMPUTATION_MARKER, False):
                functions.append((modname, name, obj.__doc__))
    return functions
