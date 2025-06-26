# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration (:mod:`sigima_.config`)
-------------------------------------

The :mod:`sigima_.config` module provides a way to manage configuration options for the
`sigima_` library.

It allows users to set and retrieve options that affect the behavior of the library,
such as whether to keep results of computations or not. The options are handled
as in-memory key-value pairs, with default values provided.

It includes a context manager for temporary overrides of options.

.. autofunction:: set_options
.. autofunction:: get_option
.. autofunction:: temporary_options

The temporary options context manager allows you to temporarily change options within a
specific scope, ensuring that the original options are restored after the context is
exited. It is used as follows:

.. code-block:: python

    from sigima_.config import temporary_options

    with temporary_options(keep_results=False):
        # Code here will use the temporary option
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from typing import Any, Generator


@dataclasses.dataclass
class SigimaOptions:
    """Configuration options for the sigima_ library.

    Attributes:
        keep_results: If True, computation functions will not delete previous results
         when creating new objects. This allows for retaining results across
         invocations. By default, it is set to False, because it may be confusing
         to keep results if computations may affect those results.
    """

    keep_results: bool = True


# Global instance of `SigimaOptions` to hold current configuration
OPTIONS: SigimaOptions = SigimaOptions()


def set_options(keep_results: bool | None = None):
    """Set configuration options for the `sigima_` library.

    Args:
        keep_results: If True, computation functions will not delete previous results
         when creating new objects. This allows for retaining results across
         invocations. By default, it is set to False, because it may be confusing
         to keep results if computations may affect those results.
    """
    global OPTIONS
    if keep_results is not None:
        OPTIONS.keep_results = keep_results


def get_option(name: str) -> str | Any:
    """Get the value of a configuration option.

    Args:
        name: The name of the option to retrieve.

    Returns:
        The value of the requested option.

    Raises:
        KeyError: If the option name is unknown.
    """
    global OPTIONS
    if not hasattr(OPTIONS, name):
        raise KeyError(f"Unknown option: {name}")
    return getattr(OPTIONS, name)


@contextmanager
def temporary_options(keep_results: bool | None = None) -> Generator[None, None, None]:
    """Context manager for temporarily modifying options.

    This allows you to change options within a specific scope, and they will be
    restored to their original values after the context is exited.

    Args:
        keep_results: If True, the result object will not delete previous results.
         This allows for retaining results across invocations. By default, it is set
         to False, because it may be confusing to keep results if computations may
         affect those results.

    Yields:
        The context manager does not return any value, but modifies the
         global `OPTIONS` during its execution.

    Raises:
        KeyError: If the option name is unknown.
    """
    global OPTIONS
    old_options = OPTIONS
    try:
        new_options = dataclasses.replace(OPTIONS)
        if keep_results is not None:
            new_options.keep_results = keep_results
        OPTIONS = new_options
        yield
    finally:
        OPTIONS = old_options
