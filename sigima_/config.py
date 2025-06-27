# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration (:mod:`sigima_.config`)
-------------------------------------

The :mod:`sigima_.config` module provides a way to manage configuration options for the
`sigima_` library.

It allows users to set and retrieve options that affect the behavior of the library,
such as whether to keep results of computations or not. The options are handled as
in-memory objects with default values provided, and can be temporarily overridden using
a context manager.

Typical usage:

.. code-block:: python

    from sigima_.config import options

    # Get an option
    value = options.keep_results.get(default=True)

    # Set an option
    options.keep_results.set(False)

    # Temporarily override an option
    with options.keep_results.context(True):
        ...
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator


class OptionField:
    """A configurable option field with get/set/context interface.

    Args:
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
    """

    def __init__(self, name: str, default: Any, description: str = "") -> None:
        self.name = name
        self._value = default
        self.description = description

    def get(self) -> Any:
        """Return the current value of the option.

        Returns:
            The current value of the option.
        """
        return self._value

    def set(self, value: Any) -> None:
        """Set the value of the option.

        Args:
            value: The new value to assign.
        """
        self._value = value

    @contextmanager
    def context(self, temp_value: Any) -> Generator[None, None, None]:
        """Temporarily override the option within a context.

        Args:
            temp_value: Temporary value to use within the context.

        Yields:
            None. Restores the original value upon exit.
        """
        old_value = self._value
        self._value = temp_value
        try:
            yield
        finally:
            self._value = old_value


class OptionsContainer:
    """Container for all configurable options in the `sigima_` library.

    Options are exposed as attributes with `.get()`, `.set()` and `.context()` methods.
    """

    def __init__(self) -> None:
        self.keep_results = OptionField(
            "keep_results",
            default=True,
            description=(
                "If True, computation functions will not delete previous results "
                "when creating new objects. This allows for retaining results across "
                "invocations."
            ),
        )
        # Add new options here

    def describe_all(self) -> None:
        """Print the name, value, and description of all options."""
        for name in vars(self):
            opt = getattr(self, name)
            if isinstance(opt, OptionField):
                print(f"{name} = {opt.get()}  # {opt.description}")

    def get_option(self, name: str) -> Any:
        """Get the value of an option by name.

        Args:
            name: The name of the option to retrieve.

        Returns:
            The value of the requested option.

        Raises:
            KeyError: If the option does not exist.
        """
        try:
            field = getattr(self, name)
        except AttributeError:
            raise KeyError(f"Unknown option: {name}")
        if not isinstance(field, OptionField):
            raise KeyError(f"Attribute '{name}' is not a configurable option.")
        return field.get()

    def set_option(self, name: str, value: Any) -> None:
        """Set the value of an option by name.

        Args:
            name: The name of the option to modify.
            value: The value to assign.

        Raises:
            KeyError: If the option does not exist.
        """
        try:
            field = getattr(self, name)
        except AttributeError:
            raise KeyError(f"Unknown option: {name}")
        if not isinstance(field, OptionField):
            raise KeyError(f"Attribute '{name}' is not a configurable option.")
        field.set(value)


#: Global instance of the options container
options = OptionsContainer()
