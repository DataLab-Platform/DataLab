# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Configuration (:mod:`sigima_.config`)
-------------------------------------

The :mod:`sigima_.config` module provides a way to manage configuration options for the
`sigima_` library, as well as to handle translations and data paths, and other
configuration-related tasks.

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

The following table lists the available options:

.. autodata:: sigima_.config.OPTIONS_RST

.. note::

    The options are stored in an environment variable in JSON format, allowing for
    synchronization with external configurations or other processes that may need to
    read or modify the options. The environment variable name is defined by
    :attr:`sigima_.config.OptionsContainer.ENV_VAR`. This is especially useful for
    applications such as DataLab (where the `sigima_` library is used as a core
    component) as computations may be run in separate processes.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator

from guidata import configtools

# Translation and data path configuration
MOD_NAME = "sigima_"
_ = configtools.get_translation(MOD_NAME)
DATAPATH = configtools.get_module_data_path(MOD_NAME, "data")


class OptionField:
    """A configurable option field with get/set/context interface.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
    """

    def __init__(
        self,
        container: OptionsContainer,
        name: str,
        default: Any,
        description: str = "",
    ) -> None:
        self._container = container
        self.name = name
        self.check(default)  # Validate the default value
        self._value = default
        self.description = description

    def check(self, value: Any) -> None:  # pylint: disable=unused-argument
        """Check if the value is valid for this option.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not valid.
        """
        # This method can be overridden in subclasses for specific validation

    def get(self) -> Any:
        """Return the current value of the option.

        Returns:
            The current value of the option.
        """
        self._container.ensure_loaded_from_env()
        return self._value

    def set(self, value: Any, sync_env: bool = True) -> None:
        """Set the value of the option.

        Args:
            value: The new value to assign.
            sync_env: Whether to synchronize the environment variable.
        """
        self.check(value)  # Validate the new value
        self._value = value
        if sync_env:
            self._container.sync_env()

    def context(self, temp_value: Any) -> Generator[None, None, None]:
        """Temporarily override the option within a context.

        Args:
            temp_value: Temporary value to use within the context.

        Yields:
            None. Restores the original value upon exit.
        """

        @contextmanager
        def _ctx():
            old_value = self._value
            self.set(temp_value)
            try:
                yield
            finally:
                self.set(old_value)

        return _ctx()


class TypedOptionField(OptionField):
    """A configurable option field with type checking.

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
        expected_type: Expected type of the option value.
        description: Description of the option.
    """

    def __init__(
        self,
        container: OptionsContainer,
        name: str,
        default: Any,
        expected_type: type,
        description: str = "",
    ) -> None:
        self.expected_type = expected_type
        super().__init__(container, name, default, description)

    def check(self, value: Any) -> None:
        """Check if the value is of the expected type.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not of the expected type.
        """
        if not isinstance(value, self.expected_type):
            raise ValueError(
                f"Expected {self.expected_type.__name__}, got {type(value).__name__}"
            )


class ImageIOOptionField(OptionField):
    """A configurable option field for image I/O formats.

    .. note::

        This option is specifically for image I/O formats and expects a tuple of
        tuple of strings representing the formats, similar to the following:

        ... code-block:: python

            imageio_formats = (
                ("*.gel", "Opticks GEL"),
                ("*.spe", "Princeton Instruments SPE"),
                ("*.ndpi", "Hamamatsu Slide Scanner NDPI"),
                ("*.rec", "PCO Camera REC"),
            )

    Args:
        container: Options container instance to which this option belongs.
        name: Name of the option (used for introspection or errors).
        default: Default value of the option.
        description: Description of the option.
    """

    def check(self, value: Any) -> None:
        """Check if the value is a valid image I/O format.

        Args:
            value: The value to check.

        Raises:
            ValueError: If the value is not a valid image I/O format.
        """
        if not isinstance(value, tuple) or not all(
            isinstance(item, tuple) and len(item) == 2 for item in value
        ):
            raise ValueError(
                "Expected a tuple of tuples with two elements each "
                "(format, description)"
            )
        for item in value:
            if not isinstance(item[0], str) or not isinstance(item[1], str):
                raise ValueError(
                    "Each item must be a tuple of (format, description) as strings"
                )

    def set(self, value: Any, sync_env: bool = True) -> None:
        """Set the value of the option.

        Args:
            value: The new value to assign.
            sync_env: Whether to synchronize the environment variable.
        """
        super().set(value, sync_env)
        from sigima_.io.image import formats  # pylint: disable=import-outside-toplevel

        # Generate image I/O format classes based on the new value
        # This allows dynamic loading of formats based on the configuration
        formats.generate_imageio_format_classes()


IMAGEIO_FORMATS = (
    ("*.gel", "Opticks GEL"),
    ("*.spe", "Princeton Instruments SPE"),
    ("*.ndpi", "Hamamatsu Slide Scanner NDPI"),
    ("*.rec", "PCO Camera REC"),
)  # Default image I/O formats


class OptionsContainer:
    """Container for all configurable options in the `sigima_` library.

    Options are exposed as attributes with `.get()`, `.set()` and `.context()` methods.
    """

    #: Environment variable name for options in JSON format
    # This is used to synchronize options with external configurations or with
    # separate processes that may need to read or modify the options.
    ENV_VAR = "SIGIMA_OPTIONS_JSON"

    def __init__(self) -> None:
        self._loaded_from_env = False
        self.keep_results = TypedOptionField(
            self,
            "keep_results",
            default=True,
            expected_type=bool,
            description=_(
                "If True, computation functions will not delete previous results "
                "when creating new objects. This allows for retaining results across "
                "invocations."
            ),
        )
        self.fft_shift_enabled = TypedOptionField(
            self,
            "fft_shift_enabled",
            default=True,
            expected_type=bool,
            description=_(
                "If True, the FFT operations will apply a shift to the zero frequency "
                "component to the center of the spectrum. This is useful for "
                "visualizing frequency components in a more intuitive way."
            ),
        )
        self.imageio_formats = ImageIOOptionField(
            self,
            "imageio_formats",
            default=IMAGEIO_FORMATS,
            description=_(
                """List of supported image I/O formats. Each format is a tuple of
(file extension, description).

.. note::

    The `sigima` library supports any image format that can be read by the `imageio`
    library, provided that the associated plugin(s) are installed (see `imageio
    documentation <https://imageio.readthedocs.io/en/stable/formats/index.html>`_)
    and that the output NumPy array data type and shape are supported by `sigima`.

    To add a new file format, you may use the `imageio_formats` option to specify
    additional formats. Each entry should be a tuple of (file extension, description).

    Example:

    .. autodata:: sigima_.config.IMAGEIO_FORMATS
"""
            ),
        )
        # Add new options here

    def describe_all(self) -> None:
        """Print the name, value, and description of all options."""
        for name in vars(self):
            opt = getattr(self, name)
            if isinstance(opt, OptionField):
                print(f"{name} = {opt.get()}  # {opt.description}")

    def generate_rst_doc(self) -> str:
        """Generate reStructuredText documentation for all options.

        Returns:
            A string containing the reStructuredText documentation.
        """
        doc = """.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Default Value
      - Description
"""
        for name in vars(self):
            opt = getattr(self, name)
            if isinstance(opt, OptionField):
                doc += f"    * - {name}\n"
                doc += f"      - {opt.get()}\n"
                doc += f"      - {opt.description}\n"
        return doc

    def ensure_loaded_from_env(self) -> None:
        """Lazy-load from JSON env var on first access."""
        if self._loaded_from_env:
            return
        if self.ENV_VAR in os.environ:
            try:
                values = json.loads(os.environ[self.ENV_VAR])
                self.from_dict(values)
            except Exception as exc:  # pylint: disable=broad-except
                # If loading fails, we just log a warning and continue with defaults
                print(f"[sigima] Warning: failed to load options from env: {exc}")
        self._loaded_from_env = True

    def sync_env(self) -> None:
        """Update env var with current option values."""
        os.environ[self.ENV_VAR] = json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the current option values as a dictionary.

        Returns:
            A dictionary with option names as keys and their current values.
        """
        return {
            name: getattr(self, name).get()
            for name in vars(self)
            if isinstance(getattr(self, name), OptionField)
        }

    def from_dict(self, values: dict[str, Any]) -> None:
        """Set option values from a dictionary.

        Args:
            values: A dictionary with option names as keys and their new values.
        """
        for name, value in values.items():
            if hasattr(self, name):
                opt = getattr(self, name)
                if isinstance(opt, OptionField):
                    opt.set(value, sync_env=False)
        self.sync_env()


#: Global instance of the options container
options = OptionsContainer()

OPTIONS_RST = options.generate_rst_doc()
