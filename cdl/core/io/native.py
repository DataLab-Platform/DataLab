# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Native I/O module (native HDF5/JSON formats)
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import importlib
from typing import Any, Callable

from guidata.io import HDF5Reader, HDF5Writer
from guidata.io.h5fmt import NoDefault

import cdl

DATALAB_VERSION_NAME = "DataLab_Version"
DATALAB_PACKAGE_NAME = "cdl"

H5_CALLABLE_PREFIX = "#callable#"


class NativeH5Writer(HDF5Writer):
    """DataLab signal/image objects HDF5 guidata Dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.h5[DATALAB_VERSION_NAME] = cdl.__version__

    @staticmethod
    def serialize_func_or_class(obj: Callable | type) -> str:
        """Serialize a function or a class object

        Args:
            obj: Object to serialize

        Returns:
            str: Serialized object
        """
        if not obj.__module__.startswith(DATALAB_PACKAGE_NAME):
            raise ValueError(
                f"Only {DATALAB_PACKAGE_NAME} functions and classes can be serialized"
            )
        val = f"{H5_CALLABLE_PREFIX}{obj.__module__}."
        if isinstance(obj, type):
            return val + obj.__name__
        return val + obj.__qualname__

    # Reimplement the write method to handle callable objects
    def write(self, val: Any, group_name: str | None = None) -> None:
        """
        Write a value depending on its type, optionally within a named group.

        Args:
            val: The value to be written.
            group_name: The name of the group. If provided, the group
             context will be used for writing the value.
        """
        try:
            super().write(val, group_name)
        except NotImplementedError:
            if callable(val):
                super().write_str(self.serialize_func_or_class(val))
                if group_name:
                    self.end(group_name)
            else:
                raise


class NativeH5Reader(HDF5Reader):
    """DataLab signal/image objects HDF5 guidata dataset Writer class

    Args:
        filename (str): HDF5 file name
    """

    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.version = self.h5[DATALAB_VERSION_NAME]

    @staticmethod
    def deserialize_func_or_class(obj: str) -> Callable | type:
        """Deserialize a function or a class object

        Args:
            obj: Serialized object

        Returns:
            Callable | type: Deserialized object
        """
        module_name, obj_name = obj[len(H5_CALLABLE_PREFIX) :].split(".")
        if not module_name.startswith(DATALAB_PACKAGE_NAME):
            raise ValueError(
                f"Only {DATALAB_PACKAGE_NAME} functions and classes can be deserialized"
            )
        module = importlib.import_module(module_name)
        return getattr(module, obj_name)

    # Reimplement the read method to handle callable objects
    def read(
        self,
        group_name: str | None = None,
        func: Callable[[], Any] | None = None,
        instance: Any | None = None,
        default: Any | NoDefault = NoDefault,
    ) -> Any:
        """
        Read a value from the current group or specified group_name.

        Args:
            group_name: The name of the group to read from. Defaults to None.
            func: The function to use for reading the value. Defaults to None.
            instance: An object that implements the DataSet-like `deserialize` method.
             Defaults to None.
            default: The default value to return if the value is not found.
             Defaults to `NoDefault` (no default value: raises an exception if the
             value is not found).

        Returns:
            The read value.
        """
        val = super().read(group_name)
        if isinstance(val, str) and val.startswith(H5_CALLABLE_PREFIX):
            return self.deserialize_func_or_class(val)
        return val
