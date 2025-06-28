# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Environment (:mod:`sigima_.env`)
--------------------------------

This module defines the execution environment for `sigima_` library, which is used
by DataLab. It provides a way to manage execution settings, such as unattended mode
and verbosity level, and allows for parsing command line arguments to set these
settings.

.. note::

    This module is intended for use in testing and development environments, and
    should not be used in production code. It is designed to facilitate testing
    and debugging by providing a controlled environment for running tests and
    executing code.
"""

from __future__ import annotations

import argparse
import enum
import os
import pprint
import sys
from contextlib import contextmanager
from typing import Any, Generator

from guidata.env import ExecEnv as GuiDataExecEnv

# We could import DEBUG from cdl.config, but is it really worth it?
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true")


class VerbosityLevels(enum.Enum):
    """Print verbosity levels (for testing purpose)"""

    QUIET = "quiet"
    NORMAL = "normal"
    DEBUG = "debug"


class SigimaExecEnv:
    """Object representing `sigima_` test environment"""

    UNATTENDED_ARG = "unattended"
    VERBOSE_ARG = "verbose"
    UNATTENDED_ENV = GuiDataExecEnv.UNATTENDED_ENV
    VERBOSE_ENV = GuiDataExecEnv.VERBOSE_ENV

    def __init__(self):
        # Check if "pytest" is in the command line arguments:
        if "pytest" not in sys.argv[0]:
            # Do not parse command line arguments when running tests with pytest
            # (otherwise, pytest arguments are parsed as DataLab arguments)
            self.parse_args()
        if self.unattended:  # Do not run this code in production
            # Check that calling `to_dict` do not raise any exception
            self.to_dict()

    def iterate_over_attrs_envvars(self) -> Generator[tuple[str, str], None, None]:
        """Iterate over CDL environment variables

        Yields:
            A tuple (attribute name, environment variable name)
        """
        for name in dir(self):
            if name.endswith("_ENV"):
                envvar: str = getattr(self, name)
                attrname = "_".join(name.split("_")[:-1]).lower()
                yield attrname, envvar

    def to_dict(self):
        """Return a dictionary representation of the object"""
        # The list of properties match the list of environment variable attribute names,
        # modulo the "_ENV" suffix:
        props = [attrname for attrname, _envvar in self.iterate_over_attrs_envvars()]

        # Check that all properties are defined in the class and that they are
        # really properties:
        for prop in props:
            assert hasattr(self, prop), (
                f"Property {prop} is not defined in class {self.__class__.__name__}"
            )
            assert isinstance(getattr(self.__class__, prop), property), (
                f"Attribute {prop} is not a property in class {self.__class__.__name__}"
            )

        # Return a dictionary with the properties as keys and their values as values:
        return {p: getattr(self, p) for p in props}

    def __str__(self):
        """Return a string representation of the object"""
        return pprint.pformat(self.to_dict())

    @staticmethod
    def __get_mode(env):
        """Get mode value"""
        env_val = os.environ.get(env)
        if env_val is None:
            return False
        return env_val.lower() in ("1", "true", "yes", "on", "enable", "enabled")

    @staticmethod
    def __set_mode(env, value):
        """Set mode value"""
        if env in os.environ:
            os.environ.pop(env)
        if value:
            os.environ[env] = "1"

    @property
    def unattended(self):
        """Get unattended value"""
        return self.__get_mode(self.UNATTENDED_ENV)

    @unattended.setter
    def unattended(self, value):
        """Set unattended value"""
        self.__set_mode(self.UNATTENDED_ENV, value)

    @property
    def verbose(self):
        """Get verbosity level"""
        env_val = os.environ.get(self.VERBOSE_ENV)
        if env_val in (None, ""):
            return VerbosityLevels.NORMAL.value
        return env_val.lower()

    @verbose.setter
    def verbose(self, value):
        """Set verbosity level"""
        os.environ[self.VERBOSE_ENV] = value

    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="Run `sigima_` tests")
        parser.add_argument(
            "--" + self.UNATTENDED_ARG,
            action="store_true",
            help="non-interactive mode",
            default=None,
        )
        parser.add_argument(
            "--" + self.VERBOSE_ARG,
            choices=[lvl.value for lvl in VerbosityLevels],
            required=False,
            default=None,
            help="verbosity level: for debugging/testing purpose",
        )
        args, _unknown = parser.parse_known_args()
        self.set_env_from_args(args)

    def set_env_from_args(self, args):
        """Set appropriate environment variables"""
        for argname in (self.UNATTENDED_ARG, self.VERBOSE_ARG):
            argvalue = getattr(args, argname)
            if argvalue is not None:
                setattr(self, argname, argvalue)

    def log(self, source: Any, *objects: Any) -> None:
        """Log text on screen

        Args:
            source: object from which the log is issued
            *objects: objects to log
        """
        if DEBUG or self.verbose == VerbosityLevels.DEBUG.value:
            print(str(source) + ":", *objects)
            #  TODO: [P4] Eventually, log in a file (optionally)

    def print(self, *objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        """Print in file, depending on verbosity level"""
        if self.verbose != VerbosityLevels.QUIET.value or DEBUG:
            print(*objects, sep=sep, end=end, file=file, flush=flush)

    def pprint(
        self,
        obj,
        stream=None,
        indent=1,
        width=80,
        depth=None,
        compact=False,
        sort_dicts=True,
    ):
        """Pretty-print in stream, depending on verbosity level"""
        if self.verbose != VerbosityLevels.QUIET.value or DEBUG:
            pprint.pprint(
                obj,
                stream=stream,
                indent=indent,
                width=width,
                depth=depth,
                compact=compact,
                sort_dicts=sort_dicts,
            )

    @contextmanager
    def context(
        self,
        unattended=None,
        verbose=None,
    ) -> Generator[None, None, None]:
        """Return a context manager that sets some execenv properties at enter,
        and restores them at exit. This is useful to run some code in a
        controlled environment, for example to accept dialogs in unattended
        mode, and restore the previous value at exit.

        Args:
            unattended: whether to run in unattended mode
            verbose: verbosity level

        .. note::
            If a passed value is None, the corresponding property is not changed.
        """
        old_values = self.to_dict()
        new_values = {"unattended": unattended, "verbose": verbose}
        for key, value in new_values.items():
            if value is not None:
                setattr(self, key, value)
        try:
            yield
        finally:
            for key, value in old_values.items():
                setattr(self, key, value)


execenv = SigimaExecEnv()
