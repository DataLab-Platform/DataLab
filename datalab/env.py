# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab environmnent utilities
"""

from __future__ import annotations

import argparse
import enum
import os
import platform
import pprint
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Generator

from guidata.env import ExecEnv as GuiDataExecEnv

# We could import DEBUG from datalab.config, but is it really worth it?
DEBUG = os.environ.get("DEBUG", "").lower() in ("1", "true")


class VerbosityLevels(enum.Enum):
    """Print verbosity levels (for testing purpose)"""

    QUIET = "quiet"
    NORMAL = "normal"
    DEBUG = "debug"


# TODO: [P3] Rewrite this class so that options are automatically associated with
#       environment variables and command line arguments.
#
#       Use the unit test "datalab\tests\backbone\execenv_unit.py" to check that
#       everything still works as expected.
#
#       This could be done using objects deriving from something like this (and
#       implementing integer, boolean, string, choices):
#
#        class EnvVar:
#            """Descriptor for handling attributes
#            associated with environment variables"""
#
#            def __init__(
#                self, name: str, default: Optional[str] = None,
#                argname: Optional[str] = None
#            ):
#                """
#                Initialize the EnvVar descriptor.
#
#                Args:
#                    name: The name of the associated environment variable.
#                    default: The default value for the attribute.
#                    argname: The name of the command-line argument (optional).
#                """
#                self.name = name
#                self.default = default
#                self.argname = argname
#
#            def __get__(self, instance: Optional[object],
#                        owner: type) -> Optional[str]:
#                """
#                Get the value of the attribute.
#
#                Args:
#                    instance: The instance of the class.
#                    owner: The class that owns the attribute.
#
#                Returns:
#                    The value of the attribute.
#                """
#                if instance is None:
#                    return self
#                value = os.environ.get(self.name)
#                if value is None:
#                    return self.default
#                return self._envvar_to_value(value)
#
#            def __set__(self, instance: object, value: Optional[str]) -> None:
#                """
#                Set the value of the attribute.
#
#                Args:
#                    instance: The instance of the class.
#                    value: The value to be set.
#                """
#                if value is not None:
#                    os.environ[self.name] = self._value_to_envvar(value)
#                elif self.name in os.environ:
#                    os.environ.pop(self.name)
#
#            def __delete__(self, instance: object) -> None:
#                """
#                Delete the attribute and remove the associated environment variable.
#
#                Args:
#                    instance: The instance of the class.
#                """
#                os.environ.pop(self.name, None)
#
#            def add_argument(self, parser: argparse.ArgumentParser) -> None:
#                """
#                Add the command-line argument to the given ArgumentParser.
#
#                Args:
#                    parser: The ArgumentParser to add the argument to.
#                """
#                parser.add_argument(
#                    f"--{self.argname}",
#                    default=self.default,
#                    help=f"{self.argname} (environment variable: {self.name})",
#                )
#
#            def _envvar_to_value(self, value: str) -> str:
#                """
#                Convert the environment variable value to the attribute value.
#                  This method must be implemented by subclasses.
#                """
#               return value
#
#            def _value_to_envvar(self, value: str) -> str:
#                """
#                Convert the attribute value to the environment variable value.
#                  This method must be implemented by subclasses.
#                """
#                return value
#


class CDLExecEnv:
    """Object representing DataLab test environment"""

    UNATTENDED_ARG = "unattended"
    ACCEPT_DIALOGS_ARG = "accept_dialogs"
    VERBOSE_ARG = "verbose"
    SCREENSHOT_ARG = "screenshot"
    DELAY_ARG = "delay"
    XMLRPCPORT_ARG = "xmlrpcport"
    DO_NOT_QUIT_ENV = "CDL_DO_NOT_QUIT"
    UNATTENDED_ENV = GuiDataExecEnv.UNATTENDED_ENV
    ACCEPT_DIALOGS_ENV = GuiDataExecEnv.ACCEPT_DIALOGS_ENV
    VERBOSE_ENV = GuiDataExecEnv.VERBOSE_ENV
    SCREENSHOT_ENV = GuiDataExecEnv.SCREENSHOT_ENV
    DELAY_ENV = GuiDataExecEnv.DELAY_ENV
    XMLRPCPORT_ENV = "CDL_XMLRPCPORT"
    CATCHER_TEST_ENV = "CDL_CATCHER_TEST"

    def __init__(self):
        self.h5files = None
        self.h5browser_file = None
        self.demo_mode = False
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

        # Add complementary properties:
        props += [
            "h5files",
            "h5browser_file",
            "demo_mode",
        ]

        # Return a dictionary with the properties as keys and their values as values:
        return {p: getattr(self, p) for p in props}

    def __str__(self):
        """Return a string representation of the object"""
        return pprint.pformat(self.to_dict())

    def enable_demo_mode(self, delay: int):
        """Enable demo mode

        Args:
            delay: Delay (ms) before quitting application in unattended mode
        """
        self.demo_mode = True
        self.unattended = True
        self.delay = delay

    def disable_demo_mode(self):
        """Disable demo mode"""
        self.demo_mode = False
        self.unattended = False
        self.delay = 0

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
    def do_not_quit(self):
        """Keep QApplication running (and widgets opened) after test execution,
        even in unattended mode (e.g. useful for testing the remote client API:
        we need to run DataLab in unattended mode [to avoid any user interaction
        during the test] but we also need to keep the QApplication running to
        be able to send commands to the remote client API).
        """
        return self.__get_mode(self.DO_NOT_QUIT_ENV)

    @do_not_quit.setter
    def do_not_quit(self, value):
        """Set do_not_quit value"""
        self.__set_mode(self.DO_NOT_QUIT_ENV, value)

    @property
    def unattended(self):
        """Get unattended value"""
        return self.__get_mode(self.UNATTENDED_ENV)

    @unattended.setter
    def unattended(self, value):
        """Set unattended value"""
        self.__set_mode(self.UNATTENDED_ENV, value)

    @property
    def accept_dialogs(self):
        """Whether to accept dialogs in unattended mode"""
        return self.__get_mode(self.ACCEPT_DIALOGS_ENV)

    @accept_dialogs.setter
    def accept_dialogs(self, value):
        """Set whether to accept dialogs in unattended mode"""
        self.__set_mode(self.ACCEPT_DIALOGS_ENV, value)

    @property
    def catcher_test(self):
        """Get catcher_test value"""
        return self.__get_mode(self.CATCHER_TEST_ENV)

    @catcher_test.setter
    def catcher_test(self, value):
        """Set catcher_test value"""
        self.__set_mode(self.CATCHER_TEST_ENV, value)

    @property
    def screenshot(self):
        """Get screenshot value"""
        return self.__get_mode(self.SCREENSHOT_ENV)

    @screenshot.setter
    def screenshot(self, value):
        """Set screenshot value"""
        self.__set_mode(self.SCREENSHOT_ENV, value)

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

    @property
    def delay(self):
        """Delay (ms) before quitting application in unattended mode"""
        try:
            return int(os.environ.get(self.DELAY_ENV))
        except (TypeError, ValueError):
            return 0

    @delay.setter
    def delay(self, value: int):
        """Set delay (ms) before quitting application in unattended mode"""
        os.environ[self.DELAY_ENV] = str(value)

    @property
    def xmlrpcport(self):
        """XML-RPC port number"""
        try:
            return int(os.environ.get(self.XMLRPCPORT_ENV))
        except (TypeError, ValueError):
            return None

    @xmlrpcport.setter
    def xmlrpcport(self, value: int):
        """Set XML-RPC port number"""
        os.environ[self.XMLRPCPORT_ENV] = str(value)

    def parse_args(self):
        """Parse command line arguments"""

        # <!> WARNING <!>
        # Do not add an option '-c' to avoid any conflict with macro command
        # execution mecanism used with DataLab standalone version (see start.pyw)

        parser = argparse.ArgumentParser(description="Run DataLab")
        parser.add_argument(
            "h5",
            nargs="?",
            type=str,
            help="HDF5 file names (separated by ';'), "
            "optionally with dataset name (separated by ',')",
        )
        parser.add_argument(
            "-b",
            "--h5browser",
            required=False,
            type=str,
            metavar="path",
            help="path to open with HDF5 browser",
        )
        parser.add_argument(
            "-v", "--version", action="store_true", help="show DataLab version"
        )
        parser.add_argument(
            "--reset", action="store_true", help="reset DataLab configuration"
        )
        parser.add_argument(
            "--" + self.UNATTENDED_ARG,
            action="store_true",
            help="non-interactive mode",
            default=None,
        )
        parser.add_argument(
            "--" + self.ACCEPT_DIALOGS_ARG,
            action="store_true",
            help="accept dialogs in unattended mode",
            default=None,
        )
        parser.add_argument(
            "--" + self.SCREENSHOT_ARG,
            action="store_true",
            help="automatic screenshots",
            default=None,
        )
        parser.add_argument(
            "--" + self.DELAY_ARG,
            type=int,
            default=None,
            help="delay (ms) before quitting application in unattended mode",
        )
        parser.add_argument(
            "--" + self.XMLRPCPORT_ARG,
            type=int,
            default=None,
            help="XML-RPC port number",
        )
        parser.add_argument(
            "--" + self.VERBOSE_ARG,
            choices=[lvl.value for lvl in VerbosityLevels],
            required=False,
            default=None,
            help="verbosity level: for debugging/testing purpose",
        )
        args, _unknown = parser.parse_known_args()
        if args.h5:
            self.h5files = args.h5.split(";")
        if args.h5browser:
            self.h5browser_file = args.h5browser
        if args.version:
            version = os.environ["CDL_VERSION"]
            print(f"DataLab {version} on {platform.system()}")
            sys.exit()
        if args.reset:  # Remove ".DataLab" configuration directory
            # pylint: disable=import-outside-toplevel
            from datalab.config import Conf

            print("Resetting DataLab configuration...", end=" ")
            try:
                Conf.reset()
            except Exception:  # pylint: disable=broad-except
                print("Failed.")
                traceback.print_exc()
            finally:
                print("Done.")
            sys.exit()
        self.set_env_from_args(args)

    def set_env_from_args(self, args):
        """Set appropriate environment variables"""
        for argname in (
            self.UNATTENDED_ARG,
            self.ACCEPT_DIALOGS_ARG,
            self.SCREENSHOT_ARG,
            self.VERBOSE_ARG,
            self.DELAY_ARG,
            self.XMLRPCPORT_ARG,
        ):
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
        accept_dialogs=None,
        screenshot=None,
        delay=None,
        verbose=None,
        xmlrpcport=None,
        catcher_test=None,
    ) -> Generator[None, None, None]:
        """Return a context manager that sets some execenv properties at enter,
        and restores them at exit. This is useful to run some code in a
        controlled environment, for example to accept dialogs in unattended
        mode, and restore the previous value at exit.

        Args:
            unattended: whether to run in unattended mode
            accept_dialogs: whether to accept dialogs in unattended mode
            screenshot: whether to take screenshots
            delay: delay (ms) before quitting application in unattended mode
            verbose: verbosity level
            xmlrpcport: XML-RPC port number
            catcher_test: whether to run catcher test

        .. note::
            If a passed value is None, the corresponding property is not changed.
        """
        old_values = self.to_dict()
        new_values = {
            "unattended": unattended,
            "accept_dialogs": accept_dialogs,
            "screenshot": screenshot,
            "delay": delay,
            "verbose": verbose,
            "xmlrpcport": xmlrpcport,
            "catcher_test": catcher_test,
        }
        for key, value in new_values.items():
            if value is not None:
                setattr(self, key, value)
        try:
            yield
        finally:
            for key, value in old_values.items():
                setattr(self, key, value)


execenv = CDLExecEnv()
