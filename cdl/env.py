# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab environmnent utilities
"""

import argparse
import enum
import os
import platform
import pprint
import sys


class VerbosityLevels(enum.Enum):
    """Print verbosity levels (for testing purpose)"""

    QUIET = "quiet"
    MINIMAL = "minimal"
    NORMAL = "normal"


# TODO: Rewrite this class so that options are automatically associated with
#       environment variables and command line arguments. This could be done
#       using objects deriving from something like this (and implementing
#       integer, boolean, string, choices):
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
    VERBOSE_ARG = "verbose"
    SCREENSHOT_ARG = "screenshot"
    DELAY_ARG = "delay"
    XMLRPCPORT_ARG = "xmlrpcport"
    DONOTQUIT_ENV = "CDL_DO_NOT_QUIT"
    UNATTENDED_ENV = "CDL_UNATTENDED"
    VERBOSE_ENV = "CDL_VERBOSE"
    SCREENSHOT_ENV = "CDL_SCREENSHOT"
    DELAY_ENV = "CDL_DELAY"
    XMLRPCPORT_ENV = "CDL_XMLRPCPORT"
    CATCHER_TEST_ENV = "CDL_CATCHER_TEST"

    def __init__(self):
        self.h5files = None
        self.h5browser_file = None
        self.demo_mode = False
        self.parse_args()

    def to_dict(self):
        """Return a dictionary representation of the object"""
        # Return textual representation of object attributes and properties
        props = [
            "h5files",
            "h5browser_file",
            "demo_mode",
            "do_not_quit",
            "unattended",
            "catcher_test",
            "screenshot",
            "verbose",
            "delay",
            "xmlrpcport",
        ]
        return {p: getattr(self, p) for p in props}

    def __str__(self):
        """Return a string representation of the object"""
        return pprint.pformat(self.to_dict())

    def enable_demo_mode(self, delay: int):
        """Enable demo mode"""
        self.demo_mode = True
        self.unattended = True
        self.delay = delay

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
        return self.__get_mode(self.DONOTQUIT_ENV)

    @do_not_quit.setter
    def do_not_quit(self, value):
        """Set do_not_quit value"""
        self.__set_mode(self.DONOTQUIT_ENV, value)

    @property
    def unattended(self):
        """Get unattended value"""
        return self.__get_mode(self.UNATTENDED_ENV)

    @unattended.setter
    def unattended(self, value):
        """Set unattended value"""
        self.__set_mode(self.UNATTENDED_ENV, value)

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
        """Delay (seconds) before quitting application in unattended mode"""
        try:
            return int(os.environ.get(self.DELAY_ENV))
        except (TypeError, ValueError):
            return 0

    @delay.setter
    def delay(self, value: int):
        """Set delay (seconds) before quitting application in unattended mode"""
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
            "--" + self.UNATTENDED_ARG, action="store_true", help="non-interactive mode"
        )
        parser.add_argument(
            "--" + self.SCREENSHOT_ARG,
            action="store_true",
            help="automatic screenshots",
        )
        parser.add_argument(
            "--" + self.DELAY_ARG,
            type=int,
            default=0,
            help="delay (seconds) before quitting application in unattended mode",
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
            default=VerbosityLevels.NORMAL.value,
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
        self.set_env_from_args(args)

    def set_env_from_args(self, args):
        """Set appropriate environment variables"""
        for argname in (
            self.UNATTENDED_ARG,
            self.SCREENSHOT_ARG,
            self.VERBOSE_ARG,
            self.DELAY_ARG,
            self.XMLRPCPORT_ARG,
        ):
            argvalue = getattr(args, argname)
            if argvalue is not None:
                setattr(self, argname, argvalue)

    def print(self, *objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        """Print in file, depending on verbosity level"""
        # print(f"unattended={self.unattended} ; verbose={self.verbose} ; ")
        # print(f"screenshot={self.screenshot}; delay={self.delay}")
        if (self.verbose != VerbosityLevels.QUIET.value) and (
            self.verbose != VerbosityLevels.MINIMAL.value or file == sys.stderr
        ):
            print(*objects, sep=sep, end=end, file=file, flush=flush)
        # TODO: [P4] Eventually add logging here

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
        if (self.verbose != VerbosityLevels.QUIET.value) and (
            self.verbose != VerbosityLevels.MINIMAL.value or stream == sys.stderr
        ):
            pprint.pprint(
                obj,
                stream=stream,
                indent=indent,
                width=width,
                depth=depth,
                compact=compact,
                sort_dicts=sort_dicts,
            )


execenv = CDLExecEnv()
