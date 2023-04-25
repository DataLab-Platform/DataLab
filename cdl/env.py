# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see cdl/__init__.py for details)

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


class CDLExecEnv:
    """Object representing DataLab test environment"""

    UNATTENDED_ARG = "unattended"
    VERBOSE_ARG = "verbose"
    SCREENSHOT_ARG = "screenshot"
    DELAY_ARG = "delay"
    XMLRPCPORT_ARG = "port"
    UNATTENDED_ENV = "CDL_UNATTENDED_TESTS"
    VERBOSE_ENV = "CDL_VERBOSITY_LEVEL"
    SCREENSHOT_ENV = "CDL_TAKE_SCREENSHOT"
    DELAY_ENV = "CDL_DELAY_BEFORE_QUIT"
    XMLRPCPORT_ENV = "CDL_XMLRPC_PORT"

    def __init__(self):
        self.h5files = None
        self.h5browser_file = None
        self.demo_mode = False
        self.parse_args()

    def enable_demo_mode(self, delay: int):
        """Enable demo mode"""
        self.demo_mode = True
        self.unattended = True
        self.delay = delay

    @staticmethod
    def __get_mode(env):
        """Get mode value"""
        return os.environ.get(env) is not None

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
    def screenshot(self):
        """Get screenshot value"""
        return self.__get_mode(self.SCREENSHOT_ENV)

    @screenshot.setter
    def screenshot(self, value):
        """Set screenshot value"""
        self.__set_mode(self.SCREENSHOT_ENV, value)
        if value:  # pragma: no cover
            self.unattended = value

    @property
    def verbose(self):
        """Get verbosity level"""
        return os.environ.get(self.VERBOSE_ENV, VerbosityLevels.NORMAL.value)

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
    def port(self):
        """XML-RPC port number"""
        try:
            return int(os.environ.get(self.XMLRPCPORT_ENV))
        except (TypeError, ValueError):
            return None

    @port.setter
    def port(self, value: int):
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
            "--mode",
            choices=[self.UNATTENDED_ARG, self.SCREENSHOT_ARG],
            required=False,
            help="unattended: non-interactive test mode ; "
            "screenshot: unattended mode, with automatic screenshots",
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
        if args.mode is not None:
            self.unattended = args.mode == self.UNATTENDED_ARG
            self.screenshot = args.mode == self.SCREENSHOT_ARG
        if args.verbose is not None:
            self.verbose = args.verbose
        self.delay = args.delay
        if args.port is not None:
            self.port = args.port

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
