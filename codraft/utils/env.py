# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
CodraFT environmnent utilities
"""

import argparse
import enum
import os
import pprint
import sys


class VerbosityLevels(enum.Enum):
    """Print verbosity levels (for testing purpose)"""

    QUIET = "quiet"
    MINIMAL = "minimal"
    NORMAL = "normal"


class CodraFTExecEnv:
    """Object representing CodraFT test environment"""

    UNATTENDED_ARG = "unattended"
    VERBOSE_ARG = "verbose"
    SCREENSHOT_ARG = "screenshot"
    DELAY_ARG = "delay"
    UNATTENDED_ENV = "CODRAFT_UNATTENDED_TESTS"
    VERBOSE_ENV = "CODRAFT_VERBOSITY_LEVEL"
    SCREENSHOT_ENV = "CODRAFT_TAKE_SCREENSHOT"
    DELAY_ENV = "CODRAFT_DELAY_BEFORE_QUIT"

    def __init__(self):
        self.parse_args()

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

    def parse_args(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="Run ??? test")
        parser.add_argument(
            "--mode",
            choices=[self.UNATTENDED_ARG, self.SCREENSHOT_ARG],
            required=False,
        )
        parser.add_argument(
            "--" + self.VERBOSE_ARG,
            choices=[lvl.value for lvl in VerbosityLevels],
            required=False,
            default=VerbosityLevels.NORMAL.value,
        )
        parser.add_argument(
            "--" + self.DELAY_ARG, type=int, default=0, help=self.delay.__doc__
        )
        args, _unknown = parser.parse_known_args()
        self.set_env_from_args(args)

    def set_env_from_args(self, args):
        """Set appropriate environment variables"""
        if args.mode is not None:
            self.unattended = args.mode == self.UNATTENDED_ARG
            self.screenshot = args.mode == self.SCREENSHOT_ARG
        if args.verbose is not None:
            self.verbose = args.verbose
        os.environ[self.DELAY_ENV] = str(args.delay)

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


execenv = CodraFTExecEnv()
