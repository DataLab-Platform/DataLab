# Copyright (c) DataLab Platform Developers, BSD 3-Clause license.

"""DataLab environment utilities."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import traceback
from typing import Any

from sigimax.env import SGMXExecEnv, VerbosityLevels

# We could import DEBUG from datalab.config, but is it really worth it?
DEBUG = os.environ.get("DATALAB_DEBUG", "").lower() in ("1", "true")


class DLExecEnv(SGMXExecEnv):
    """Object representing the DataLab execution environment."""

    XMLRPCPORT_ARG = "xmlrpcport"
    DO_NOT_QUIT_ENV = SGMXExecEnv.DO_NOT_QUIT_ENV
    XMLRPCPORT_ENV = "DATALAB_XMLRPCPORT"
    CATCHER_TEST_ENV = "DATALAB_CATCHER_TEST"

    @property
    def xmlrpcport(self) -> int | None:
        """Return the XML-RPC port number."""
        try:
            return int(os.environ.get(self.XMLRPCPORT_ENV))
        except (TypeError, ValueError):
            return None

    @xmlrpcport.setter
    def xmlrpcport(self, value: int | None) -> None:
        """Set the XML-RPC port number."""
        if value is None:
            os.environ.pop(self.XMLRPCPORT_ENV, None)
        else:
            os.environ[self.XMLRPCPORT_ENV] = str(value)

    def parse_args(self) -> None:
        """Parse DataLab-specific command-line arguments."""
        # Do not add an option '-c' to avoid conflict with macro execution.
        parser = argparse.ArgumentParser(description="Run DataLab")
        parser.add_argument(
            "h5",
            nargs="?",
            type=str,
            help="HDF5 file names (separated by ';'), optionally with dataset "
            "name (separated by ',')",
        )
        parser.add_argument(
            "-b",
            "--h5browser",
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
            default=None,
            help="non-interactive mode",
        )
        parser.add_argument(
            "--" + self.ACCEPT_DIALOGS_ARG,
            action="store_true",
            default=None,
            help="accept dialogs in unattended mode",
        )
        parser.add_argument(
            "--" + self.SCREENSHOT_ARG,
            action="store_true",
            default=None,
            help="automatic screenshots",
        )
        parser.add_argument(
            "--" + self.SCREENSHOT_PATH_ARG,
            type=str,
            default=None,
            help="path to save screenshots",
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
            choices=[level.value for level in VerbosityLevels],
            default=None,
            help="verbosity level: for debugging/testing purpose",
        )
        args, _unknown = parser.parse_known_args()

        if args.h5:
            self.h5files = args.h5.split(";")
        if args.h5browser:
            self.h5browser_file = args.h5browser
        if args.version:
            version = os.environ["DATALAB_VERSION"]
            print(f"DataLab {version} on {platform.system()}")
            sys.exit()
        if args.reset:
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

    def set_env_from_args(self, args: argparse.Namespace) -> None:
        """Set DataLab environment variables from parsed arguments."""
        for argname in (
            self.UNATTENDED_ARG,
            self.ACCEPT_DIALOGS_ARG,
            self.SCREENSHOT_ARG,
            self.SCREENSHOT_PATH_ARG,
            self.VERBOSE_ARG,
            self.DELAY_ARG,
            self.XMLRPCPORT_ARG,
        ):
            argvalue = getattr(args, argname)
            if argvalue is not None:
                setattr(self, argname, argvalue)

    def log(self, source: Any, *objects: Any) -> None:
        """Log text on screen using the DataLab debug flag."""
        if DEBUG or self.verbose == VerbosityLevels.DEBUG.value:
            print(str(source) + ":", *objects)


execenv = DLExecEnv()
