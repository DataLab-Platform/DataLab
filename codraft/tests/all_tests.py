# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Run all tests in unattended mode
"""

import argparse
import os.path as osp

from guidata.guitest import get_tests

import codraft
from codraft.config import Conf

SHOW = True  # Show test in GUI-based test launcher


def run_all_tests(args="", contains="", timeout=None):
    """Run all CodraFT tests"""
    for testmodule in get_tests(codraft):
        if not osp.samefile(testmodule.path, __file__) and contains in testmodule.path:
            rpath = osp.relpath(testmodule.path, osp.dirname(codraft.__file__))
            print(f"===🍺=== Running test [{rpath}]")
            testmodule.run(args=args, timeout=timeout)


def run():
    """Parse arguments and run tests"""
    Conf.reset()  # Reset configuration (remove configuration file)
    parser = argparse.ArgumentParser(description="Run all test in unattended mode")
    parser.add_argument("--contains", default="")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    run_all_tests("--mode unattended", args.contains, args.timeout)


if __name__ == "__main__":
    run()
