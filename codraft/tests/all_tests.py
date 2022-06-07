# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Run all tests in unattended mode
"""

import argparse
import os
import os.path as osp

from guidata.guitest import get_tests

import codraft
from codraft import config
from codraft.utils.tests import TST_PATH

SHOW = True  # Show test in GUI-based test launcher


def run_all_tests(args="", contains="", timeout=None):
    """Run all CodraFT tests"""
    testmodules = [
        tmod
        for tmod in get_tests(codraft)
        if not osp.samefile(tmod.path, __file__) and contains in tmod.path
    ]
    tnb = len(testmodules)
    print("*** CodraFT automatic unit tests ***")
    print("")
    print("Test parameters:")
    print(f"  Selected {tnb} tests ({len(get_tests(codraft)) - 1} total available)")
    print("  Test data path:")
    for path in TST_PATH:
        print(f"    {path}")
    print("  Environment:")
    for vname in ("DATA_CODRAFT", "PYTHONPATH", "DEBUG"):
        print(f"    {vname}={os.environ.get(vname, '')}")
    print("")
    print("Please wait while test scripts are executed (a few minutes).")
    print("Only error messages will be printed out (no message = test OK).")
    print("")
    for idx, testmodule in enumerate(testmodules):
        rpath = osp.relpath(testmodule.path, osp.dirname(codraft.__file__))
        print(f"===[{(idx+1):02d}/{tnb:02d}]=== 🍺 Running test [{rpath}]")
        testmodule.run(args=args, timeout=timeout)


def run():
    """Parse arguments and run tests"""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    parser = argparse.ArgumentParser(description="Run all test in unattended mode")
    parser.add_argument("--contains", default="")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    run_all_tests("--mode unattended --verbose quiet", args.contains, args.timeout)


if __name__ == "__main__":
    run()
