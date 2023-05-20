# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Run all tests in unattended mode
"""

import argparse
import os
import os.path as osp

from guidata.guitest import get_tests

import cdl
from cdl import __version__, config
from cdl.config import Conf
from cdl.utils.conf import Option
from cdl.utils.tests import TST_PATH

SHOW = True  # Show test in GUI-based test launcher


def get_test_modules(package, contains=""):
    """Return test module list for package"""
    return [
        tmod
        for tmod in get_tests(package)
        if osp.basename(tmod.path) != osp.basename(__file__) and contains in tmod.path
    ]


def __get_enabled(confopt: Option) -> str:
    """Return enable state of a configuration option"""
    return "enabled" if confopt.get() else "disabled"


def run_all_tests(args="", contains="", timeout=None, other_package=None):
    """Run all DataLab tests"""
    testmodules = get_test_modules(cdl, contains=contains)
    testnb = len(get_tests(cdl)) - 1
    if other_package is not None:
        testmodules += get_test_modules(other_package, contains=contains)
        testnb += len(get_tests(other_package)) - 1
    tnb = len(testmodules)
    print("=" * 80)
    print(f"🚀 DataLab v{__version__} automatic unit tests 🌌")
    print("=" * 80)
    print("")
    print("🔥 DataLab characteristics/environment:")
    print(f"  Configuration version: {config.CONF_VERSION}")
    print(f"  Path: {config.APP_PATH}")
    print(f"  Frozen: {config.IS_FROZEN}")
    print(f"  Debug: {config.DEBUG}")
    print("")
    print("🔥 DataLab configuration:")
    print(f"  Process isolation: {__get_enabled(Conf.main.process_isolation_enabled)}")
    print(f"  RPC server: {__get_enabled(Conf.main.rpc_server_enabled)}")
    print(f'  Console: {__get_enabled(Conf.console.enabled)}')
    mem_threshold = Conf.main.available_memory_threshold.get()
    print(f"  Available memory threshold: {mem_threshold:d} MB")
    print(f"  Ignored dependencies: {__get_enabled(Conf.main.ignore_dependency_check)}")
    print("  Processing:")
    if Conf.proc.extract_roi_singleobj:
        roi_extract = "Extract all ROIs in a single signal or image"
    else:
        roi_extract = "Extract each ROI in a separate signal or image"
    print(f"    {roi_extract}")
    print(f"    FFT shift: {__get_enabled(Conf.proc.fft_shift_enabled)}")
    print("")
    print("🔥 Test parameters:")
    print(f"  ⚡ Selected {tnb} tests ({testnb} total available)")
    if other_package is not None:
        print("  Additional package:")
        print(f"    {other_package.__name__}")
    print("  ⚡ Test data path:")
    for path in TST_PATH:
        print(f"    {path}")
    print("  ⚡ Environment:")
    for vname in ("CDL_DATA", "PYTHONPATH", "DEBUG"):
        print(f"    {vname}={os.environ.get(vname, '')}")
    print("")
    print("Please wait while test scripts are executed (a few minutes).")
    print("Only error messages will be printed out (no message = test OK).")
    print("")
    for idx, testmodule in enumerate(testmodules):
        rpath = osp.relpath(testmodule.path, osp.dirname(cdl.__file__))
        print(f"===[{(idx+1):02d}/{tnb:02d}]=== 🍺 Running test [{rpath}]")
        testmodule.run(args=args, timeout=timeout)


def run(other_package=None):
    """Parse arguments and run tests"""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    parser = argparse.ArgumentParser(description="Run all test in unattended mode")
    parser.add_argument("--contains", default="")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run_all_tests(
        "--mode unattended --verbose quiet",
        args.contains,
        args.timeout,
        other_package,
    )


if __name__ == "__main__":
    run()
