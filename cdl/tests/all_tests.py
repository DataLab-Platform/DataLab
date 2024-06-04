# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Run all tests in unattended mode
"""

# guitest: show

import argparse
import os
import os.path as osp

from guidata.guitest import get_tests

import cdl
from cdl import config
from cdl.config import Conf
from cdl.utils.conf import Option
from cdl.utils.tests import TST_PATH


def get_test_modules(package, contains=""):
    """Return test module list for package

    Args:
        package (module): package to test
        contains (str): string to match in test module path

    Returns:
        tuple: (selected test module list, total number of test modules)
    """
    allbatch_testmodules = get_tests(package, category="batch")
    selected_testmodules = [
        tmod
        for tmod in allbatch_testmodules
        if osp.basename(tmod.path) != osp.basename(__file__) and contains in tmod.path
    ]
    return selected_testmodules, len(allbatch_testmodules) - 1


def __get_enabled(confopt: Option) -> str:
    """Return enable state of a configuration option"""
    return "enabled" if confopt.get() else "disabled"


def run_all_tests(args="", contains="", timeout=None, other_package=None):
    """Run all DataLab tests"""
    testmodules, testnb = get_test_modules(cdl, contains=contains)
    if other_package is not None:
        othermodules, othernb = get_test_modules(other_package, contains=contains)
        testmodules += othermodules
        testnb += othernb
    tnb = len(testmodules)
    print("")
    print(f"            🚀 DataLab v{cdl.__version__} automatic unit tests 🌌")
    print("")
    print("🔥 DataLab characteristics/environment:")
    print(f"  Configuration version: {config.CONF_VERSION}")
    print(f"  Path: {config.APP_PATH}")
    print(f"  Debug: {config.DEBUG}")
    print("")
    print("🔥 DataLab configuration:")
    print(f"  Process isolation: {__get_enabled(Conf.main.process_isolation_enabled)}")
    print(f"  RPC server: {__get_enabled(Conf.main.rpc_server_enabled)}")
    print(f"  Console: {__get_enabled(Conf.console.console_enabled)}")
    mem_threshold = Conf.main.available_memory_threshold.get()
    print(f"  Available memory threshold: {mem_threshold:d} MB")
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
    print(f"  ⚡ Timeout: {timeout} s")
    print("")
    print("Please wait while test scripts are executed (a few minutes).")
    print("Only error messages will be printed out (no message = test OK).")
    print("")
    for idx, testmodule in enumerate(testmodules):
        rpath = osp.relpath(testmodule.path, osp.dirname(cdl.__file__))
        print(f" 🔹 [{(idx+1):02d}/{tnb:02d}] 🍺 Running test: {rpath}")
        testmodule.run(args=args, timeout=timeout)


def run(other_package=None):
    """Parse arguments and run tests"""
    config.reset()  # Reset configuration (remove configuration file and initialize it)
    parser = argparse.ArgumentParser(description="Run all test in unattended mode")
    parser.add_argument("--contains", default="")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    run_all_tests(
        "--unattended --verbose quiet",
        args.contains,
        args.timeout,
        other_package,
    )


if __name__ == "__main__":
    run()
