# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab pytest configuration
----------------------------

This file contains the configuration for running pytest in DataLab. It is
executed before running any tests.
"""

import os
import os.path as osp

import guidata
import h5py
import numpy
import plotpy
import pytest
import qtpy
import qwt
import scipy
import sigima
import skimage
from guidata.config import ValidationMode, set_validation_mode
from guidata.utils.gitreport import format_git_info_for_pytest, get_git_info_for_modules
from sigima.tests import helpers

import datalab
from datalab.env import execenv
from datalab.plugins import PluginRegistry, get_available_plugins

# Set validation mode to STRICT for all tests
set_validation_mode(ValidationMode.STRICT)

# Turn on unattended mode for executing tests without user interaction
execenv.unattended = True
execenv.verbose = "quiet"

INITIAL_CWD = os.getcwd()


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--show-windows",
        action="store_true",
        default=False,
        help="Display Qt windows during tests (disables QT_QPA_PLATFORM=offscreen)",
    )


def pytest_report_header(config):  # pylint: disable=unused-argument
    """Add additional information to the pytest report header."""
    nfstr = ", ".join(
        f"{plugin.info.name} {plugin.info.version}"
        for plugin in get_available_plugins()
    )
    qtbindings_version = qtpy.PYSIDE_VERSION
    if qtbindings_version is None:
        qtbindings_version = qtpy.PYQT_VERSION
    infolist = [
        f"DataLab {datalab.__version__} [Plugins: {nfstr if nfstr else 'None'}]",
        f"  sigima {sigima.__version__},",
        f"  guidata {guidata.__version__}, PlotPy {plotpy.__version__}",
        f"  PythonQwt {qwt.__version__}, "
        f"{qtpy.API_NAME} {qtbindings_version} [Qt version: {qtpy.QT_VERSION}]",
        f"  NumPy {numpy.__version__}, SciPy {scipy.__version__}, "
        f"h5py {h5py.__version__}, scikit-image {skimage.__version__}",
    ]
    try:
        import cv2  # pylint: disable=import-outside-toplevel

        infolist[-1] += f", OpenCV {cv2.__version__}"
    except ImportError:
        pass
    envlist = []
    for vname in ("DATALAB_DATA", "PYTHONPATH", "DEBUG", "QT_API", "QT_QPA_PLATFORM"):
        value = os.environ.get(vname, "")
        if value:
            if vname == "PYTHONPATH":
                pathlist = value.split(os.pathsep)
                envlist.append(f"  {vname}:")
                envlist.extend(f"    {p}" for p in pathlist if p)
            else:
                envlist.append(f"  {vname}: {value}")
    if envlist:
        infolist.append("Environment variables:")
        infolist.extend(envlist)
    infolist.append("Test paths:")
    for test_path in helpers.get_test_paths():
        test_path = osp.abspath(test_path)
        infolist.append(f"  {test_path}")

    # Git information for all modules using the new gitreport module
    modules_config = [
        ("DataLab", datalab, "."),  # DataLab uses current directory
        ("guidata", guidata, None),
        ("PlotPy", plotpy, None),
        ("Sigima", sigima, None),
    ]
    git_repos = get_git_info_for_modules(modules_config)
    git_info_lines = format_git_info_for_pytest(git_repos, "DataLab")
    if git_info_lines:
        infolist.extend(git_info_lines)

    return infolist


def pytest_configure(config):
    """Add custom markers to pytest."""
    if config.option.durations is None:
        config.option.durations = 20  # Default to showing 20 slowest tests
    config.addinivalue_line(
        "markers",
        "validation: mark a test as a validation test (ground truth or analytical)",
    )
    if not config.getoption("--show-windows"):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def reset_cwd(request):  # pylint: disable=unused-argument
    """Reset the current working directory to the initial one after each test."""
    yield
    os.chdir(INITIAL_CWD)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):  # pylint: disable=unused-argument
    """Run teardown after each test."""
    PluginRegistry.unregister_all_plugins()
