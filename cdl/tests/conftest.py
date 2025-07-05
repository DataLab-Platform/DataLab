# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab pytest configuration
----------------------------

This file contains the configuration for running pytest in DataLab. It is
executed before running any tests.
"""

import os
import os.path as osp
import subprocess

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
from sigima.tests import helpers

import cdl
from cdl.env import execenv
from cdl.plugins import get_available_plugins

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
        f"DataLab {cdl.__version__} [Plugins: {nfstr if nfstr else 'None'}]",
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
    for vname in ("CDL_DATA", "PYTHONPATH", "DEBUG", "QT_API", "QT_QPA_PLATFORM"):
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
    sco = subprocess.check_output
    try:
        gitlist = ["Git information:"]
        branch = sco(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        commit = sco(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        message = sco(["git", "log", "-1", "--pretty=%B"], text=True).strip()
        if len(message.splitlines()) > 1:
            message = message.splitlines()[0]
        message = message[:60] + "[…]" if len(message) > 60 else message
        gitlist.append(f"  Branch: {branch}")
        gitlist.append(f"  Commit: {commit}")
        gitlist.append(f"  Message: {message}")
        infolist.extend(gitlist)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return infolist


def pytest_configure(config):
    """Add custom markers to pytest."""
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
