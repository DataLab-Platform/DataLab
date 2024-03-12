"""
DataLab pytest configuration
----------------------------

This file contains the configuration for running pytest in DataLab. It is
executed before running any tests.
"""

import os

import pytest

from cdl import __version__
from cdl.env import execenv
from cdl.plugins import get_available_plugins

# Turn on unattended mode for executing tests without user interaction
execenv.unattended = True
execenv.verbose = "quiet"

INITIAL_CWD = os.getcwd()


def pytest_report_header(config):
    """Add additional information to the pytest report header."""
    nfstr = ", ".join(
        f"{plugin.info.name} {plugin.info.version}"
        for plugin in get_available_plugins()
    )
    return f"DataLab {__version__} [Available plugins: {nfstr if nfstr else 'None'}]"


@pytest.fixture(autouse=True)
def reset_cwd(request):
    """Reset the current working directory to the initial one after each test."""
    yield
    os.chdir(INITIAL_CWD)
