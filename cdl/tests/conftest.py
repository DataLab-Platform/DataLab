"""
DataLab pytest configuration
----------------------------

This file contains the configuration for running pytest in DataLab. It is
executed before running any tests.
"""

import os

import pytest

from cdl.env import execenv

# Turn on unattended mode for executing tests without user interaction
execenv.unattended = True
execenv.verbose = "quiet"

INITIAL_CWD = os.getcwd()


@pytest.fixture(autouse=True)
def reset_cwd(request):
    """Reset the current working directory to the initial one after each test."""
    yield
    os.chdir(INITIAL_CWD)
