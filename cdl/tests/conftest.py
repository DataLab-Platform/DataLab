"""
DataLab pytest configuration
----------------------------

This file contains the configuration for running pytest in DataLab. It is
executed before running any tests.
"""

from cdl.env import execenv

# Turn on unattended mode for executing tests without user interaction
execenv.unattended = True
execenv.verbose = "quiet"
