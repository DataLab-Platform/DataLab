# DataLab pytest configuration
# ----------------------------

from cdl.env import execenv

# Turn on unattended mode for executing tests without user interaction
execenv.unattended = True
execenv.verbose = "quiet"
