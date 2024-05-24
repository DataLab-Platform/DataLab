# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab
=======

DataLab is a generic signal and image processing software based on Python
scientific libraries (such as NumPy, SciPy or scikit-image) and Qt graphical
user interfaces (thanks to `PlotPyStack`_ libraries).

.. _PlotPyStack: https://github.com/PlotPyStack
"""

from __future__ import annotations

import os
import subprocess

RELEASE = "0.15.1"


def get_git_revision() -> str | None:
    """Get the current Git revision (short hash) of the repository"""
    try:
        # Run the git command to get the current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        return sha
    except subprocess.CalledProcessError:
        # If the git command fails (e.g., not a git repository), return None
        return None
    except FileNotFoundError:
        # If git is not installed, return None
        return None


def get_version() -> str:
    """Get the version number of the package, including the Git revision if available"""
    git_revision = get_git_revision()
    if git_revision is not None:
        # Append the Git revision to the version number
        return RELEASE + f"-dev.{git_revision}"
    return RELEASE


__version__ = get_version()
__docurl__ = __homeurl__ = "https://datalab-platform.com/"
__supporturl__ = "https://github.com/DataLab-Platform/DataLab/issues/new/choose"

os.environ["CDL_VERSION"] = __version__

try:
    import cdl.patch  # analysis:ignore  # noqa: F401
except ImportError:
    if not os.environ.get("CDL_DOC"):
        raise

# Dear (Debian, RPM, ...) package makers, please feel free to customize the
# following path to module's data (images) and translations:
DATAPATH = LOCALEPATH = ""
