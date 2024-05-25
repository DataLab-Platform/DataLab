# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab information module
--------------------------

This module provides information about the DataLab package, such as the complete
version number including the Git revision, if available.
"""

import subprocess

from cdl import __version__ as RELEASE


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
