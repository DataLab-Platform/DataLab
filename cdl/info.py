# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab information module
--------------------------

This module provides information about the DataLab package, such as the complete
version number including the Git revision, if available.
"""

from __future__ import annotations

import os
import subprocess
import sys

from cdl import __version__ as RELEASE


def get_git_revision() -> tuple[str, str] | None:
    """Get the current Git branch and revision (short hash) of the repository.

    Returns:
        A tuple containing the branch name and the short revision hash.
        If the current branch is 'main' or the Git command fails, return None.
    """
    if __file__.startswith(sys.prefix):
        # If the package is installed in the current Python environment, return None
        # because we won't have access to the Git repository anyway, so we must
        # assume that this is a stable release.
        return None

    startupinfo = None
    if os.name == "nt":  # Check if the OS is Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

    try:
        # Run the git command to get the current branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            startupinfo=startupinfo,
        )
        branch = result.stdout.strip()
        if branch == "main":
            # If the branch is "main", return None (assume the main branch is stable)
            return None
        # Run the git command to get the current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            startupinfo=startupinfo,
        )
        sha = result.stdout.strip()
        return branch, sha
    except subprocess.CalledProcessError:
        # If the git command fails (e.g., not a git repository), return None
        return None
    except FileNotFoundError:
        # If git is not installed, return None
        return None


def get_version() -> str:
    """Get the version number of the package, including the Git revision if available"""
    git_branch_revision = get_git_revision()
    if git_branch_revision is not None:
        # Append the Git revision to the version number
        return RELEASE + f"-{git_branch_revision[0]}.{git_branch_revision[1]}"
    return RELEASE
