# -*- coding: utf-8 -*-
"""
CI helpers used by the GitHub release workflows.

The release tag checks and the release notes extraction come from the release
actions shared by the DataLab Platform projects
(https://github.com/DataLab-Platform/release-actions).

Subcommands
-----------

``install-dev-deps``
    Install Sigima, SigimaX and DataLab-Capsule from their git repository, at the
    branch named like ``--branch`` if it exists, else ``develop`` (like the test
    workflows), with compiled translations. Used to build release artifacts
    before these dependencies are published (``dependencies: development`` input
    of ``_build.yml``). Requires git, and guidata in the current environment.

These commands are intentionally dependency-free (stdlib only) so they
can run on a minimal Python install at the very start of a CI job.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: Dependencies installed by ``install-dev-deps``: git repository and gettext
#: domain of the translations to compile (None: no translation)
DEV_DEPENDENCIES = (
    ("https://github.com/DataLab-Platform/Sigima.git", "sigima"),
    ("https://github.com/DataLab-Platform/SigimaX.git", "sigimax"),
    ("https://github.com/DataLab-Platform/DataLab-Capsule.git", None),
)


def _run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _has_branch(url: str, branch: str) -> bool:
    result = subprocess.run(
        ["git", "ls-remote", "--exit-code", "--heads", url, f"refs/heads/{branch}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def cmd_install_dev_deps(args: argparse.Namespace) -> int:
    """Install Sigima, SigimaX and DataLab-Capsule from their git repository."""
    directory = Path(args.directory).resolve()
    folders = []
    domains = []
    for url, domain in DEV_DEPENDENCIES:
        folder = directory / url.rsplit("/", 1)[-1].removesuffix(".git")
        branch = "develop"
        if args.branch and _has_branch(url, args.branch):
            branch = args.branch
        print(f"{url}: branch {branch}", flush=True)
        if folder.exists():
            shutil.rmtree(folder)
        _run("git", "clone", "--depth", "1", "--branch", branch, url, str(folder))
        folders.append(str(folder))
        if domain is not None:
            domains.append((domain, str(folder)))
    _run(sys.executable, "-m", "pip", "install", "babel")
    for domain, folder in domains:
        _run(
            sys.executable,
            "-m",
            "guidata.utils.translations",
            "compile",
            "--name",
            domain,
            "--directory",
            folder,
        )
    # Not editable: installed like released packages (data files included)
    _run(sys.executable, "-m", "pip", "install", *folders)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_deps = sub.add_parser(
        "install-dev-deps",
        help="Install Sigima, SigimaX and DataLab-Capsule from their git repository",
    )
    p_deps.add_argument(
        "--branch",
        default="",
        help="Branch to install if it exists in a repository (default: develop)",
    )
    p_deps.add_argument(
        "--directory",
        default=os.path.join(
            os.environ.get("RUNNER_TEMP", tempfile.gettempdir()), "dev-deps"
        ),
        help="Directory where repositories are cloned",
    )
    p_deps.set_defaults(func=cmd_install_dev_deps)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
