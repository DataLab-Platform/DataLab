# -*- coding: utf-8 -*-
"""
CI helpers used by the GitHub release workflows.

Subcommands
-----------

``check-tag``
    Compare a release tag (``vX.Y.Z`` or ``vX.Y.Z-rcN``) against
    ``datalab.__version__``. Fails with exit code 1 on mismatch.

``release-notes``
    Print the contents of ``doc/release_notes/release_<MAJ>.<MIN:02d>.md``
    matching the given tag/version to stdout (or to ``--output`` file).
    Fails with exit code 1 if the file does not exist.

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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_NOTES_DIR = REPO_ROOT / "doc" / "release_notes"

#: Dependencies installed by ``install-dev-deps``: git repository and gettext
#: domain of the translations to compile (None: no translation)
DEV_DEPENDENCIES = (
    ("https://github.com/DataLab-Platform/Sigima.git", "sigima"),
    ("https://github.com/DataLab-Platform/SigimaX.git", "sigimax"),
    ("https://github.com/DataLab-Platform/DataLab-Capsule.git", None),
)

TAG_RE = re.compile(
    r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<pre>[-.][\w.]+)?$"
)


def _parse_tag(tag: str) -> tuple[int, int, int, str]:
    """Return ``(major, minor, patch, pre)`` from ``vX.Y.Z[-preN]``."""
    match = TAG_RE.match(tag)
    if not match:
        raise SystemExit(
            f"Tag {tag!r} does not match the expected pattern vMAJOR.MINOR.PATCH[-preN]"
        )
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        match["pre"] or "",
    )


def _read_datalab_version() -> str:
    init = REPO_ROOT / "datalab" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Could not find __version__ in {init}")
    return match.group(1)


def cmd_check_tag(args: argparse.Namespace) -> int:
    major, minor, patch, pre = _parse_tag(args.tag)
    tag_version = f"{major}.{minor}.{patch}"
    code_version = _read_datalab_version()
    # Drop any local pre-release suffix in the code version for comparison.
    code_base = re.split(r"[-+]", code_version, maxsplit=1)[0]
    if tag_version != code_base:
        print(
            f"::error::Tag version {tag_version!r} does not match "
            f"datalab.__version__ {code_version!r}",
            file=sys.stderr,
        )
        return 1
    print(f"Tag {args.tag} matches datalab.__version__={code_version}")
    return 0


def _release_notes_path(major: int, minor: int) -> Path:
    return RELEASE_NOTES_DIR / f"release_{major}.{minor:02d}.md"


def cmd_release_notes(args: argparse.Namespace) -> int:
    major, minor, patch, pre = _parse_tag(args.tag)
    path = _release_notes_path(major, minor)
    if not path.is_file():
        print(
            f"::error::Release notes file not found: {path.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )
        return 1
    content = path.read_text(encoding="utf-8")
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        # Force UTF-8 on stdout so the script works on Windows consoles too.
        sys.stdout.buffer.write(content.encode("utf-8"))
    return 0


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

    p_check = sub.add_parser(
        "check-tag", help="Validate tag against datalab.__version__"
    )
    p_check.add_argument("tag", help="Release tag, e.g. v1.3.0 or v1.3.0-rc1")
    p_check.set_defaults(func=cmd_check_tag)

    p_notes = sub.add_parser(
        "release-notes", help="Print release notes file for the given tag"
    )
    p_notes.add_argument("tag", help="Release tag, e.g. v1.3.0")
    p_notes.add_argument(
        "-o", "--output", help="Optional output file (default: stdout)"
    )
    p_notes.set_defaults(func=cmd_release_notes)

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
