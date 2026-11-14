# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Select the CI jobs to run according to the files changed by a push or a pull request.

Used by the ``changes`` job of ``.github/workflows/ci.yml``: in GitHub Actions,
without arguments, it reads the triggering event, lists the changed files with git,
then writes ``tests=true|false`` and ``docs=true|false`` to ``$GITHUB_OUTPUT`` and a
summary to ``$GITHUB_STEP_SUMMARY``.

- Tests: unless every changed file is documentation (``doc/``) or a contributor
  file (``.github/``, ``.vscode/``, root Markdown files...).
- Documentation build: when a changed file is an input of the documentation
  (``doc/``, the documented code and examples, the dependencies).
- Both when the CI definition changes or when the changed files are unknown.

To check the selection for given paths: ``python scripts/ci_changes.py PATH...``

This script is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass

#: Files defining the CI jobs selected by this script: changing them runs every job
CI_FILES = frozenset(
    {
        ".github/workflows/ci.yml",
        ".github/workflows/test_doc.yml",
        ".github/workflows/test_pyqt5.yml",
        ".github/workflows/test_pyqt6.yml",
        "scripts/ci_changes.py",
    }
)
CI_PREFIXES = (".github/actions/",)
TEST_INPUT_FILES = frozenset({"doc/update_screenshots.py"})

#: Files not used by the tests (other CI files excepted)
NO_TEST_PREFIXES = ("doc/", ".github/", ".vscode/")
NO_TEST_ROOT_FILES = frozenset(
    {"CITATION.cff", "LICENSE", ".env.template", "DataLab.code-workspace.template"}
)

#: Inputs of the documentation build (``doc/conf.py`` imports DataLab, autodoc
#: documents its API, ``literalinclude`` quotes examples)
DOC_PREFIXES = ("doc/", "datalab/", "plugins/")
DOC_FILES = frozenset({"pyproject.toml", "requirements.txt"})

#: Value of ``before`` in the push event of a new branch
NULL_SHA = "0" * 40


@dataclass(frozen=True)
class Selection:
    """CI jobs to run

    Args:
        tests: Run the tests
        docs: Build the documentation
        reason: Reason of the selection
    """

    tests: bool
    docs: bool
    reason: str


def is_ci_file(path: str) -> bool:
    """Return True if the file defines the CI jobs selected by this script"""
    return path in CI_FILES or path.startswith(CI_PREFIXES)


def needs_tests(path: str) -> bool:
    """Return True if changing the file requires running the tests"""
    if is_ci_file(path) or path in TEST_INPUT_FILES:
        return True
    if path.startswith(NO_TEST_PREFIXES):
        return False
    if "/" not in path and (path.endswith(".md") or path in NO_TEST_ROOT_FILES):
        return False
    return True


def needs_docs(path: str) -> bool:
    """Return True if changing the file requires building the documentation"""
    return is_ci_file(path) or path.startswith(DOC_PREFIXES) or path in DOC_FILES


def select_jobs(paths: Iterable[str] | None) -> Selection:
    """Select the CI jobs to run for the changed files

    Args:
        paths: Paths of the changed files, relative to the repository root, with
         ``/`` separators, or None if they are unknown

    Returns:
        Selected jobs
    """
    if paths is None:
        return Selection(True, True, "changed files unknown")
    paths = list(paths)
    ci_files = [path for path in paths if is_ci_file(path)]
    if ci_files:
        return Selection(True, True, f"CI definition changed ({ci_files[0]})")
    tests = [path for path in paths if needs_tests(path)]
    docs = [path for path in paths if needs_docs(path)]
    reasons = []
    if tests:
        reasons.append(f"tests needed by {tests[0]}")
    if docs:
        reasons.append(f"documentation build needed by {docs[0]}")
    reason = ", ".join(reasons) or "no file used by the tests or the documentation"
    return Selection(bool(tests), bool(docs), f"{len(paths)} changed file(s): {reason}")


def git_changed_files(base: str, head: str) -> list[str] | None:
    """Return the files changed between two commits, or None on failure"""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--no-renames", base, head],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot list the changed files: {exc}", file=sys.stderr)
        return None
    return [line for line in result.stdout.splitlines() if line]


def event_changed_files(event_name: str, event: dict) -> list[str] | None:
    """Return the files changed by a GitHub event, or None if unknown

    Args:
        event_name: Name of the event (``pull_request``, ``push``...)
        event: Payload of the event

    Returns:
        Paths of the changed files, or None if they cannot be determined
    """
    if event_name == "pull_request":
        # The checked out commit merges the pull request into its base branch
        # (`fetch-depth: 2` gives its parents)
        return git_changed_files("HEAD^1", "HEAD")
    if event_name == "push":
        before, after = event.get("before", NULL_SHA), event.get("after", NULL_SHA)
        if NULL_SHA in (before, after):
            return None  # New or deleted branch
        try:
            subprocess.run(
                ["git", "fetch", "--no-tags", "--depth=1", "origin", before],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"Cannot fetch {before}: {exc}", file=sys.stderr)
            return None
        return git_changed_files(before, after)
    return None


def write_github_file(variable: str, text: str) -> None:
    """Append text to the file named by an environment variable, if defined"""
    path = os.environ.get(variable)
    if path:
        with open(path, "a", encoding="utf-8") as file:
            file.write(text)


def main(argv: list[str] | None = None) -> int:
    """Select the CI jobs to run, then print and write the selection"""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        help="Changed files to classify (default: files changed by the GitHub event)",
    )
    args = parser.parse_args(argv)
    if args.paths:
        paths = args.paths
    elif "GITHUB_EVENT_NAME" in os.environ:
        with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as file:
            event = json.load(file)
        paths = event_changed_files(os.environ["GITHUB_EVENT_NAME"], event)
    else:
        parser.error("no path given, and not running in GitHub Actions")
    selection = select_jobs(paths)
    tests, docs = str(selection.tests).lower(), str(selection.docs).lower()
    print(f"tests={tests} docs={docs} ({selection.reason})")
    if paths:
        print("Changed files:", *paths, sep="\n  ")
    write_github_file("GITHUB_OUTPUT", f"tests={tests}\ndocs={docs}\n")
    write_github_file(
        "GITHUB_STEP_SUMMARY",
        "## Selected CI jobs\n\n"
        f"- Tests: **{'yes' if selection.tests else 'no'}**\n"
        f"- Documentation build: **{'yes' if selection.docs else 'no'}**\n\n"
        f"Reason: {selection.reason}.\n",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
