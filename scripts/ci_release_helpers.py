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

These commands are intentionally dependency-free (stdlib only) so they
can run on a minimal Python install at the very start of a CI job.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_NOTES_DIR = REPO_ROOT / "doc" / "release_notes"

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
