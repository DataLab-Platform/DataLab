# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests of the selection of the CI jobs according to the changed files

The selection (``scripts/ci_changes.py``, used by ``.github/workflows/ci.yml``)
must never skip a job that a change may break: the cases come from real pull
requests.
"""

from __future__ import annotations

import importlib.util
import os.path as osp
import sys

import pytest

SCRIPT_PATH = osp.join(
    osp.dirname(osp.abspath(__file__)), "..", "..", "..", "scripts", "ci_changes.py"
)


@pytest.fixture(name="ci_changes", scope="module")
def fixture_ci_changes():
    """Import the script, available in a source checkout only"""
    if not osp.isfile(SCRIPT_PATH):
        pytest.skip("scripts/ci_changes.py is not available (installed package)")
    spec = importlib.util.spec_from_file_location("ci_changes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look up their module in `sys.modules`
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize(
    "paths, tests, docs",
    [
        pytest.param(
            [
                "doc/intro/introduction.rst",
                "doc/locale/fr/LC_MESSAGES/intro/introduction.po",
                "doc/release_notes/release_1.04.md",
            ],
            False,
            True,
            id="documentation only (#544)",
        ),
        pytest.param(
            [
                ".github/copilot-instructions.md",
                ".github/skills/issue-pr-fusion/SKILL.md",
                ".github/workflows/release.yml",
                "README.md",
            ],
            False,
            False,
            id="contributor files only (#534)",
        ),
        pytest.param(
            [
                "datalab/gui/objectview.py",
                "datalab/tests/features/common/bulk_object_operations_test.py",
                "doc/release_notes/release_1.04.md",
            ],
            True,
            True,
            id="code and release notes (#530)",
        ),
        pytest.param(["pyproject.toml"], True, True, id="dependencies"),
        pytest.param(
            ["doc/update_screenshots.py"], True, True, id="tested screenshot generator"
        ),
        pytest.param(
            ["plugins/examples/datalab_custom_func.py"], True, True, id="example plugin"
        ),
        pytest.param(
            ["datalab/tests/features/plugins/templates/README.md"],
            True,
            True,
            id="Markdown file outside the root",
        ),
        pytest.param(
            ["doc/index.rst", "scripts/build_resources.py"],
            True,
            True,
            id="unknown file",
        ),
        pytest.param(
            [".github/workflows/test_pyqt6.yml"], True, True, id="test workflow"
        ),
        pytest.param(
            [".github/actions/install-dependencies/action.yml"],
            True,
            True,
            id="CI action",
        ),
        pytest.param(["scripts/ci_changes.py"], True, True, id="selection script"),
        pytest.param(None, True, True, id="changed files unknown"),
    ],
)
def test_select_jobs(ci_changes, paths, tests, docs) -> None:
    """Jobs selected for the changed files"""
    selection = ci_changes.select_jobs(paths)
    assert (selection.tests, selection.docs) == (tests, docs), selection.reason


@pytest.mark.parametrize(
    "event_name, event",
    [
        ("push", {"before": "0" * 40, "after": "1" * 40}),
        ("push", {"before": "1" * 40, "after": "0" * 40}),
        ("workflow_dispatch", {}),
    ],
    ids=["new branch", "deleted branch", "other event"],
)
def test_changed_files_unknown(ci_changes, event_name, event) -> None:
    """Events whose changed files are unknown, hence running every job"""
    assert ci_changes.event_changed_files(event_name, event) is None


def test_github_output(ci_changes, tmp_path, monkeypatch) -> None:
    """Selection written to the outputs of the step, as read by ``ci.yml``"""
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    assert ci_changes.main(["doc/index.rst"]) == 0
    assert output.read_text(encoding="utf-8") == "tests=false\ndocs=true\n"
