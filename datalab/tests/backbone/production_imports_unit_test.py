# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Unit tests keeping the application independent from the test suite

The standalone application (MSI installer) and ``pip install datalab-platform``
do not provide the test dependencies (``pytest``...): the application must not
import :mod:`datalab.tests`, which requires them (see issue #365, where the demo
failed with "No module named 'pytest'").
"""

from __future__ import annotations

import ast
import os
import os.path as osp
import subprocess
import sys
from collections.abc import Iterator

import pytest

import datalab

DATALAB_ROOT = osp.dirname(osp.abspath(datalab.__file__))
TESTS_PACKAGE = "datalab.tests"

#: Top-level modules of the test dependencies (``test`` extra of DataLab)
TEST_DEPENDENCIES = ("pytest", "_pytest", "pytest_xvfb", "httpx")

#: Modules of the application which must be importable without test dependencies
APPLICATION_MODULES = ("datalab.app", "datalab.gui.demo")

#: Launchers which are allowed to run the test suite
TEST_LAUNCHERS = ("datalab-tests",)

BLOCKED_IMPORT_SCRIPT = """\
import importlib
import importlib.abc
import sys

BLOCKED = {blocked!r}


class TestDependencyBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{fullname!r}}", name=fullname)
        return None


sys.meta_path.insert(0, TestDependencyBlocker())
for name in {modules!r}:
    importlib.import_module(name)
print(
    sorted(
        name
        for name in sys.modules
        if name.partition(".")[0] in BLOCKED
        or name == {tests!r}
        or name.startswith({tests!r} + ".")
    )
)
"""


def iter_application_files() -> Iterator[tuple[str, str]]:
    """Yield the path and the package name of DataLab modules, except tests

    Yields:
        Tuples (file path, name of the package containing the module)
    """
    parent = osp.dirname(DATALAB_ROOT)
    for dirpath, dirnames, filenames in os.walk(DATALAB_ROOT):
        if dirpath == DATALAB_ROOT:
            dirnames.remove("tests")
        dirnames[:] = [name for name in dirnames if name != "__pycache__"]
        package = osp.relpath(dirpath, parent).replace(os.sep, ".")
        for filename in filenames:
            if filename.endswith(".py"):
                yield osp.join(dirpath, filename), package


def resolve_import(node: ast.ImportFrom, package: str) -> str:
    """Return the absolute name of the module of a ``from ... import`` statement

    Args:
        node: import statement
        package: name of the package containing the module

    Returns:
        Absolute module name
    """
    if node.level == 0:
        return node.module or ""
    base = package.split(".")[: len(package.split(".")) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def is_tests_module(name: str) -> bool:
    """Return True if the module name belongs to the DataLab test suite"""
    return name == TESTS_PACKAGE or name.startswith(TESTS_PACKAGE + ".")


def find_tests_imports(path: str, package: str) -> list[str]:
    """Return the imports of the test suite in a module, including lazy ones

    Args:
        path: module file path
        package: name of the package containing the module

    Returns:
        List of "path:line: statement" descriptions
    """
    with open(path, encoding="utf-8") as file:
        source = file.read()
    found = []
    for node in ast.walk(ast.parse(source, filename=path)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = resolve_import(node, package)
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        else:
            continue
        if any(is_tests_module(name) for name in names):
            relpath = osp.relpath(path, osp.dirname(DATALAB_ROOT))
            found.append(
                f"{relpath}:{node.lineno}: {ast.get_source_segment(source, node)}"
            )
    return found


def test_application_does_not_import_test_suite() -> None:
    """No module of the application imports the test suite, even lazily"""
    found = []
    for path, package in iter_application_files():
        found.extend(find_tests_imports(path, package))
    assert not found, "Application modules importing the test suite:\n" + "\n".join(
        found
    )


def test_application_imports_without_test_dependencies() -> None:
    """The application and its demo import without the test dependencies"""
    code = BLOCKED_IMPORT_SCRIPT.format(
        blocked=TEST_DEPENDENCIES, modules=APPLICATION_MODULES, tests=TESTS_PACKAGE
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout


def test_launchers_do_not_run_test_suite() -> None:
    """Command line launchers, except the test launcher, use the application"""
    tomllib = pytest.importorskip("tomllib")
    path = osp.join(osp.dirname(DATALAB_ROOT), "pyproject.toml")
    if not osp.isfile(path):
        pytest.skip("pyproject.toml is not available (installed package)")
    with open(path, "rb") as file:
        project = tomllib.load(file)["project"]
    launchers = project["gui-scripts"] | project["scripts"]
    assert {
        name: target
        for name, target in launchers.items()
        if name not in TEST_LAUNCHERS and is_tests_module(target.partition(":")[0])
    } == {}


if __name__ == "__main__":
    test_application_does_not_import_test_suite()
    test_application_imports_without_test_dependencies()
    test_launchers_do_not_run_test_suite()
