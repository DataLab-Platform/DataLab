# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Update validation status CSV files from compute functions and validation tests"""

from __future__ import annotations

import csv
import importlib
import inspect
import os.path as osp
import pkgutil

from _pytest.mark import Mark

import cdl.core.computation as computation_pkg
import cdl.tests as tests_pkg


def get_compute_functions(package: str) -> list:
    """Retrieve list of `compute_` functions from a package and its submodules

    Args:
        package: Python package

    Returns:
        List of tuples containing the module name, function name, and docstring
    """
    compute_functions = []
    package_path = package.__path__
    for _, module_name, _ in pkgutil.walk_packages(
        package_path, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("compute_"):
                compute_functions.append((module_name, name, obj.__doc__))
    return compute_functions


def check_for_validation_test(
    full_function_name: str, validation_tests: list[tuple[str, str]]
) -> str:
    """Check if a validation test exists for a compute function

    Args:
        full_function_name: Compute function name
        validation_tests: List of validation tests

    Returns:
        Path to the validation test file or None if it doesn't exist
    """
    basefuncname = full_function_name.replace(".", "_").replace("compute_", "")
    ending = "_" + "_".join(basefuncname.split("_")[-2:])
    possible_test_name_endings = [ending, ending + "_unit", ending + "_validation"]
    for test, path in validation_tests:
        if test.endswith(tuple(possible_test_name_endings)):
            # Path relative to the `cdl` package:
            path = osp.relpath(path, start=osp.dirname(osp.join(tests_pkg.__file__)))
            return "/".join(path.split(osp.sep))
    return None


def get_validation_tests(package: str) -> list:
    """Retrieve list of validation tests from a package and its submodules

    Args:
        package: Python package

    Returns:
        List of tuples containing the test name and module path
    """
    validation_tests = []
    package_path = package.__path__
    for _, module_name, _ in pkgutil.walk_packages(
        package_path, package.__name__ + "."
    ):
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if hasattr(obj, "pytestmark"):
                for mark in obj.pytestmark:
                    if isinstance(mark, Mark) and mark.name == "validation":
                        module_path = inspect.getfile(obj)
                        validation_tests.append((name, module_path))
    return validation_tests


def generate_csv_files() -> None:
    """Generate CSV files containing the validation status of compute functions"""
    compute_functions = get_compute_functions(computation_pkg)
    validation_tests = get_validation_tests(tests_pkg)

    submodules = {"signal": [], "image": []}

    for module_name, function_name, docstring in compute_functions:
        if "signal" in module_name:
            submodules["signal"].append((module_name, function_name, docstring))
        elif "image" in module_name:
            submodules["image"].append((module_name, function_name, docstring))

    statistics_rows = []

    total_functions = {"signal": 0, "image": 0, "total": 0}
    validated_counts = {"signal": 0, "image": 0, "total": 0}

    for submodule, functions in submodules.items():
        function_rows = []
        for module_name, function_name, docstring in functions:
            full_function_name = f"{module_name}.{function_name}"
            test_path = check_for_validation_test(full_function_name, validation_tests)
            if test_path:
                validated_counts[submodule] += 1
                validated_counts["total"] += 1
            total_functions[submodule] += 1
            total_functions["total"] += 1
            description = docstring.split("\n")[0] if docstring else "-"
            test_script = f"``{test_path}``" if test_path else "N/A"
            short_name = function_name.replace("compute_", "")
            pyfunc_link = f":py:func:`{short_name} <{full_function_name}>`"
            function_rows.append([pyfunc_link, description, test_script])

        fname = osp.join(osp.dirname(__file__), f"validation_status_{submodule}.csv")
        with open(fname, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(function_rows)

        signal_percentage = (
            int((validated_counts["signal"] / total_functions["signal"]) * 100)
            if total_functions["signal"] > 0
            else 0
        )
        image_percentage = (
            int((validated_counts["image"] / total_functions["image"]) * 100)
            if total_functions["image"] > 0
            else 0
        )
        total_percentage = (
            int((validated_counts["total"] / total_functions["total"]) * 100)
            if total_functions["total"] > 0
            else 0
        )

    statistics_rows.append(
        [
            "Number of compute functions",
            total_functions["signal"],
            total_functions["image"],
            total_functions["total"],
        ]
    )
    statistics_rows.append(
        [
            "Number of validated compute functions",
            validated_counts["signal"],
            validated_counts["image"],
            validated_counts["total"],
        ]
    )
    statistics_rows.append(
        [
            "Percentage of validated compute functions",
            f"{signal_percentage}%",
            f"{image_percentage}%",
            f"{total_percentage}%",
        ]
    )

    fname = osp.join(osp.dirname(__file__), "validation_statistics.csv")
    with open(fname, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(statistics_rows)


if __name__ == "__main__":
    generate_csv_files()
