# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Update validation status CSV files from compute functions and validation tests"""

from __future__ import annotations

import csv
import importlib
import inspect
import os.path as osp
import pkgutil
import re

import sigima.computation
import sigima.tests as tests_pkg
from _pytest.mark import Mark
from sigima import __version__

from datalab.utils.strings import shorten_docstring


def check_for_validation_test(
    full_function_name: str, validation_tests: list[tuple[str, str]]
) -> str:
    """Check if a validation test exists for a compute function

    Args:
        full_function_name: Compute function name
        validation_tests: List of validation tests

    Returns:
        Text to be included in the CSV file or None if it doesn't exist
    """
    family, funcname = full_function_name.split(".")[-2:]  # "signal" or "image"
    shortname = funcname.removeprefix("compute_")
    endings = [shortname, shortname + "_unit", shortname + "_validation"]
    beginnings = ["test", f"test_{family}", f"test_{family[:3]}", f"test_{family[0]}"]
    names = [f"{beginning}_{ending}" for beginning in beginnings for ending in endings]
    stable_version = re.sub(r"\.?(post|dev|rc|b|a)\S*", "", __version__)
    for test, path, line_number in validation_tests:
        if test in names:
            # Path relative to the `datalab` package:
            path = osp.relpath(path, start=osp.dirname(osp.join(tests_pkg.__file__)))
            name = "/".join(path.split(osp.sep))
            link = f"https://github.com/DataLab-Platform/Sigima/blob/v{stable_version}/sigima/tests/{name}#L{line_number}"
            return f"`{test} <{link}>`_"
    return None


def get_validation_tests(package: str) -> list:
    """Retrieve list of validation tests from a package and its submodules

    Args:
        package: Python package

    Returns:
        List of tuples containing the test name, module path and line number
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
                        line_number = inspect.getsourcelines(obj)[1]
                        validation_tests.append((name, module_path, line_number))
    return validation_tests


def generate_csv_files() -> None:
    """Generate CSV files containing the validation status of compute functions"""
    compute_functions = sigima.computation.find_computation_functions(
        sigima.computation
    )
    validation_tests = get_validation_tests(tests_pkg)

    submodules = {"signal": [], "image": []}

    for modname, funcname, docstring in compute_functions:
        if "signal" in modname:
            submodules["signal"].append((modname, funcname, docstring))
        elif "image" in modname:
            submodules["image"].append((modname, funcname, docstring))

    statistics_rows = []

    t_count = {"signal": 0, "image": 0, "total": 0}
    v_count = {"signal": 0, "image": 0, "total": 0}

    for submodule, functions in submodules.items():
        function_rows = []
        for modname, funcname, docstring in functions:
            full_funcname = f"{modname}.{funcname}"
            test_link = check_for_validation_test(full_funcname, validation_tests)
            if test_link:
                v_count[submodule] += 1
                v_count["total"] += 1
            t_count[submodule] += 1
            t_count["total"] += 1
            description = shorten_docstring(docstring)
            test_script = test_link if test_link else "N/A"
            pyfunc_link = f":py:func:`{funcname} <{full_funcname}>`"
            function_rows.append([pyfunc_link, description, test_script])

        fname = osp.join(osp.dirname(__file__), f"validation_status_{submodule}.csv")
        with open(fname, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(function_rows)

        signal_pct = (
            int((v_count["signal"] / t_count["signal"]) * 100)
            if t_count["signal"] > 0
            else 0
        )
        image_pct = (
            int((v_count["image"] / t_count["image"]) * 100)
            if t_count["image"] > 0
            else 0
        )
        total_pct = (
            int((v_count["total"] / t_count["total"]) * 100)
            if t_count["total"] > 0
            else 0
        )

    statistics_rows.append(
        [
            "Number of compute functions",
            t_count["signal"],
            t_count["image"],
            t_count["total"],
        ]
    )
    statistics_rows.append(
        [
            "Number of validated compute functions",
            v_count["signal"],
            v_count["image"],
            v_count["total"],
        ]
    )
    statistics_rows.append(
        [
            "Percentage of validated compute functions",
            f"{signal_pct}%",
            f"{image_pct}%",
            f"{total_pct}%",
        ]
    )

    fname = osp.join(osp.dirname(__file__), "validation_statistics.csv")
    with open(fname, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(statistics_rows)

    # Print statistics:
    print("Validation statistics:")
    print(f"  Signal: {v_count['signal']}/{t_count['signal']} ({signal_pct}%)")
    print(f"  Image: {v_count['image']}/{t_count['image']} ({image_pct}%)")
    print(f"  Total: {v_count['total']}/{t_count['total']} ({total_pct}%)")
    print()


if __name__ == "__main__":
    generate_csv_files()
