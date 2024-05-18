import csv
import importlib
import inspect
import os.path as osp
import pkgutil

from _pytest.mark import Mark

import cdl
import cdl.core.computation as computation_package
import cdl.tests as tests_package


# Step 1: Retrieve List of `compute_` Functions
def get_compute_functions(package):
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


# Step 2: Check for Corresponding Validation Tests
def check_for_validation_test(function_name, validation_tests):
    test_name = "test_" + function_name.replace(".", "_")
    for test, path in validation_tests:
        if test_name in test or test_name.replace("compute_", "") in test:
            # Path relative to the `cdl` package:
            cdl_dir = osp.dirname(osp.join(cdl.__file__))
            return osp.relpath(path, start=cdl_dir)
    return None


# Step 3: Retrieve List of Validation Tests
def get_validation_tests(package):
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


# Step 4: Generate CSV Files
def generate_csv_files():
    compute_functions = get_compute_functions(computation_package)
    validation_tests = get_validation_tests(tests_package)

    validated_counts = {"base": 0, "signal": 0, "image": 0, "total": 0}
    total_functions = {"base": 0, "signal": 0, "image": 0, "total": 0}

    function_rows = [["Function", "Description", "Test Script"]]

    for module_name, function_name, docstring in compute_functions:
        full_function_name = f"{module_name}.{function_name}"
        short_function_name = ".".join(full_function_name.split(".")[-2:])
        test_path = check_for_validation_test(short_function_name, validation_tests)
        if test_path:
            validated_counts["total"] += 1
            if "base" in module_name:
                validated_counts["base"] += 1
            elif "signal" in module_name:
                validated_counts["signal"] += 1
            elif "image" in module_name:
                validated_counts["image"] += 1
        if "base" in module_name:
            total_functions["base"] += 1
        elif "signal" in module_name:
            total_functions["signal"] += 1
        elif "image" in module_name:
            total_functions["image"] += 1
        total_functions["total"] += 1
        description = (
            docstring.split("\n")[0] if docstring else "No description available"
        )
        test_script = test_path if test_path else "N/A"

        pyfunc_link = f":py:func:`{short_function_name} <{full_function_name}>`"
        function_rows.append([pyfunc_link, description, test_script])

    with open(
        osp.join(osp.dirname(__file__), "validation_status.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(function_rows)

    base_percentage = (
        (validated_counts["base"] / total_functions["base"]) * 100
        if total_functions["base"] > 0
        else 0
    )
    signal_percentage = (
        (validated_counts["signal"] / total_functions["signal"]) * 100
        if total_functions["signal"] > 0
        else 0
    )
    image_percentage = (
        (validated_counts["image"] / total_functions["image"]) * 100
        if total_functions["image"] > 0
        else 0
    )
    total_percentage = (
        (validated_counts["total"] / total_functions["total"]) * 100
        if total_functions["total"] > 0
        else 0
    )

    statistics_rows = [
        ["Category", "Base", "Signal", "Image", "Total"],
        [
            "Number of compute functions",
            total_functions["base"],
            total_functions["signal"],
            total_functions["image"],
            total_functions["total"],
        ],
        [
            "Number of validated compute functions",
            validated_counts["base"],
            validated_counts["signal"],
            validated_counts["image"],
            validated_counts["total"],
        ],
        [
            "Percentage of validated compute functions",
            f"{base_percentage:.2f}%",
            f"{signal_percentage:.2f}%",
            f"{image_percentage:.2f}%",
            f"{total_percentage:.2f}%",
        ],
    ]

    with open(
        osp.join(osp.dirname(__file__), "validation_statistics.csv"), "w", newline=""
    ) as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(statistics_rows)


# Main Execution
if __name__ == "__main__":
    generate_csv_files()
