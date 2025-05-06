# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Update processor methods CSV files
----------------------------------

The purpose of this script is to generate CSV files containing the list of compute
methods attached to the processor classes:

- :class:`cdl.core.gui.processor.signal.SignalProcessor`
- :class:`cdl.core.gui.processor.image.ImageProcessor`

Those methods are *almost* associated elementwise with the compute functions defined in
the :mod:`cdl.computation` package.
"""

from __future__ import annotations

import csv
import inspect
import os.path as osp

from cdl.core.gui.processor.image import ImageProcessor
from cdl.core.gui.processor.signal import SignalProcessor
from cdl.utils.strings import shorten_docstring


def get_compute_methods(klass: type) -> list[str]:
    """Retrieve list of `compute_` methods from a processor class

    Args:
        klass: Processor class

    Returns:
        List of tuples containing the function name, and docstring
    """
    compute_methods = []
    for name, obj in inspect.getmembers(klass, inspect.isfunction):
        if name.startswith("compute_") and name not in (
            "compute_1_to_0",
            "compute_1_to_1",
            "compute_1_to_n",
            "compute_n_to_1",
            "compute_2_to_1",
        ):
            compute_methods.append((name, obj.__doc__))
    return compute_methods


def generate_csv_files() -> None:
    """Generate CSV files containing the validation status of compute functions"""
    lengths = []
    for category, klass in (("signal", SignalProcessor), ("image", ImageProcessor)):
        rows = []
        methods = get_compute_methods(klass)
        lengths.append(len(methods))
        for name, docstring in methods:
            pyfunc_link = f":py:func:`~{klass.__module__}.{klass.__name__}.{name}`"
            rows.append([pyfunc_link, shorten_docstring(docstring)])
        fname = osp.join(osp.dirname(__file__), f"processor_methods_{category}.csv")
        with open(fname, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
    lengths.append(sum(lengths))

    fname = osp.join(osp.dirname(__file__), "processor_methods_nb.csv")
    with open(fname, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[str(length) for length in lengths]])

    # Print statistics:
    print("Processor methods:")
    print(f"  Signal: {lengths[0]}")
    print(f"  Image:  {lengths[1]}")
    print(f"  Total:  {lengths[2]}")
    print()


if __name__ == "__main__":
    generate_csv_files()
