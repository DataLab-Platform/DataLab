# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Update validation status CSV files from compute functions and validation tests"""

from __future__ import annotations

import os.path as osp

from sigima.proc.validation import ValidationStatistics


def generate_csv_files() -> None:
    """Generate CSV files containing the validation status of compute functions"""
    path = osp.dirname(__file__)
    stats = ValidationStatistics()
    stats.collect_validation_status(verbose=True)
    stats.generate_csv_files(path)
    stats.generate_statistics_csv(path)


if __name__ == "__main__":
    generate_csv_files()
