#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Generate build manifest for DataLab executable

This script generates a JSON manifest file containing information about the build
environment, Python version, system information, and all installed packages with
their versions. This manifest is then included in the frozen executable and displayed
in the installation configuration viewer.
"""

import json
import platform
from datetime import datetime
from importlib.metadata import distributions


def generate_manifest(file_path="manifest.json"):
    """Generate a manifest file with build information and package list

    Args:
        file_path: Path to the output manifest JSON file (default: "manifest.json")
    """
    manifest = {
        "build_time": datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "packages": {dist.metadata["Name"]: dist.version for dist in distributions()},
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {file_path}")
    print(f"  Build time: {manifest['build_time']}")
    print(f"  Python version: {manifest['python_version']}")
    print(f"  System: {manifest['system']} {manifest['release']}")
    print(f"  Architecture: {manifest['architecture']}")
    print(f"  Total packages: {len(manifest['packages'])}")


if __name__ == "__main__":
    generate_manifest()
