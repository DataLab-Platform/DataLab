# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Print the dependencies of DataLab and SigimaX not installed from git clones.

Used by the ``install-dependencies`` action, in the DataLab checkout, after the
editable install of the cloned dependencies (SigimaX with ``--no-deps``). The
optional dependency groups of DataLab listed in the comma-separated ``EXTRAS``
environment variable (e.g. ``doc``) are included.
"""

from __future__ import annotations

import os

import tomli

#: Dependencies installed from their git clone (substring of the requirement)
CLONED = ("guidata", "PlotPy", "Sigima", "SigimaX", "datalab-capsule")


def main() -> None:
    """Print the requirements to install, one per line."""
    extras = [extra for extra in os.environ.get("EXTRAS", "").split(",") if extra]
    deps = []
    for path in ("pyproject.toml", "../SigimaX/pyproject.toml"):
        with open(path, "rb") as file:
            project = tomli.load(file)["project"]
        deps += project["dependencies"]
        if path == "pyproject.toml":
            for extra in extras:
                deps += project["optional-dependencies"][extra]
    for dep in deps:
        if not any(name in dep for name in CLONED):
            print(dep)


if __name__ == "__main__":
    main()
