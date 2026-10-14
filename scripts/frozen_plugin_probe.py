# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Check that the DataLab executable loads a plugin installed from a file

The probe plugin, installed like a user plugin wheel, records which modules a
plugin can import and which distribution metadata are available, so that the
frozen build is checked from a plugin's point of view.

Usage (with ``DATALAB_PLUGIN_PROBE_MARKER`` set to the result file path)::

    python scripts/frozen_plugin_probe.py install
    DataLab.exe --unattended --delay 5000
    python scripts/frozen_plugin_probe.py verify
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import zipfile

MARKER_ENV = "DATALAB_PLUGIN_PROBE_MARKER"

#: Modules commonly imported by plugins
MODULES = (
    "numpy",
    "numpy.fft",
    "scipy.interpolate",
    "scipy.ndimage",
    "scipy.optimize",
    "scipy.signal",
    "scipy.special",
    "scipy.stats",
    "skimage.filters",
    "skimage.measure",
    "skimage.morphology",
    "pandas",
    "h5py",
    "guidata.dataset",
    "sigima.objects",
    "sigima.params",
    "sigima.proc.image",
    "sigima.proc.signal",
    "sigima.tools.image",
    "sigima.tools.signal",
    "datalab.plugins.examples",
    "datalab.plugins.instruments",
    "datalab.plugins.recipes",
    "datalab.plugins.tiles",
    "datalab.plugins.tools",
)

#: Distributions plugin wheels may declare as requirements
DISTRIBUTIONS = ("guidata", "numpy", "plotpy", "scipy", "sigima")

PLUGIN_SOURCE = f'''\
import importlib
import json
import os

from packaging.utils import canonicalize_name

from datalab.plugins import PluginBase, PluginInfo
from datalab.plugins.base import get_installed_plugin_store


class FrozenProbe(PluginBase):
    PLUGIN_INFO = PluginInfo(
        id="org.datalab.frozen-probe", name="Frozen build probe", version="1.0.0"
    )

    def create_actions(self):
        imports = {{}}
        for name in {MODULES!r}:
            try:
                importlib.import_module(name)
                imports[name] = "ok"
            except Exception as exc:  # pylint: disable=broad-except
                imports[name] = f"{{type(exc).__name__}}: {{exc}}"
        available = {{
            canonicalize_name(name): version
            for name, version in get_installed_plugin_store()
            .get_available_distributions()
            .items()
        }}
        result = {{
            "imports": imports,
            "distributions": {{
                name: available.get(name) for name in {DISTRIBUTIONS!r}
            }},
        }}
        with open(os.environ["{MARKER_ENV}"], "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
'''


def make_probe_wheel() -> tuple[str, bytes]:
    """Return the file name and content of the probe plugin wheel."""
    dist_info = "datalab_frozen_probe-1.0.0.dist-info"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("datalab_frozen_probe/__init__.py", "")
        archive.writestr("datalab_frozen_probe/plugin.py", PLUGIN_SOURCE)
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: datalab-frozen-probe\nVersion: 1.0.0\n"
            "Requires-Dist: datalab-platform\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            f"{dist_info}/entry_points.txt",
            "[datalab.plugins]\nprobe = datalab_frozen_probe.plugin:FrozenProbe\n",
        )
        archive.writestr(f"{dist_info}/top_level.txt", "datalab_frozen_probe\n")
    return "datalab_frozen_probe-1.0.0-py3-none-any.whl", buffer.getvalue()


def install() -> int:
    """Install the probe wheel in the installed plugin store of the profile."""
    # pylint: disable=import-outside-toplevel
    from datalab.plugins.base import get_installed_plugin_store

    filename, data = make_probe_wheel()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, filename)
        with open(path, "wb") as file:
            file.write(data)
        store = get_installed_plugin_store()
        store.install_wheel(path)
    print(f"Probe plugin installed in {store.root}")
    return 0


def verify() -> int:
    """Report the probe results and fail if the plugin was not fully usable."""
    marker = os.environ[MARKER_ENV]
    if not os.path.isfile(marker):
        print("The probe plugin was not loaded by DataLab", file=sys.stderr)
        return 1
    with open(marker, encoding="utf-8") as file:
        result = json.load(file)
    failures = [
        f"import {name}: {status}"
        for name, status in result["imports"].items()
        if status != "ok"
    ]
    failures.extend(
        f"no metadata for {name}"
        for name, version in result["distributions"].items()
        if version is None
    )
    print(json.dumps(result, indent=2))
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    COMMANDS = {"install": install, "verify": verify}
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"Usage: {sys.argv[0]} {{{'|'.join(COMMANDS)}}}")
    sys.exit(COMMANDS[sys.argv[1]]())
