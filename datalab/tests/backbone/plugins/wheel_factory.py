# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Build in-memory plugin wheels for installer unit tests."""

from __future__ import annotations

import io
import zipfile

PLUGIN_SOURCE = """\
from datalab.plugins import PluginBase, PluginInfo


class ExamplePlugin(PluginBase):
    PLUGIN_INFO = PluginInfo(id={plugin_id!r}, name={name!r}, version={version!r})

    def create_actions(self):
        pass
"""

_DEFAULT = object()


def wheel_filename(distribution: str, version: str) -> str:
    """Return the pure-Python wheel file name of a distribution."""
    return f"{distribution.replace('-', '_')}-{version}-py3-none-any.whl"


def make_plugin_wheel(
    *,
    distribution: str = "example-datalab-plugin",
    version: str = "1.2.0",
    package: str = "example_datalab_plugin",
    plugin_id: str = "org.example.wheel-plugin",
    entry_points: str | None = None,
    requires_dist: tuple[str, ...] = ("sigima>=1.1",),
    requires_python: str = ">=3.9",
    top_level: str | None | object = _DEFAULT,
    files: dict[str, str | bytes] | None = None,
    extra_files: dict[str, str | bytes] | None = None,
) -> bytes:
    """Return the bytes of a DataLab plugin wheel.

    By default, the wheel provides a real desktop plugin class declared in the
    ``datalab.plugins`` entry-point group.
    """
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    if entry_points is None:
        entry_points = (
            f"[datalab.plugins]\nexample_plugin = {package}.desktop:ExamplePlugin\n"
        )
    if top_level is _DEFAULT:
        top_level = f"{package}\n"
    if files is None:
        files = {
            f"{package}/__init__.py": "",
            f"{package}/desktop.py": PLUGIN_SOURCE.format(
                plugin_id=plugin_id, name=f"Plugin {plugin_id}", version=version
            ),
        }
    metadata = [
        "Metadata-Version: 2.3",
        f"Name: {distribution}",
        f"Version: {version}",
        "Summary: Example DataLab plugin",
        f"Requires-Python: {requires_python}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requires_dist)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in {**files, **(extra_files or {})}.items():
            archive.writestr(name, data)
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: DataLab tests\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/entry_points.txt", entry_points)
        if top_level is not None:
            archive.writestr(f"{dist_info}/top_level.txt", top_level)
    return buffer.getvalue()
