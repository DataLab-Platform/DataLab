# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Unit tests for plugin wheel inspection, which never imports plugin code."""

from __future__ import annotations

import sys

import pytest

from datalab.plugins import wheels
from datalab.plugins.wheels import (
    DESKTOP_ENTRY_POINT_GROUP,
    WEB_ENTRY_POINT_GROUP,
    WheelInspectionError,
    inspect_wheel,
    is_compatible_wheel,
)
from datalab.tests.backbone.plugins.wheel_factory import (
    make_plugin_wheel,
    wheel_filename,
)

FILENAME = wheel_filename("example-datalab-plugin", "1.2.0")


def inspect(data: bytes, **kwargs) -> dict[str, object]:
    """Inspect a test wheel as the desktop host does."""
    options = {
        "filename": FILENAME,
        "entry_point_group": DESKTOP_ENTRY_POINT_GROUP,
        "available_distributions": {"sigima": "1.2.0"},
        "python_version": "3.11",
    }
    options.update(kwargs)
    return inspect_wheel(data, **options)


def test_inspection_reads_contract_without_importing_plugin() -> None:
    """The manifest is read from metadata, without executing plugin code."""
    sys.modules.pop("wheel_import_sentinel", None)
    data = make_plugin_wheel(
        files={
            "example_datalab_plugin/__init__.py": (
                "import sys\nsys.modules['wheel_import_sentinel'] = object()\n"
            ),
            "example_datalab_plugin/desktop.py": "class ExamplePlugin: pass\n",
        }
    )

    manifest = inspect(data)

    assert manifest["distribution"] == "example-datalab-plugin"
    assert manifest["version"] == "1.2.0"
    assert manifest["summary"] == "Example DataLab plugin"
    assert manifest["entry_points"] == [
        {
            "name": "example_plugin",
            "module": "example_datalab_plugin.desktop",
            "attribute": "ExamplePlugin",
        }
    ]
    assert manifest["top_level_packages"] == ["example_datalab_plugin"]
    assert manifest["size_bytes"] == len(data)
    assert "wheel_import_sentinel" not in sys.modules


def test_inspection_requires_the_host_entry_point_group() -> None:
    """A Web-only wheel is rejected by the desktop host, and conversely."""
    data = make_plugin_wheel(
        entry_points=(
            f"[{WEB_ENTRY_POINT_GROUP}]\n"
            "example_plugin = example_datalab_plugin.web:ExamplePlugin\n"
        )
    )

    with pytest.raises(WheelInspectionError, match="'datalab.plugins'"):
        inspect(data)
    manifest = inspect(data, entry_point_group=WEB_ENTRY_POINT_GROUP)
    assert manifest["entry_points"][0]["module"] == "example_datalab_plugin.web"


@pytest.mark.parametrize(
    ("extra_files", "top_level", "message"),
    [
        ({"../escape.py": b""}, None, "Unsafe path"),
        ({"example_datalab_plugin/native.pyd": b"binary"}, None, "Native payload"),
        ({"datalab/__init__.py": b""}, "datalab\n", "reserved package"),
        ({"qtpy/__init__.py": b""}, "qtpy\n", "reserved package"),
    ],
)
def test_inspection_rejects_unsafe_payloads(
    extra_files: dict[str, bytes], top_level: str | None, message: str
) -> None:
    """Traversal paths, native code and reserved package names are rejected."""
    options = {} if top_level is None else {"top_level": top_level}
    with pytest.raises(WheelInspectionError, match=message):
        inspect(make_plugin_wheel(extra_files=extra_files, **options))


def test_inspection_rejects_missing_host_dependency() -> None:
    """Dependencies must already be provided by the host."""
    data = make_plugin_wheel(requires_dist=("unknown-science-package>=1",))

    with pytest.raises(WheelInspectionError, match="not provided by DataLab$"):
        inspect(data)
    with pytest.raises(WheelInspectionError, match="not provided by DataLab-Web"):
        inspect(
            data,
            entry_point_group=DESKTOP_ENTRY_POINT_GROUP,
            host_name="DataLab-Web",
        )
    with pytest.raises(WheelInspectionError, match="incompatible installed version"):
        inspect(make_plugin_wheel(requires_dist=("sigima>=2",)))


def test_inspection_skips_dependencies_with_inapplicable_markers() -> None:
    """A requirement whose marker does not apply needs no host distribution."""
    manifest = inspect(
        make_plugin_wheel(requires_dist=('pyqt5>=5 ; sys_platform == "emscripten"',)),
        marker_environment={"sys_platform": "win32"},
    )

    (dependency,) = manifest["dependencies"]
    assert dependency["applies"] is False
    assert dependency["compatible"] is True


def test_inspection_checks_python_requirement() -> None:
    """The wheel Requires-Python must accept the host interpreter."""
    with pytest.raises(WheelInspectionError, match="requires Python >=3.12"):
        inspect(make_plugin_wheel(requires_python=">=3.12"))


@pytest.mark.parametrize(
    ("tag", "python_version"),
    [("cp313", "3.9"), ("py312", "3.11"), ("py2", "3.11"), ("cp39", "3.11")],
)
def test_inspection_rejects_wheels_built_for_another_python(
    tag: str, python_version: str
) -> None:
    """Pure-Python tags must target the host Python, even without Requires-Python."""
    filename = FILENAME.replace("py3-none-any", f"{tag}-none-any")
    data = make_plugin_wheel(requires_python=">=2.7")

    with pytest.raises(WheelInspectionError, match=f"built for {tag}, host has"):
        inspect(data, filename=filename, python_version=python_version)
    assert not is_compatible_wheel(filename, None, python_version)


@pytest.mark.parametrize("tag", ["py3", "py2.py3", "py39", "py311", "cp311"])
def test_inspection_accepts_wheels_for_the_host_python(tag: str) -> None:
    """Generic and host-specific pure-Python tags are accepted, as by pip."""
    filename = FILENAME.replace("py3-none-any", f"{tag}-none-any")

    assert inspect(make_plugin_wheel(), filename=filename)["tags"]
    assert is_compatible_wheel(filename, ">=3.9", "3.11")
    assert not is_compatible_wheel(filename, ">=3.12", "3.11")


def test_inspection_rejects_duplicate_dist_info_directories() -> None:
    """A wheel carries the metadata of a single distribution."""
    data = make_plugin_wheel(
        extra_files={"other_plugin-1.0.dist-info/METADATA": b"Name: other\n"}
    )

    with pytest.raises(WheelInspectionError, match="exactly one .dist-info"):
        inspect(data)


def test_inspection_rejects_oversized_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The archive size is bounded before it is opened."""
    data = make_plugin_wheel()
    monkeypatch.setattr(wheels, "MAX_WHEEL_BYTES", len(data) - 1)

    with pytest.raises(WheelInspectionError, match="size limit"):
        inspect(data)


def test_inspection_rejects_mismatched_or_native_filenames() -> None:
    """File name and metadata identify the same pure-Python distribution."""
    data = make_plugin_wheel()

    with pytest.raises(WheelInspectionError, match="pure-Python"):
        inspect(data, filename="example_datalab_plugin-1.2.0-cp311-cp311-win_amd64.whl")
    with pytest.raises(WheelInspectionError, match="version differ"):
        inspect(data, filename=wheel_filename("example-datalab-plugin", "1.3.0"))
    with pytest.raises(WheelInspectionError, match="must end with .whl"):
        inspect(data, filename="example_datalab_plugin-1.2.0.zip")


def test_inspection_derives_flat_module_layout_and_reserved_names() -> None:
    """Without top_level.txt, root modules and packages are derived."""
    manifest = inspect(
        make_plugin_wheel(
            entry_points="[datalab.plugins]\nflat_plugin = flat_plugin:FlatPlugin\n",
            top_level=None,
            files={"flat_plugin.py": b"class FlatPlugin: pass\n"},
        )
    )
    assert manifest["top_level_packages"] == ["flat_plugin"]

    with pytest.raises(WheelInspectionError, match="reserved package"):
        inspect(
            make_plugin_wheel(
                top_level=None,
                files={"sigima.py": b"class Impostor: pass\n"},
            )
        )


def test_inspection_ignores_data_directories_without_top_level() -> None:
    """Wheel ``.data`` payload directories are not importable packages."""
    manifest = inspect(
        make_plugin_wheel(
            top_level=None,
            extra_files={
                "example_datalab_plugin-1.2.0.data/scripts/run": b"#!/bin/sh\n"
            },
        )
    )

    assert manifest["top_level_packages"] == ["example_datalab_plugin"]


def test_inspection_treats_invalid_host_version_as_unavailable() -> None:
    """A non-PEP 440 host version cannot satisfy a requirement."""
    with pytest.raises(WheelInspectionError, match="not provided"):
        inspect(
            make_plugin_wheel(),
            available_distributions={"sigima": "not-a-version"},
        )


def test_inspection_rejects_stdlib_names_without_stdlib_module_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.9 has no ``sys.stdlib_module_names``: modules are located."""
    monkeypatch.delattr(sys, "stdlib_module_names", raising=False)
    data = make_plugin_wheel(top_level="json\n")

    with pytest.raises(WheelInspectionError, match="reserved package: 'json'"):
        inspect(data)
    assert inspect(make_plugin_wheel())["top_level_packages"] == [
        "example_datalab_plugin"
    ]


if __name__ == "__main__":
    pytest.main([__file__])
