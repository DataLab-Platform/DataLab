# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Inspect single-file plugins without importing or executing their code.

The public catalog keeps a verbatim copy of this Qt-free module. Only literal
strings and references to previously declared string constants are read.
"""

from __future__ import annotations

import ast
import hashlib
import os.path as osp
from typing import Any

from packaging.version import InvalidVersion, Version

MAX_MODULE_BYTES = 4 * 1024 * 1024
_FIELDS = {
    "id": "plugin_id",
    "name": "display_name",
    "version": "version",
    "description": "summary",
}


class ModuleInspectionError(ValueError):
    """A source file is not an acceptable single-file plugin artifact."""


def module_name(filename: str) -> str:
    """Return the importable name of a ``datalab_*.py`` file."""
    name, extension = osp.splitext(filename)
    if (
        "/" in filename
        or "\\" in filename
        or extension != ".py"
        or not name.startswith("datalab_")
        or len(name) <= len("datalab_")
        or not name.isidentifier()
    ):
        raise ModuleInspectionError(
            "A plugin module file name must look like datalab_<name>.py"
        )
    return name


def _string(node: ast.expr, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _assignment(node: ast.stmt) -> tuple[str, ast.expr] | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id, node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return node.target.id, node.value
    return None


def _metadata(tree: ast.Module) -> tuple[dict[str, str], list[str]]:
    constants: dict[str, str] = {}
    constructors: set[str] = set()
    namespaces: set[str] = set()
    records: list[dict[str, str]] = []
    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                bound = alias.asname or alias.name.split(".", maxsplit=1)[0]
                constants.pop(bound, None)
                constructors.discard(bound)
                namespaces.discard(bound)
        if isinstance(statement, ast.ImportFrom) and statement.module in (
            "datalab.plugins",
            "datalab.plugins.base",
        ):
            constructors.update(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name == "PluginInfo"
            )
        elif isinstance(statement, ast.Import):
            namespaces.update(
                alias.asname or alias.name
                for alias in statement.names
                if alias.name == "datalab.plugins" and alias.asname
            )
        assignment = _assignment(statement)
        if assignment is not None:
            name, value = assignment
            text = _string(value, constants)
            constants.pop(name, None)
            constructors.discard(name)
            namespaces.discard(name)
            if text is not None:
                constants[name] = text
        if not isinstance(statement, ast.ClassDef):
            if assignment is None and not isinstance(
                statement, (ast.Import, ast.ImportFrom)
            ):
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        constants.pop(node.id, None)
                        constructors.discard(node.id)
                        namespaces.discard(node.id)
            continue
        local = dict(constants)
        local_constructors = set(constructors)
        local_namespaces = set(namespaces)
        for member in statement.body:
            assignment = _assignment(member)
            if assignment is None:
                for node in ast.walk(member):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        local.pop(node.id, None)
                        local_constructors.discard(node.id)
                        local_namespaces.discard(node.id)
                continue
            name, value = assignment
            if name != "PLUGIN_INFO":
                text = _string(value, local)
                local.pop(name, None)
                local_constructors.discard(name)
                local_namespaces.discard(name)
                if text is not None:
                    local[name] = text
                continue
            record: dict[str, str] = {}
            if isinstance(value, ast.Call):
                function = value.func
                recognized = (
                    isinstance(function, ast.Name) and function.id in local_constructors
                ) or (
                    isinstance(function, ast.Attribute)
                    and function.attr == "PluginInfo"
                    and isinstance(function.value, ast.Name)
                    and function.value.id in local_namespaces
                )
                if recognized and not any(
                    keyword.arg is None for keyword in value.keywords
                ):
                    for keyword in value.keywords:
                        if keyword.arg in _FIELDS:
                            text = _string(keyword.value, local)
                            if text is not None:
                                record[_FIELDS[keyword.arg]] = text
            records.append(record)
    warnings: list[str] = []
    if len(records) != 1:
        warnings.append(
            "Plugin metadata is not unique or not statically readable; "
            "the catalog describes the module as a whole"
        )
        return {}, warnings
    record = records[0]
    if "version" in record and record["version"]:
        try:
            Version(record["version"])
        except InvalidVersion:
            warnings.append("The declared version is not comparable; using the digest")
            record.pop("version")
    if not all(record.get(field) for field in ("plugin_id", "display_name", "version")):
        warnings.append(
            "Some plugin metadata is unavailable without executing the file"
        )
    return record, warnings


def inspect_module(path_or_bytes: str | bytes, *, filename: str) -> dict[str, Any]:
    """Return file facts and safely readable plugin metadata.

    Valid Python is not necessarily a working or trustworthy plugin. Missing
    metadata stays unknown; callers may supply it explicitly in a catalog.
    """
    name = module_name(filename)
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes, "rb") as stream:
            data = stream.read(MAX_MODULE_BYTES + 1)
    else:
        data = path_or_bytes
    if not data:
        raise ModuleInspectionError("Plugin module is empty")
    if len(data) > MAX_MODULE_BYTES:
        raise ModuleInspectionError(
            f"Plugin module exceeds the {MAX_MODULE_BYTES} byte size limit"
        )
    try:
        tree = ast.parse(data, filename=filename)
        # Parsing alone accepts e.g. a return outside a function.
        compile(tree, filename, "exec")
    except (SyntaxError, ValueError, UnicodeError, RecursionError) as exc:
        raise ModuleInspectionError(f"Invalid Python plugin {filename}: {exc}") from exc
    metadata, warnings = _metadata(tree)
    return {
        "filename": filename,
        "name": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "version": metadata.get("version", ""),
        "plugin_id": metadata.get("plugin_id", ""),
        "display_name": metadata.get("display_name", ""),
        "summary": metadata.get("summary", ""),
        "warnings": warnings,
    }


def apply_catalog_metadata(
    manifest: dict[str, Any],
    *,
    plugin_id: str,
    name: str,
    version: str,
    summary: str = "",
) -> dict[str, Any]:
    """Supplement unknown metadata, refusing a known identity/version conflict."""
    if version:
        Version(version)
    for field, expected in (("plugin_id", plugin_id), ("version", version)):
        actual = manifest[field]
        if (
            actual
            and expected
            and (
                Version(actual) != Version(expected)
                if field == "version"
                else actual != expected
            )
        ):
            raise ModuleInspectionError(
                f"Catalog {field} {expected!r} differs from the file: {actual!r}"
            )
    return {
        **manifest,
        "plugin_id": plugin_id,
        "display_name": name,
        "version": version or manifest["version"],
        "summary": summary or manifest["summary"],
    }
