# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
DataLab Macro templates
=======================

This package bundles ready-to-use macro examples that are surfaced in the
"New macro" dropdown of the Macro panel. Each template is a single ``.py``
file whose first comment line is parsed as the template description::

    # DataLab template: Short human-readable description

The rest of the file is loaded verbatim as the macro source code.

User-defined templates
----------------------

Additional templates can be dropped into the user templates directory
(by default ``<USER_CONFIG_DIR>/macro_templates`` — typically
``~/.DataLab/macro_templates`` on Linux/macOS and
``%APPDATA%\\.DataLab\\macro_templates`` on Windows). The location can be
overridden through ``Conf.macro_templates_path``.

Any ``*.py`` file placed in that directory is exposed in the "New macro"
dropdown after the bundled templates. Filenames starting with an
underscore are ignored.
"""

from __future__ import annotations

import os
import os.path as osp
import pkgutil
from dataclasses import dataclass

_TEMPLATE_FILES = (
    "simple_macro.py",
    "imageproc_macro.py",
    "call_method_macro.py",
)

_DESCRIPTION_TAG = "# DataLab template:"


@dataclass(frozen=True)
class MacroTemplate:
    """A bundled macro template.

    Attributes:
        name: Stable identifier (file stem, e.g. ``"simple_macro"``).
        title: Suggested macro title (used as the tab name).
        description: One-line description shown in the menu tooltip.
        code: Python source code (without the description tag line).
    """

    name: str
    title: str
    description: str
    code: str


def _parse(name: str, raw: str) -> MacroTemplate:
    """Parse a raw template file content into a MacroTemplate."""
    lines = raw.splitlines()
    description = ""
    if lines and lines[0].startswith(_DESCRIPTION_TAG):
        description = lines[0][len(_DESCRIPTION_TAG) :].strip()
        lines = lines[1:]
        # Strip leading blank lines after the tag
        while lines and not lines[0].strip():
            lines = lines[1:]
    title = description or name.replace("_", " ").title()
    code = "\n".join(lines).rstrip() + "\n"
    return MacroTemplate(name=name, title=title, description=description, code=code)


def _user_templates_dir() -> str | None:
    """Return the user templates directory path, or None if disabled."""
    # Imported lazily to avoid a circular import at module load time.
    try:
        from datalab.config import Conf
    except ImportError:
        return None
    try:
        return Conf.macro_templates_path.get(None)
    except Exception:  # pragma: no cover - defensive
        return None


def _load_user_templates() -> list[MacroTemplate]:
    """Load user-defined templates from the configured directory."""
    directory = _user_templates_dir()
    if not directory or not osp.isdir(directory):
        return []
    templates: list[MacroTemplate] = []
    seen: set[str] = set()
    try:
        filenames = sorted(os.listdir(directory))
    except OSError:
        return []
    for filename in filenames:
        if not filename.endswith(".py") or filename.startswith("_"):
            continue
        name = osp.splitext(filename)[0]
        if name in seen:
            continue
        seen.add(name)
        path = osp.join(directory, filename)
        try:
            with open(path, encoding="utf-8") as fdesc:
                raw = fdesc.read()
        except OSError:
            continue
        templates.append(_parse(name, raw))
    return templates


def list_templates() -> list[MacroTemplate]:
    """Return all macro templates (bundled first, then user-defined)."""
    templates: list[MacroTemplate] = []
    bundled_names: set[str] = set()
    for filename in _TEMPLATE_FILES:
        data = pkgutil.get_data(__name__, filename)
        if data is None:
            continue
        raw = data.decode("utf-8")
        name = osp.splitext(filename)[0]
        bundled_names.add(name)
        templates.append(_parse(name, raw))
    for template in _load_user_templates():
        if template.name in bundled_names:
            # Bundled templates take precedence; skip user file with same stem
            continue
        templates.append(template)
    return templates


def get_template(name: str) -> MacroTemplate | None:
    """Return a template by its stable name (file stem)."""
    for template in list_templates():
        if template.name == name:
            return template
    return None
