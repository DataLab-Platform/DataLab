# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Portable declarations for plugin-provided welcome page tiles."""

from __future__ import annotations

import dataclasses

from datalab.plugin_resources import LOCAL_ID_PATTERN, split_package_resource

__all__ = ["WelcomeTile"]


@dataclasses.dataclass(frozen=True)
class WelcomeTile:
    """Welcome page tile exposed by an application plugin

    Args:
        id: plugin-local tile ID (lowercase letters, digits, ``.``, ``_``, ``-``)
        title: tile title
        description: short description shown below the title
        icon: ``package:path`` resource (SVG or bitmap) or DataLab icon file
         name; ``None`` falls back to the plugin icon
        launcher: name of the plugin method called when the tile is clicked;
         ``None`` opens the plugin page of the **Applications** catalog
    """

    id: str
    title: str
    description: str = ""
    icon: str | None = None
    launcher: str | None = None

    def __post_init__(self) -> None:
        """Validate identity, icon resource, and launcher name."""
        if not isinstance(self.id, str) or not LOCAL_ID_PATTERN.fullmatch(self.id):
            raise ValueError(
                "Welcome tile ID must contain lowercase letters, digits, '.', "
                "'_' or '-'"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("Welcome tile title must be a non-empty string")
        if not isinstance(self.description, str):
            raise TypeError("Welcome tile description must be a string")
        if self.icon is not None:
            if not isinstance(self.icon, str) or not self.icon.strip():
                raise ValueError("Welcome tile icon must be a non-empty string")
            if self.is_package_icon:
                split_package_resource(self.icon, "Welcome tile icon")
        if self.launcher is not None and (
            not isinstance(self.launcher, str) or not self.launcher.isidentifier()
        ):
            raise ValueError("Welcome tile launcher must be a method name")

    @property
    def is_package_icon(self) -> bool:
        """Return True if the icon is a ``package:path`` resource"""
        return self.icon is not None and ":" in self.icon
