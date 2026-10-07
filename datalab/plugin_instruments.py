# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""Instruments exposed by application plugins as tools.

An instrument is a headless object: it holds settings (a guidata DataSet),
returns a live frame for these settings, and acquires objects. DataLab shows
it in a window with the live view on the left and the settings on the right.
DataLab Desktop and DataLab-Web share this module.
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Sequence

import guidata.dataset as gds
from sigima.objects import ImageObj, SignalObj

__all__ = ["InstrumentAcquisition", "InstrumentFrame", "PluginInstrument"]


def _validate_objects(objects: Sequence[SignalObj | ImageObj], name: str) -> tuple:
    """Return objects as a tuple of signals or of images."""
    objects = tuple(objects)
    if not objects:
        raise ValueError(f"{name} must contain at least one object")
    if not (
        all(isinstance(obj, SignalObj) for obj in objects)
        or all(isinstance(obj, ImageObj) for obj in objects)
    ):
        raise TypeError(f"{name} must contain signals only or images only")
    return objects


@dataclasses.dataclass(frozen=True)
class InstrumentFrame:
    """Live frame shown by DataLab for the current instrument settings.

    Args:
        objects: signals drawn together, or a single image
        summary: short text shown below the view (e.g. measured levels)
        value_range: fixed Y range of signals, or color range of the image
         (``None``: automatic)
    """

    objects: Sequence[SignalObj | ImageObj]
    summary: str = ""
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Validate the frame contents."""
        objects = _validate_objects(self.objects, "Instrument frame")
        if isinstance(objects[0], ImageObj) and len(objects) > 1:
            raise ValueError("Instrument frame must contain a single image")
        object.__setattr__(self, "objects", objects)
        if not isinstance(self.summary, str):
            raise TypeError("Instrument frame summary must be a string")
        if self.value_range is not None:
            low, high = (float(value) for value in self.value_range)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("Instrument frame range must be finite and increasing")
            object.__setattr__(self, "value_range", (low, high))


@dataclasses.dataclass(frozen=True)
class InstrumentAcquisition:
    """Objects acquired by an instrument, added by DataLab in a new group.

    Args:
        group_title: title of the group created for the acquisition
        objects: acquired signals, or acquired images
    """

    group_title: str
    objects: Sequence[SignalObj | ImageObj]

    def __post_init__(self) -> None:
        """Validate the acquisition contents."""
        if not isinstance(self.group_title, str) or not self.group_title.strip():
            raise ValueError("Instrument acquisition group title must not be empty")
        object.__setattr__(
            self, "objects", _validate_objects(self.objects, "Instrument acquisition")
        )


class PluginInstrument(abc.ABC):
    """Instrument opened by a plugin tool.

    DataLab writes the edited values into :attr:`settings` before calling
    :meth:`preview` or :meth:`acquire`. Both methods raise ``ValueError``
    with a user-facing message when the settings cannot be used. The plugin
    keeps one instance per tool during the session, so settings are kept
    from one opening to the next.

    Args:
        settings: instrument settings, edited by DataLab
    """

    #: Interval between two frames in live mode, in milliseconds
    live_interval_ms: int = 500

    def __init__(self, settings: gds.DataSet) -> None:
        if not isinstance(settings, gds.DataSet):
            raise TypeError("Instrument settings must be a guidata DataSet")
        self.settings = settings

    @abc.abstractmethod
    def preview(self) -> InstrumentFrame:
        """Return a live frame for the current settings."""

    @abc.abstractmethod
    def acquire(self) -> InstrumentAcquisition:
        """Acquire objects with the current settings."""
