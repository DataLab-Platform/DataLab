# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapters registry unit test
---------------------------

Test the generic result-adapter resolver of :mod:`datalab.adapters_metadata`:
resolution of registered result typologies, subclass tolerance, error on
unsupported types and the public registration hook
(:func:`datalab.adapters_metadata.register_result_adapter`).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects.scalar import GeometryResult, KindShape, TableResult

from datalab.adapters_metadata import (
    GeometryAdapter,
    TableAdapter,
    create_adapter,
    register_result_adapter,
)
from datalab.adapters_metadata.common import _ADAPTER_REGISTRY
from datalab.env import execenv


def make_geometry_result(cls: type[GeometryResult] = GeometryResult) -> GeometryResult:
    """Build a minimal segment geometry result.

    Args:
        cls: Result class to instantiate (allows building subclasses).

    Returns:
        Geometry result instance.
    """
    return cls(
        title="fwhm",
        func_name="fwhm",
        kind=KindShape.SEGMENT,
        coords=np.array([[3.5, 0.6, 6.5, 0.6]]),
        roi_indices=None,
        attrs={},
    )


def make_table_result() -> TableResult:
    """Build a minimal statistics table result."""
    return TableResult(
        title="stats",
        func_name="stats",
        headers=["Mean", "Std"],
        data=[[5.0, 1.5]],
        roi_indices=None,
        attrs={},
    )


def test_create_adapter_resolves_registered_typologies() -> None:
    """Resolve GeometryResult and TableResult to their respective adapters."""
    geometry_adapter = create_adapter(make_geometry_result())
    assert isinstance(geometry_adapter, GeometryAdapter)
    table_adapter = create_adapter(make_table_result())
    assert isinstance(table_adapter, TableAdapter)
    execenv.print("test_create_adapter_resolves_registered_typologies: ✓")


def test_create_adapter_rejects_unsupported_type() -> None:
    """Raise TypeError naming the unsupported result type."""
    with pytest.raises(TypeError, match="dict"):
        create_adapter({"not": "a result"})
    execenv.print("test_create_adapter_rejects_unsupported_type: ✓")


def test_create_adapter_falls_back_to_isinstance() -> None:
    """Resolve an unregistered subclass through the isinstance fallback."""

    class CustomGeometryResult(GeometryResult):
        """Subclass without a dedicated adapter registration."""

    adapter = create_adapter(make_geometry_result(CustomGeometryResult))
    assert isinstance(adapter, GeometryAdapter)
    execenv.print("test_create_adapter_falls_back_to_isinstance: ✓")


def test_register_result_adapter_custom_typology() -> None:
    """Register a custom adapter and resolve the custom typology exactly."""

    class CustomGeometryResult(GeometryResult):
        """Custom result typology (e.g. contributed by a plugin)."""

    class CustomAdapter(GeometryAdapter):
        """Adapter dedicated to the custom typology."""

    register_result_adapter(CustomGeometryResult, CustomAdapter)
    try:
        adapter = create_adapter(make_geometry_result(CustomGeometryResult))
        assert isinstance(adapter, CustomAdapter)
        # Base typology resolution is unaffected by the extra registration
        assert isinstance(create_adapter(make_geometry_result()), GeometryAdapter)
    finally:
        _ADAPTER_REGISTRY.pop(CustomGeometryResult, None)
    execenv.print("test_register_result_adapter_custom_typology: ✓")


def test_create_adapter_fallback_prefers_most_specific_base() -> None:
    """Resolve an unregistered sub-subclass to the most specific adapter."""

    class CustomGeometryResult(GeometryResult):
        """Custom result typology with its own registered adapter."""

    class CustomAdapter(GeometryAdapter):
        """Adapter dedicated to the custom typology."""

    class UnregisteredSubResult(CustomGeometryResult):
        """Sub-subclass without a dedicated adapter registration."""

    register_result_adapter(CustomGeometryResult, CustomAdapter)
    try:
        adapter = create_adapter(make_geometry_result(UnregisteredSubResult))
        assert isinstance(adapter, CustomAdapter)
    finally:
        _ADAPTER_REGISTRY.pop(CustomGeometryResult, None)
    execenv.print("test_create_adapter_fallback_prefers_most_specific_base: ✓")


if __name__ == "__main__":
    test_create_adapter_resolves_registered_typologies()
    test_create_adapter_rejects_unsupported_type()
    test_create_adapter_falls_back_to_isinstance()
    test_register_result_adapter_custom_typology()
    test_create_adapter_fallback_prefers_most_specific_base()
