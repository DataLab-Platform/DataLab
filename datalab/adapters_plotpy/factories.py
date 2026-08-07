# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
PlotPy Adapter Factories
------------------------
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

from __future__ import annotations

from sigimax.adapters_plotpy import PlotPyAdapterFactory

__all__ = ["DataLabPlotPyAdapterFactory"]


class DataLabPlotPyAdapterFactory(PlotPyAdapterFactory):
    """Adapter factory adding DataLab scalar results and metadata rendering.

    Signal and image objects use the DataLab adapters (which render geometry
    results stored in metadata); every other type falls back to SigimaX.
    """

    def get_adapter_class(self, object_to_adapt) -> type:
        """Return the adapter class for the given object.

        Args:
            object_to_adapt: The object to adapt (signal, image, ROI or scalar
             result)

        Returns:
            The adapter class

        Raises:
            TypeError: If the object type is not supported
        """
        # pylint: disable=import-outside-toplevel
        # Import adapters as needed to avoid circular imports
        from sigima.objects import ImageObj, SignalObj

        from datalab.adapters_metadata import GeometryAdapter, TableAdapter
        from datalab.adapters_plotpy.objects.adapters import (
            ImageObjPlotPyAdapter,
            SignalObjPlotPyAdapter,
        )
        from datalab.adapters_plotpy.objects.scalar import (
            GeometryPlotPyAdapter,
            TablePlotPyAdapter,
        )

        if isinstance(object_to_adapt, GeometryAdapter):
            return GeometryPlotPyAdapter
        if isinstance(object_to_adapt, TableAdapter):
            return TablePlotPyAdapter
        if isinstance(object_to_adapt, SignalObj):
            return SignalObjPlotPyAdapter
        if isinstance(object_to_adapt, ImageObj):
            return ImageObjPlotPyAdapter
        return super().get_adapter_class(object_to_adapt)
