"""
Adapters for Sigima's TableResult and GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

# Import legacy module to provide backward compatibility with clean wrapper classes
from . import legacy  # noqa: F401
from .geometry_adapter import GeometryAdapter
from .table_adapter import TableAdapter

__all__ = ["GeometryAdapter", "TableAdapter"]
