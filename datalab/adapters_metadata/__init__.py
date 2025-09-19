# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Adapters for Sigima's TableResult and GeometryResult, providing features
for storing and retrieving those objects as metadata for DataLab's signal
and image objects.
"""

from .common import ResultData, create_resultdata_dict, show_resultdata
from .geometry_adapter import GeometryAdapter
from .table_adapter import TableAdapter

__all__ = [
    "GeometryAdapter",
    "TableAdapter",
    "ResultData",
    "create_resultdata_dict",
    "show_resultdata",
]
