"""
Results Management Service
==========================

This module provides a clean service interface for managing analysis results
on DataLab objects, replacing the need for monkey-patching Sigima objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.adapters_metadata import GeometryAdapter, TableAdapter

if TYPE_CHECKING:
    from sigima import ImageObj, SignalObj


class ResultsManager:
    """Service for managing analysis results on DataLab objects."""

    @staticmethod
    def delete_results(obj: ImageObj | SignalObj) -> None:
        """
        Delete all analysis results from an object's metadata.

        This function removes:
        - Geometry results (stored via GeometryAdapter)
        - Table results (stored via TableAdapter)
        - Any other analysis results stored in metadata

        Args:
            obj: The Sigima object to clean results from

        Note:
            This replaces the removed `delete_results` method from Sigima.
        """
        # Collect keys to remove to avoid modifying dict during iteration
        keys_to_remove = []

        # Find geometry result keys
        for key in obj.metadata.keys():
            if GeometryAdapter.match(key, None):
                # Find all related keys for this geometry result
                base_key = key[: -len(GeometryAdapter.ARRAY_SUFFIX)]
                for suffix in [
                    GeometryAdapter.ARRAY_SUFFIX,
                    GeometryAdapter.TITLE_SUFFIX,
                    GeometryAdapter.SHAPE_SUFFIX,
                    GeometryAdapter.ADDLABEL_SUFFIX,
                ]:
                    related_key = base_key + suffix
                    if related_key in obj.metadata:
                        keys_to_remove.append(related_key)
                # Also remove any attribute keys
                for meta_key in list(obj.metadata.keys()):
                    if (
                        meta_key.startswith(base_key + "_")
                        and meta_key not in keys_to_remove
                    ):
                        keys_to_remove.append(meta_key)

        # Find table result keys
        for key in obj.metadata.keys():
            if TableAdapter.match(key, None):
                # Find all related keys for this table result
                base_key = key[: -len(TableAdapter.ARRAY_SUFFIX)]
                for suffix in [
                    TableAdapter.ARRAY_SUFFIX,
                    TableAdapter.TITLE_SUFFIX,
                    TableAdapter.HEADERS_SUFFIX,
                    TableAdapter.LABELS_SUFFIX,
                ]:
                    related_key = base_key + suffix
                    if related_key in obj.metadata:
                        keys_to_remove.append(related_key)

        # Remove all collected keys
        for key in set(keys_to_remove):  # Use set to avoid duplicates
            if key in obj.metadata:
                del obj.metadata[key]

    @staticmethod
    def has_results(obj: ImageObj | SignalObj) -> bool:
        """
        Check if object has any analysis results.

        Args:
            obj: Object to check

        Returns:
            True if object has results, False otherwise
        """
        return any(
            GeometryAdapter.match(key, None) or TableAdapter.match(key, None)
            for key in obj.metadata.keys()
        )

    @staticmethod
    def get_results_count(obj: ImageObj | SignalObj) -> tuple[int, int]:
        """
        Get count of geometry and table results.

        Args:
            obj: Object to check

        Returns:
            Tuple of (geometry_count, table_count)
        """
        geometry_count = len(list(GeometryAdapter.iterate_from_obj(obj)))
        table_count = len(list(TableAdapter.iterate_from_obj(obj)))

        return geometry_count, table_count
