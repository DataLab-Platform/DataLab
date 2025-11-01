# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for multiple table results of the same kind
==================================================

This test verifies that multiple table results of the same kind can coexist
on the same object.
"""

from __future__ import annotations

import numpy as np
import sigima.objects as sio
from sigima.objects.scalar import TableResult

from datalab.adapters_metadata import TableAdapter


def test_multiple_table_results():
    """Test that multiple table results with different titles can coexist."""
    # Create a simple signal object
    x = np.linspace(0, 10, 100)
    y = np.exp(-((x - 5) ** 2) / 2)
    obj = sio.create_signal("Test Signal", x, y)

    # Create two different table results (simulating different statistics)
    stats1 = TableResult(
        title="basic_stats",
        headers=["Mean", "Std", "Min", "Max"],
        data=[[5.0, 1.5, 0.0, 10.0]],
        roi_indices=None,
        attrs={"method": "basic"},
    )

    stats2 = TableResult(
        title="advanced_stats",
        headers=["Median", "Q1", "Q3", "IQR"],
        data=[[5.0, 3.5, 6.5, 3.0]],
        roi_indices=None,
        attrs={"method": "advanced"},
    )

    # Add both results to the object
    TableAdapter(stats1).add_to(obj)
    TableAdapter(stats2).add_to(obj)

    # Verify both results are stored
    results = list(TableAdapter.iterate_from_obj(obj))
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"

    # Verify we can retrieve both results by title
    titles = {adapter.title for adapter in results}
    assert "basic_stats" in titles, "Basic stats result not found"
    assert "advanced_stats" in titles, "Advanced stats result not found"

    print("✓ Multiple table results can coexist")


if __name__ == "__main__":
    test_multiple_table_results()
    print("\n✅ All tests passed!")
