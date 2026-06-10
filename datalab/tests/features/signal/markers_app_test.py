# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Markers tables (XY/X/Y) application test.

Exercises the three :class:`sigima.objects.TableKind` markers variants:

- :attr:`~sigima.objects.TableKind.XY_MARKERS`: cross markers at ``(x, y)``
  (used here via the `extract_peak_positions` feature on the paracetamol
  spectrum).
- :attr:`~sigima.objects.TableKind.X_MARKERS`: vertical cursors at given
  abscissae (built manually from arbitrary X values).
- :attr:`~sigima.objects.TableKind.Y_MARKERS`: horizontal cursors at given
  ordinates (built manually from arbitrary Y values).
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from sigima.objects import SignalObj, TableKind, TableResult
from sigima.params import PeakDetectionParam
from sigima.tests.data import create_paracetamol_signal

from datalab.adapters_metadata import TableAdapter
from datalab.adapters_plotpy.objects.scalar import TablePlotPyAdapter
from datalab.tests import datalab_test_app_context


def __get_tables(obj: SignalObj) -> list[TableResult]:
    """Return all TableResult objects attached to `obj` as metadata."""
    return [adapter.result for adapter in TableAdapter.iterate_from_obj(obj)]


def __attach_table(obj: SignalObj, table: TableResult, func_name: str) -> None:
    """Attach an arbitrary TableResult to a signal object as metadata."""
    # `func_name` is required to compute the metadata key
    object.__setattr__(table, "func_name", func_name)
    TableAdapter(table).add_to(obj)


def __check_other_items(obj: SignalObj, table: TableResult, expected_count: int):
    """Render `table` via the PlotPy adapter and check produced items count."""
    items = TablePlotPyAdapter(TableAdapter(table)).get_other_items(obj)
    assert len(items) == expected_count, (
        f"Expected {expected_count} plot items for kind {table.kind}, got {len(items)}"
    )


def test_markers_app():
    """Markers application test (peak positions on paracetamol spectrum)."""
    with datalab_test_app_context(console=False) as win:
        panel = win.signalpanel

        sig0 = create_paracetamol_signal()

        # ------------------------------------------------------------------------------
        # MARK: XY_MARKERS test
        # ------------------------------------------------------------------------------

        sig1 = sig0.copy(title="XY markers test")
        panel.add_object(sig1)

        # Extract peak positions as an XY-markers table
        param = PeakDetectionParam.create(threshold=70, min_dist=5)
        panel.processor.run_feature("extract_peak_positions", param)

        tables = __get_tables(sig1)
        assert len(tables) == 1
        table = tables[0]
        assert table.kind == TableKind.XY_MARKERS
        assert table.is_xy_markers()
        # Headers reflect the source signal axis labels (with units when set);
        # paracetamol.txt provides ``2 theta (\u00b0)`` / ``Intensity``.
        x_header, y_header = table.headers[0], table.headers[1]
        assert x_header == "2 theta (\u00b0)"
        assert y_header == "Intensity"
        n_peaks = len(table.data)
        assert n_peaks > 0, "At least one peak should be detected"

        # All (x, y) couples must lie within the source signal range
        xs = table.col(x_header)
        ys = table.col(y_header)
        assert min(xs) >= sig1.x.min()
        assert max(xs) <= sig1.x.max()
        assert min(ys) >= sig1.y.min()
        assert max(ys) <= sig1.y.max()

        # XY_MARKERS rendering: one cross marker per row
        __check_other_items(sig1, table, expected_count=n_peaks)

        # panel.show_results()

        # ------------------------------------------------------------------------------
        # MARK: X_MARKERS test
        # ------------------------------------------------------------------------------

        sig2 = sig0.copy(title="X markers test")
        panel.add_object(sig2)

        # Arbitrary X positions (within the paracetamol spectrum range)
        x_positions = [20.0, 30.0, 50.0]
        table = TableResult.from_rows(
            title="X markers (arbitrary)",
            headers=["x"],
            rows=[[x] for x in x_positions],
            kind=TableKind.X_MARKERS,
            attrs={"show_row_index": True},
        )
        __attach_table(sig2, table, func_name="x_markers_test")

        retrieved = __get_tables(sig2)
        assert len(retrieved) == 1
        assert retrieved[0].kind == TableKind.X_MARKERS
        assert retrieved[0].is_x_markers()
        assert retrieved[0].col("x") == x_positions

        # X_MARKERS rendering: one vertical cursor per row
        __check_other_items(sig2, table, expected_count=len(x_positions))

        # panel.show_results()

        # ------------------------------------------------------------------------------
        # MARK: Y_MARKERS test
        # ------------------------------------------------------------------------------

        sig3 = sig0.copy(title="Y markers test")
        panel.add_object(sig3)

        # Arbitrary Y positions
        y_positions = [100.0, 250.0, 400.0]
        table = TableResult.from_rows(
            title="Y markers (arbitrary)",
            headers=["y"],
            rows=[[y] for y in y_positions],
            kind=TableKind.Y_MARKERS,
            attrs={"show_row_index": True},
        )
        __attach_table(sig3, table, func_name="y_markers_test")

        retrieved = __get_tables(sig3)
        assert len(retrieved) == 1
        assert retrieved[0].kind == TableKind.Y_MARKERS
        assert retrieved[0].is_y_markers()
        assert retrieved[0].col("y") == y_positions

        # Y_MARKERS rendering: one horizontal cursor per row
        __check_other_items(sig3, table, expected_count=len(y_positions))

        # panel.show_results()


if __name__ == "__main__":
    test_markers_app()
