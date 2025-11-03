# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Common functions for metadata adapters.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from guidata.qthelpers import exec_dialog
from guidata.widgets.dataframeeditor import DataFrameEditor
from sigima.objects import ImageObj, SignalObj

from datalab.adapters_metadata.base_adapter import BaseResultAdapter
from datalab.adapters_metadata.geometry_adapter import GeometryAdapter
from datalab.adapters_metadata.table_adapter import TableAdapter
from datalab.config import _
from datalab.objectmodel import get_short_id

if TYPE_CHECKING:
    from qtpy.QtWidgets import QWidget


@dataclasses.dataclass
class ResultData:
    """Result data associated to a shapetype"""

    # We now store adapted objects from the new architecture
    results: list[BaseResultAdapter] | None = None
    ylabels: list[str] | None = None
    short_ids: list[str] | None = None

    def __bool__(self) -> bool:
        """Return True if there are results stored"""
        return bool(self.results)

    @property
    def category(self) -> str:
        """Return category of results"""
        if not self.results:
            raise ValueError("No result available")
        return self.results[0].category

    @property
    def headers(self) -> list[str]:
        """Return headers of results"""
        if not self.results:
            raise ValueError("No result available")
        # Return the intersection of all headers
        headers = set(self.results[0].headers)
        if len(self.results) > 1:
            for adapter in self.results[1:]:
                headers.intersection_update(adapter.headers)
        return list(headers)

    def __post_init__(self):
        """Check and initialize fields"""
        if self.results is None:
            self.results = []
        if self.ylabels is None:
            self.ylabels = []
        if self.short_ids is None:
            self.short_ids = []

    def append(self, adapter: BaseResultAdapter, obj: SignalObj | ImageObj) -> None:
        """Append a result adapter

        Args:
            adapter: Adapter to append
            obj: Object associated to the adapter
        """
        # Check that the adapter is compatible with existing ones
        if self.results:
            if adapter.category != self.results[0].category:
                raise ValueError("Incompatible adapter category")
            if len(set(self.headers).intersection(set(adapter.headers))) == 0:
                raise ValueError("Incompatible adapter headers")
        self.results.append(adapter)
        df = adapter.to_dataframe()
        for i_row_res in range(len(df)):
            sid = get_short_id(obj)
            ylabel = f"{adapter.unique_key}({sid})"
            if "roi_index" in df.columns:
                i_roi = int(df.iloc[i_row_res]["roi_index"])
                roititle = ""
                if i_roi >= 0:
                    roititle = obj.roi.get_single_roi_title(i_roi)
                    ylabel += f"|{roititle}"
            self.ylabels.append(ylabel)
            self.short_ids.append(sid)


def create_resultdata_dict(
    objs: list[SignalObj | ImageObj],
) -> dict[str, ResultData]:
    """Return result data dictionary

    Args:
        objs: List of objects

    Returns:
        Result data dictionary: keys are result categories, values are ResultData
    """
    rdatadict: dict[str, ResultData] = {}
    for obj in objs:
        for adapter in list(GeometryAdapter.iterate_from_obj(obj)) + list(
            TableAdapter.iterate_from_obj(obj)
        ):
            rdata = rdatadict.setdefault(adapter.category, ResultData())
            rdata.append(adapter, obj)
    return rdatadict


def show_resultdata(parent: QWidget, rdata: ResultData, object_name: str = "") -> None:
    """Show result data in a DataFrame editor window

    Args:
        parent: Parent widget
        rdata: Result data to show
        object_name: Optional object name for the dialog
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # Generate dataframes with visible columns only
        # Use the object-level visible_only parameter for cleaner implementation
        dfs = [result.to_dataframe(visible_only=True) for result in rdata.results]

        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Add comparison columns if we have multiple results of the same kind
        if len(dfs) > 1:
            df = _add_comparison_columns_to_dataframe(df, rdata)

        # Remove roi_index column for display (not needed in the GUI)
        if "roi_index" in df.columns:
            df = df.drop(columns=["roi_index"])

        df.set_index(pd.Index(rdata.ylabels), inplace=True)
        dlg = DataFrameEditor(parent)
        dlg.setup_and_check(
            df,
            _("Results") + f" ({rdata.category})",
            readonly=True,
            add_title_suffix=False,
        )
        if object_name:
            dlg.setObjectName(object_name)
        dlg.resize(750, 300)
        exec_dialog(dlg)


def _add_comparison_columns_to_dataframe(
    df: pd.DataFrame, rdata: ResultData
) -> pd.DataFrame:
    """Add comparison columns to dataframe with ROI-aware grouping.

    For each original column, adds one comparison column showing the difference
    between the current row and the corresponding reference row.

    Args:
        df: Combined DataFrame with all results
        rdata: ResultData containing ylabels and short_ids

    Returns:
        DataFrame with comparison columns added (one Δ column per original column)
    """
    # Build signal/image groups:
    # list of (object_id, start_row, end_row) for each signal/image
    obj_groups = []
    current_obj_id = None
    group_start = 0

    for i, obj_id in enumerate(rdata.short_ids):
        if obj_id != current_obj_id:
            # New signal group
            if current_obj_id is not None:
                # Close previous group
                obj_groups.append((current_obj_id, group_start, i - 1))
            current_obj_id = obj_id
            group_start = i

    # Close the last group
    if current_obj_id is not None:
        obj_groups.append((current_obj_id, group_start, len(rdata.ylabels) - 1))

    # If we only have one signal group, no need for comparisons
    if len(obj_groups) <= 1:
        return df

    # Use the first signal group as reference
    reference_group = obj_groups[0]
    _ref_obj_id, reference_start, reference_end = reference_group

    # Get columns to compare (exclude roi_index)
    cols_to_compare = [col for col in df.columns if col != "roi_index"]

    # Create new dataframe with original columns plus one comparison column per
    # original column
    result_df = df.copy()

    # Add comparison columns - one per original column
    for col in cols_to_compare:
        comparison_col_name = f"Δ({col})"
        comparison_values = []

        # For each row in the entire dataframe
        for row_idx in range(len(df)):
            # Find which group this row belongs to
            row_group_idx = None
            for group_idx, (obj_id, start, end) in enumerate(obj_groups):
                if start <= row_idx <= end:
                    row_group_idx = group_idx
                    break

            if row_group_idx == 0:
                # Reference group - no comparison needed
                comparison_values.append("")
            elif row_group_idx is not None:
                # Non-reference group - calculate comparison with corresponding
                # reference row
                group_start = obj_groups[row_group_idx][1]
                ref_row_idx = reference_start + (row_idx - group_start)
                if ref_row_idx <= reference_end:
                    ref_val = df.iloc[ref_row_idx][col]
                    curr_val = df.iloc[row_idx][col]
                    comparison_values.append(
                        _compute_comparison_value(ref_val, curr_val)
                    )
                else:
                    comparison_values.append("")
            else:
                # Should not happen, but handle gracefully
                comparison_values.append("")

        # Insert comparison column right after the original column
        col_idx = result_df.columns.get_loc(col)
        result_df.insert(col_idx + 1, comparison_col_name, comparison_values)

    return result_df


def _compute_comparison_value(ref_val, curr_val) -> str:
    """Compute a comparison value between reference and current values.

    Args:
        ref_val: Reference value
        curr_val: Current value to compare

    Returns:
        String representation of the comparison
    """
    # Handle different data types
    if pd.isna(ref_val) or pd.isna(curr_val):
        return "N/A"
    elif isinstance(ref_val, str) or isinstance(curr_val, str):
        # String comparison
        return "=" if str(ref_val) == str(curr_val) else "≠"
    elif isinstance(ref_val, (int, float, np.integer, np.floating)) and isinstance(
        curr_val, (int, float, np.integer, np.floating)
    ):
        # Numeric comparison - show difference
        diff = curr_val - ref_val
        # For integers, check exact equality; for floats, use small tolerance
        if isinstance(ref_val, (int, np.integer)) and isinstance(
            curr_val, (int, np.integer)
        ):
            tolerance = 0
        else:
            tolerance = 1e-10

        if abs(diff) <= tolerance:
            return "="
        else:
            # Format the difference with appropriate sign
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.4g}"
    else:
        # Default comparison
        return "=" if ref_val == curr_val else "≠"


def resultadapter_to_html(
    adapter: BaseResultAdapter | list[BaseResultAdapter],
    obj: SignalObj | ImageObj,
    visible_only: bool = True,
    transpose_single_row: bool = True,
    **kwargs,
) -> str:
    """Convert a result adapter to HTML format

    Args:
        adapter: Adapter to convert, or list of adapters to concatenate
        obj: Object associated to the adapter
        visible_only: If True, include only visible headers based on display
         preferences. Default is False.
        transpose_single_row: If True, transpose the table when there's only one row
        **kwargs: Additional arguments passed to DataFrame.to_html()

    Returns:
        HTML representation of the adapter
    """
    if not isinstance(adapter, BaseResultAdapter) and not all(
        [isinstance(adp, BaseResultAdapter) for adp in adapter]
    ):
        raise ValueError(
            "Adapter must be a BaseResultAdapter "
            "or a list of BaseResultAdapter instances"
        )
    if isinstance(adapter, BaseResultAdapter):
        return adapter.to_html(
            obj=obj,
            visible_only=visible_only,
            transpose_single_row=transpose_single_row,
            **kwargs,
        )
    return "<hr>".join([resultadapter_to_html(res, obj) for res in adapter])
