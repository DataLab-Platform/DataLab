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

        # Add comparison rows if we have multiple results of the same kind
        if len(dfs) > 1:
            dfs, updated_ylabels = _add_comparison_rows_to_dataframes(dfs, rdata)
        else:
            updated_ylabels = rdata.ylabels

        df = pd.concat(dfs, ignore_index=True)

        # Remove roi_index column for display (not needed in the GUI)
        if "roi_index" in df.columns:
            df = df.drop(columns=["roi_index"])

        # Use updated ylabels that account for comparison rows
        df.set_index(pd.Index(updated_ylabels), inplace=True)
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


def _add_comparison_rows_to_dataframes(
    dfs: list[pd.DataFrame], rdata: ResultData
) -> tuple[list[pd.DataFrame], list[str]]:
    """Add comparison rows between multiple dataframes with ROI-aware grouping.

    This function groups rows by signal/image object (using object ID from ylabels),
    then adds comparison rows after each signal/image group.

    Args:
        dfs: List of DataFrames to add comparison rows to
        rdata: ResultData containing ylabels and short_ids

    Returns:
        Tuple of (modified DataFrames list, modified ylabels list)
    """
    if len(dfs) <= 1:
        return dfs, rdata.ylabels

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
        return dfs, rdata.ylabels

    # Use the first signal group as reference
    reference_group = obj_groups[0]
    reference_start, reference_end = reference_group[1], reference_group[2]

    # Collect rows for reference signal (used for all comparisons)
    combined_df = pd.concat(dfs, ignore_index=True)
    reference_rows = combined_df.iloc[reference_start : reference_end + 1]

    # Build result DataFrames and ylabels by processing each signal group
    result_dfs = []
    result_ylabels = []

    for group_idx, (obj_id, start, end) in enumerate(obj_groups):
        # Add the signal group rows
        group_rows = combined_df.iloc[start : end + 1]
        group_df = pd.DataFrame(group_rows.values, columns=group_rows.columns)
        result_dfs.append(group_df)

        # Add corresponding ylabels
        result_ylabels.extend(rdata.ylabels[start : end + 1])

        # Add comparison rows (except for the reference signal)
        if group_idx > 0:
            comparison_df = _create_comparison_dataframe(reference_rows, group_rows)
            result_dfs.append(comparison_df)

            # Add comparison ylabels
            for i in range(len(comparison_df)):
                # Use the same object ID as the current signal group
                base_ylabel = rdata.ylabels[start + i]
                ref_id = rdata.short_ids[reference_start]
                comparison_ylabel = f"Δ vs {ref_id}: {base_ylabel}"
                result_ylabels.append(comparison_ylabel)

    return result_dfs, result_ylabels


def _create_comparison_dataframe(
    reference_df: pd.DataFrame, current_df: pd.DataFrame
) -> pd.DataFrame:
    """Create a comparison dataframe showing differences between reference and current.

    Args:
        reference_df: Reference dataframe (first result)
        current_df: Current dataframe to compare against reference

    Returns:
        DataFrame with comparison values
    """
    # Create comparison dataframe with same structure
    comparison_data = []

    # Compare row by row
    min_rows = min(len(reference_df), len(current_df))
    for row_idx in range(min_rows):
        ref_row = reference_df.iloc[row_idx]
        curr_row = current_df.iloc[row_idx]
        comparison_row = {}

        for col in reference_df.columns:
            if col == "roi_index":
                comparison_row[col] = -999  # Special marker for comparison rows
                continue

            if col not in current_df.columns:
                comparison_row[col] = "N/A"
                continue

            ref_val = ref_row[col]
            curr_val = curr_row[col]

            # Handle different data types
            if pd.isna(ref_val) or pd.isna(curr_val):
                comparison_row[col] = "N/A"
            elif isinstance(ref_val, str) or isinstance(curr_val, str):
                # String comparison
                comparison_row[col] = "=" if str(ref_val) == str(curr_val) else "≠"
            elif isinstance(
                ref_val, (int, float, np.integer, np.floating)
            ) and isinstance(curr_val, (int, float, np.integer, np.floating)):
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
                    comparison_row[col] = "="
                else:
                    # Format the difference with appropriate sign
                    sign = "+" if diff > 0 else ""
                    comparison_row[col] = f"{sign}{diff:.4g}"
            else:
                # Default comparison
                comparison_row[col] = "=" if ref_val == curr_val else "≠"

        comparison_data.append(comparison_row)

    return pd.DataFrame(comparison_data, columns=reference_df.columns)


def resultadapter_to_html(
    adapter: BaseResultAdapter,
    obj: SignalObj | ImageObj,
    visible_only: bool = True,
    transpose_single_row: bool = True,
    **kwargs,
) -> str:
    """Convert a result adapter to HTML format

    Args:
        adapter: Adapter to convert
        obj: Object associated to the adapter
        visible_only: If True, include only visible headers based on display
         preferences. Default is False.
        transpose_single_row: If True, transpose the table when there's only one row
        **kwargs: Additional arguments passed to DataFrame.to_html()

    Returns:
        HTML representation of the adapter
    """
    return adapter.to_html(
        obj=obj,
        visible_only=visible_only,
        transpose_single_row=transpose_single_row,
        **kwargs,
    )
