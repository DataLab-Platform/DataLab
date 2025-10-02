# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Common functions for metadata adapters.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import TYPE_CHECKING

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
            ylabel = f"{adapter.title}({get_short_id(obj)})"
            if "roi_index" in df.columns:
                i_roi = int(df.iloc[i_row_res]["roi_index"])
                roititle = ""
                if i_roi >= 0:
                    roititle = obj.roi.get_single_roi_title(i_roi)
                    ylabel += f"|{roititle}"
            self.ylabels.append(ylabel)


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
        dfs = [result.to_dataframe() for result in rdata.results]
        df = pd.concat(dfs, ignore_index=True)
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
