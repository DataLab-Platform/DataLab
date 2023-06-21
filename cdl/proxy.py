# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
DataLab proxy module
--------------------

This module provides a way to access DataLab features from a proxy class.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import guidata.dataset.datatypes as gdt
import numpy as np

from cdl.core.baseproxy import BaseProxy
from cdl.core.remote import RemoteClient
from cdl.obj import ImageObj, SignalObj
from cdl.utils import qthelpers as qth


class RemoteCDLProxy(RemoteClient):
    """DataLab proxy class.

    This class provides access to DataLab features from a proxy class.

    Args:
        port (str, optional): XML-RPC port to connect to. If not specified,
            the port is automatically retrieved from DataLab configuration.
        timeout (float, optional): Timeout in seconds. Defaults to 5.0.
        retries (int, optional): Number of retries. Defaults to 10.

    Raises:
        CDLConnectionError: Unable to connect to DataLab
        ValueError: Invalid timeout (must be >= 0.0)
        ValueError: Invalid number of retries (must be >= 1)

    Examples:
        Here is a simple example of how to use RemoteCDLProxy in a Python script
        or in a Jupyter notebook:

        >>> from cdl.proxy import RemoteCDLProxy
        >>> proxy = RemoteCDLProxy()
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy.get_object_from_title("toto")
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy.get_object(0)
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy.get_object(0).data
        array([1., 2., 3.])
    """

    def __init__(
        self,
        port: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        super().__init__()
        self.connect(port, timeout, retries)


class CDLProxy(BaseProxy):
    """DataLab proxy class.

    This class provides access to DataLab features from a proxy class.

    Args:
        cdl (CDLMainWindow): CDLMainWindow instance.
    """

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title (str): Signal title
            xdata (np.ndarray): X data
            ydata (np.ndarray): Y data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.

        Returns:
            bool: True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """
        return self._cdl.add_signal(title, xdata, ydata, xunit, yunit, xlabel, ylabel)

    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title (str): Image title
            data (np.ndarray): Image data
            xunit (str, optional): X unit. Defaults to None.
            yunit (str, optional): Y unit. Defaults to None.
            zunit (str, optional): Z unit. Defaults to None.
            xlabel (str, optional): X label. Defaults to None.
            ylabel (str, optional): Y label. Defaults to None.
            zlabel (str, optional): Z label. Defaults to None.

        Returns:
            bool: True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        return self._cdl.add_image(
            title, data, xunit, yunit, zunit, xlabel, ylabel, zlabel
        )

    def calc(self, name: str, param: gdt.DataSet | None = None) -> gdt.DataSet:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param (gdt.DataSet, optional): Compute function parameter. Defaults to None.

        Returns:
            gdt.DataSet: Compute function result
        """
        return self._cdl.calc(name, param)

    def get_object_from_title(
        self, title: str, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from title

        Args:
            title (str): object
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            ValueError: if object not found
            ValueError: if panel not found
        """
        return self._cdl.get_object_from_title(title, panel)

    def get_object(
        self,
        index: int | None = None,
        group_index: int | None = None,
        panel: str | None = None,
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            index (int): Object index in current panel. Defaults to None.
            group_index (int, optional): Group index. Defaults to None.
            panel (str, optional): Panel name. Defaults to None.

        If ``index`` is not specified, returns the currently selected object.
        If ``group_index`` is not specified, return an object from the current group.
        If ``panel`` is not specified, return an object from the current panel.

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            IndexError: if object not found
        """
        return self._cdl.get_object(index, group_index, panel)

    def get_object_from_uuid(
        self, oid: str, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from uuid

        Args:
            oid (str): object uuid
            panel (str | None): panel name (valid values: "signal", "image").

        Returns:
            Union[SignalObj, ImageObj]: object

        Raises:
            ValueError: if object not found
            ValueError: if panel not found
        """
        return self._cdl.get_object_from_uuid(oid, panel)

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            items (list): annotation plot items
            refresh_plot (bool, optional): refresh plot. Defaults to True.
            panel (str | None): panel name (valid values: "signal", "image").
                If None, current panel is used.
        """
        self._cdl.add_annotations_from_items(items, refresh_plot, panel)

    # ----- Proxy specific methods ------------------------------------------------
    # (not available symetrically in AbstractCDLControl)

    def add_object(self, obj: SignalObj | ImageObj) -> None:
        """Add object to DataLab.

        Args:
            obj (SignalObj | ImageObj): Signal or image object
        """
        self._cdl.add_object(obj)


@contextmanager
def proxy_context(what: str) -> Generator[CDLProxy | RemoteCDLProxy, None, None]:
    """Context manager handling CDL proxy creation and destruction.

    Args:
        what (str): proxy type ("gui" or "remote")
            For remote proxy, the port can be specified as "remote:port"

    Yields:
        Generator[CDLProxy | RemoteCDLProxy, None, None]: proxy
            CDLProxy if what == "gui"
            RemoteCDLProxy if what == "remote" or "remote:port"

    Example:
        with proxy_context("gui") as proxy:
            proxy.add_signal(...)
    """
    assert what == "gui" or what.startswith("remote"), "Invalid proxy type"
    xmlrpcport = None
    if ":" in what:
        xmlrpcport = int(what.split(":")[1].strip())
    if what == "gui":
        # pylint: disable=import-outside-toplevel
        from cdl.core.gui.main import CDLMainWindow

        with qth.qt_app_context(exec_loop=True):
            try:
                win = CDLMainWindow()
                proxy = CDLProxy(win)
                win.show()
                yield proxy
            finally:
                pass
    else:
        try:
            proxy = RemoteCDLProxy(xmlrpcport)
            yield proxy
        finally:
            proxy.disconnect()
