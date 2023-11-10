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

import guidata.dataset as gds
import numpy as np

from cdl.core.baseproxy import BaseProxy
from cdl.core.remote import RemoteClient
from cdl.obj import ImageObj, SignalObj
from cdl.utils import qthelpers as qth


class RemoteCDLProxy(RemoteClient):
    """DataLab proxy class.

    This class provides access to DataLab features from a proxy class.

    Args:
        autoconnect (bool): Automatically connect to DataLab XML-RPC server.

    Raises:
        ConnectionRefusedError: Unable to connect to DataLab
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

    def __init__(self, autoconnect: bool = True) -> None:
        super().__init__()
        if autoconnect:
            self.connect()


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
            xdata (numpy.ndarray): X data
            ydata (numpy.ndarray): Y data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.

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
            data (numpy.ndarray): Image data
            xunit (str | None): X unit. Defaults to None.
            yunit (str | None): Y unit. Defaults to None.
            zunit (str | None): Z unit. Defaults to None.
            xlabel (str | None): X label. Defaults to None.
            ylabel (str | None): Y label. Defaults to None.
            zlabel (str | None): Z label. Defaults to None.

        Returns:
            bool: True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        return self._cdl.add_image(
            title, data, xunit, yunit, zunit, xlabel, ylabel, zlabel
        )

    def calc(self, name: str, param: gds.DataSet | None = None) -> gds.DataSet:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name (str): Compute function name
            param (guidata.dataset.DataSet | None): Compute function
            parameter. Defaults to None.

        Returns:
            guidata.dataset.DataSet: Compute function result
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
            group_index (int | None): Group index. Defaults to None.
            panel (str | None): Panel name. Defaults to None.

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
            refresh_plot (bool | None): refresh plot. Defaults to True.
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
    port = None
    if ":" in what:
        port = int(what.split(":")[1].strip())
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
            proxy = RemoteCDLProxy(autoconnect=False)
            proxy.connect(port)
            yield proxy
        finally:
            proxy.disconnect()
