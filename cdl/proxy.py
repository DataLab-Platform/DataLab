# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Proxy objects (:mod:`cdl.proxy`)
--------------------------------

The :mod:`cdl.proxy` module provides a way to access DataLab features from a proxy
class.

The list of compute methods accessible from the proxy objects is available in the
:ref:`processor_methods` section.

Remote proxy
^^^^^^^^^^^^

The remote proxy is used when DataLab is started from a different process than the
proxy. In this case, the proxy connects to DataLab XML-RPC server.

.. autoclass:: RemoteProxy
    :members:
    :inherited-members:

Local proxy
^^^^^^^^^^^

The local proxy is used when DataLab is started from the same process as the proxy.
In this case, the proxy is directly connected to DataLab main window instance. The
typical use case is high-level scripting.

.. autoclass:: LocalProxy
    :members:
    :inherited-members:

Proxy context manager
^^^^^^^^^^^^^^^^^^^^^

The proxy context manager is a convenient way to handle proxy creation and
destruction. It is used as follows:

.. code-block:: python

        with proxy_context("local") as proxy:
            proxy.add_signal(...)

The proxy type can be "local" or "remote". For remote proxy, the port can be
specified as "remote:port".

.. note:: The proxy context manager allows to use the proxy in various contexts
    (Python script, Jupyter notebook, etc.). It also allows to switch seamlessly
    between local and remote proxy, keeping the same code inside the context.

.. autofunction:: proxy_context

.. _processor_methods:

List of compute methods
^^^^^^^^^^^^^^^^^^^^^^^

All the proxy objects provide access to the DataLab computing methods exposed by
the processor classes:

- :class:`cdl.core.gui.processor.signal.SignalProcessor`
- :class:`cdl.core.gui.processor.image.ImageProcessor`

Number of compute methods
*************************

.. csv-table:: Number of compute methods
   :file: ../doc/processor_methods_nb.csv
   :header: Signal, Image, Total

Signal processing
*****************

The following table lists the signal processor methods - it is automatically
generated from the source code:

.. csv-table:: Signal processor methods
   :file: ../doc/processor_methods_signal.csv
   :header: Compute method, Description
   :widths: 40, 60

Image processing
****************

The following table lists the image processor methods - it is automatically
generated from the source code:

.. csv-table:: Image processor methods
    :file: ../doc/processor_methods_image.csv
    :header: Compute method, Description
    :widths: 40, 60
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


class RemoteProxy(RemoteClient):
    """DataLab remote proxy class.

    This class provides access to DataLab features from a proxy class. This is the
    remote version of proxy, which is used when DataLab is started from a different
    process than the proxy.

    Args:
        autoconnect (bool): Automatically connect to DataLab XML-RPC server.

    Raises:
        ConnectionRefusedError: Unable to connect to DataLab
        ValueError: Invalid timeout (must be >= 0.0)
        ValueError: Invalid number of retries (must be >= 1)

    .. note::

        The proxy object also allows to access DataLab computing methods exposed by
        the processor classes (see :ref:`processor_methods`).

    Examples:
        Here is a simple example of how to use RemoteProxy in a Python script
        or in a Jupyter notebook:

        >>> from cdl.proxy import RemoteProxy
        >>> proxy = RemoteProxy()
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy["toto"]  # from title
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1]  # from number
        <cdl.core.model.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1].data
        array([1., 2., 3.])
        >>> proxy.set_current_panel("image")
    """

    def __init__(self, autoconnect: bool = True) -> None:
        super().__init__()
        if autoconnect:
            self.connect()


class LocalProxy(BaseProxy):
    """DataLab local proxy class.

    This class provides access to DataLab features from a proxy class. This is the
    local version of proxy, which is used when DataLab is started from the same
    process as the proxy.

    Args:
        cdl (CDLMainWindow): CDLMainWindow instance.

    .. note::

        The proxy object also allows to access DataLab computing methods exposed by
        the processor classes (see :ref:`processor_methods`).
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

    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call compute function ``name`` in current panel's processor.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None.

        Raises:
            ValueError: unknown function
        """
        return self._cdl.calc(name, param)

    def get_object(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """
        return self._cdl.get_object(nb_id_title, panel)

    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object).
            panel: Panel name. Defaults to None (current panel).

        Returns:
            List of plot item shapes
        """
        return self._cdl.get_object_shapes(nb_id_title, panel)

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
def proxy_context(what: str) -> Generator[LocalProxy | RemoteProxy, None, None]:
    """Context manager handling CDL proxy creation and destruction.

    Args:
        what (str): proxy type ("local" or "remote")
         For remote proxy, the port can be specified as "remote:port"

    Yields:
        Generator[LocalProxy | RemoteProxy, None, None]: proxy
            LocalProxy if what == "local"
            RemoteProxy if what == "remote" or "remote:port"

    Example:
        with proxy_context("local") as proxy:
            proxy.add_signal(...)
    """
    assert what == "local" or what.startswith("remote"), "Invalid proxy type"
    port = None
    if ":" in what:
        port = int(what.split(":")[1].strip())
    if what == "local":
        # pylint: disable=import-outside-toplevel, cyclic-import
        from cdl.core.gui.main import CDLMainWindow

        with qth.cdl_app_context(exec_loop=True):
            try:
                win = CDLMainWindow()
                proxy = LocalProxy(win)
                win.show()
                yield proxy
            finally:
                pass
    else:
        try:
            proxy = RemoteProxy(autoconnect=False)
            proxy.connect(port)
            yield proxy
        finally:
            proxy.disconnect()
