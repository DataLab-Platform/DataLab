# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Proxy objects (:mod:`datalab.control.proxy`)
--------------------------------------------

The :mod:`datalab.control.proxy` module provides a way to access DataLab features from
a proxy class.

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

Calling processor methods using proxy objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All the proxy objects provide access to the DataLab computing methods exposed by
the processor classes:

- :class:`datalab.gui.processor.signal.SignalProcessor`
- :class:`datalab.gui.processor.image.ImageProcessor`

To run a computation feature associated to a processor, you can use the
:meth:`calc` method of the proxy object:

.. code-block:: python

    # Call a method without parameter
    proxy.calc("average")

    # Call a method with parameters
    p = sigima.params.MovingAverageParam.create(n=30)
    proxy.calc("moving_average", p)

"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import guidata.dataset as gds
import numpy as np
from sigima import ImageObj, SignalObj

from datalab.control.baseproxy import BaseProxy
from datalab.control.remote import RemoteClient
from datalab.utils import qthelpers as qth


class RemoteProxy(RemoteClient):
    """DataLab remote proxy class.

    This class provides access to DataLab features from a proxy class. This is the
    remote version of proxy, which is used when DataLab is started from a different
    process than the proxy.

    Args:
        autoconnect: Automatically connect to DataLab XML-RPC server.

    Raises:
        ConnectionRefusedError: Unable to connect to DataLab
        ValueError: Invalid timeout (must be >= 0.0)
        ValueError: Invalid number of retries (must be >= 1)

    Examples:
        Here is a simple example of how to use RemoteProxy in a Python script
        or in a Jupyter notebook:

        >>> from datalab.control.proxy import RemoteProxy
        >>> proxy = RemoteProxy()
        Connecting to DataLab XML-RPC server...OK (port: 28867)
        >>> proxy.get_version()
        '1.0.0'
        >>> proxy.add_signal("toto", np.array([1., 2., 3.]), np.array([4., 5., -1.]))
        True
        >>> proxy.get_object_titles()
        ['toto']
        >>> proxy["toto"]  # from title
        <sigima.objects.signal.SignalObj at 0x7f7f1c0b4a90>
        >>> proxy[1]  # from number
        <sigima.objects.signal.SignalObj at 0x7f7f1c0b4a90>
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
        datalab (DLMainWindow): DLMainWindow instance.
    """

    def add_signal(
        self,
        title: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add signal data to DataLab.

        Args:
            title: Signal title
            xdata: X data
            ydata: Y data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            group_id: group id in which to add the signal. Defaults to ""
            set_current: if True, set the added signal as current

        Returns:
            True if signal was added successfully, False otherwise

        Raises:
            ValueError: Invalid xdata dtype
            ValueError: Invalid ydata dtype
        """
        return self._datalab.add_signal(
            title, xdata, ydata, xunit, yunit, xlabel, ylabel, group_id, set_current
        )

    def add_image(
        self,
        title: str,
        data: np.ndarray,
        xunit: str = "",
        yunit: str = "",
        zunit: str = "",
        xlabel: str = "",
        ylabel: str = "",
        zlabel: str = "",
        group_id: str = "",
        set_current: bool = True,
    ) -> bool:  # pylint: disable=too-many-arguments
        """Add image data to DataLab.

        Args:
            title: Image title
            data: Image data
            xunit: X unit. Defaults to ""
            yunit: Y unit. Defaults to ""
            zunit: Z unit. Defaults to ""
            xlabel: X label. Defaults to ""
            ylabel: Y label. Defaults to ""
            zlabel: Z label. Defaults to ""
            group_id: group id in which to add the image. Defaults to ""
            set_current: if True, set the added image as current

        Returns:
            True if image was added successfully, False otherwise

        Raises:
            ValueError: Invalid data dtype
        """
        return self._datalab.add_image(
            title,
            data,
            xunit,
            yunit,
            zunit,
            xlabel,
            ylabel,
            zlabel,
            group_id,
            set_current,
        )

    def add_object(
        self, obj: SignalObj | ImageObj, group_id: str = "", set_current: bool = True
    ) -> None:
        """Add object to DataLab.

        Args:
            obj: Signal or image object
            group_id: group id in which to add the object. Defaults to ""
            set_current: if True, set the added object as current
        """
        self._datalab.add_object(obj, group_id, set_current)

    def calc(self, name: str, param: gds.DataSet | None = None) -> None:
        """Call computation feature ``name``

        .. note::

            This calls either the processor's ``compute_<name>`` method (if it exists),
            or the processor's ``<name>`` computation feature (if it is registered,
            using the ``run_feature`` method).
            It looks for the function in all panels, starting with the current one.

        Args:
            name: Compute function name
            param: Compute function parameter. Defaults to None

        Raises:
            ValueError: unknown function
        """
        return self._datalab.calc(name, param)

    def get_object(
        self, nb_id_title: int | str | None = None, panel: str | None = None
    ) -> SignalObj | ImageObj:
        """Get object (signal/image) from index.

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object)
            panel: Panel name. Defaults to None (current panel)

        Returns:
            Object

        Raises:
            KeyError: if object not found
        """
        return self._datalab.get_object(nb_id_title, panel)

    def get_object_shapes(
        self,
        nb_id_title: int | str | None = None,
        panel: str | None = None,
    ) -> list:
        """Get plot item shapes associated to object (signal/image).

        Args:
            nb_id_title: Object number, or object id, or object title.
             Defaults to None (current object)
            panel: Panel name. Defaults to None (current panel)

        Returns:
            List of plot item shapes
        """
        return self._datalab.get_object_shapes(nb_id_title, panel)

    def add_annotations_from_items(
        self, items: list, refresh_plot: bool = True, panel: str | None = None
    ) -> None:
        """Add object annotations (annotation plot items).

        Args:
            itemsTrue: annotation plot items
            refresh_plotTrue: refresh plot. Defaults to True
            panel: panel name (valid values: "signal", "image").
             If None, current panel is used
        """
        self._datalab.add_annotations_from_items(items, refresh_plot, panel)


@contextmanager
def proxy_context(what: str) -> Generator[LocalProxy | RemoteProxy, None, None]:
    """Context manager handling DL proxy creation and destruction.

    Args:
        what: proxy type ("local" or "remote")
         For remote proxy, the port can be specified as "remote:port"

    Yields:
        proxy
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
        from datalab.gui.main import DLMainWindow

        with qth.datalab_app_context(exec_loop=True):
            try:
                win = DLMainWindow()
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
