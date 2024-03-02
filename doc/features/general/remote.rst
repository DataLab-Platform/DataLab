.. _ref-to-remote-control:

Remote controlling
==================

.. meta::
    :description: How to remote control DataLab, the open-source scientific data analysis and visualization platform
    :keywords: DataLab, remote control, XML-RPC, Python, IDE, Spyder, Jupyter, third-party application, signal, image, HDF5, calculation, processor

DataLab may be controlled remotely using the `XML-RPC`_ protocol which is
natively supported by Python (and many other languages). Remote controlling
allows to access DataLab main features from a separate process.

.. note::

    If you are looking for a lighweight alternative solution to remote control
    DataLab (i.e. without having to install the whole DataLab package and its
    dependencies on your environment), please have a look at the
    `DataLab Simple Client`_ package (`pip install cdlclient`).

.. _DataLab Simple Client: https://github.com/DataLab-Platform/DataLabSimpleClient

From an IDE
^^^^^^^^^^^

DataLab may be controlled remotely from an IDE (e.g. `Spyder`_ or any other
IDE, or even a Jupyter Notebook) that runs a Python script. It allows to
connect to a running DataLab instance, adds a signal and an image, and then
runs calculations. This feature is exposed by the `RemoteProxy` class that
is provided in module ``cdl.proxy``.

From a third-party application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DataLab may also be controlled remotely from a third-party application, for the
same purpose.

If the third-party application is written in Python 3, it may directly use the
`RemoteProxy` class as mentioned above. From another language, it is also
achievable, but it requires to implement a XML-RPC client in this language
using the same methods of proxy server as in the `RemoteProxy` class.

Data (signals and images) may also be exchanged between DataLab and the remote
client application, in both directions.

The remote client application may be written in any language that supports
XML-RPC. For example, it is possible to write a remote client application in
Python, Java, C++, C#, etc. The remote client application may be a graphical
application or a command line application.

The remote client application may be run on the same computer as DataLab or on
a different computer. In the latter case, the remote client application must
know the IP address of the computer running DataLab.

The remote client application may be run before or after DataLab. In the latter
case, the remote client application must try to connect to DataLab until it
succeeds.

Supported features
^^^^^^^^^^^^^^^^^^

Supported features are the following:

  - Switch to signal or image panel
  - Remove all signals and images
  - Save current session to a HDF5 file
  - Open HDF5 files into current session
  - Browse HDF5 file
  - Open a signal or an image from file
  - Add a signal
  - Add an image
  - Get object list
  - Run calculation with parameters

.. note::

    The signal and image objects are described on this section: :ref:`ref-to-model`.

Some examples are provided to help implementing such a communication
between your application and DataLab:

  - See module: ``cdl.tests.remoteclient_app``
  - See module: ``cdl.tests.remoteclient_unit``

.. figure:: /images/shots/remote_control_test.png

    Screenshot of remote client application test (``cdl.tests.remoteclient_app``)

Examples
^^^^^^^^

When using Python 3, you may directly use the `RemoteProxy` class as in
examples cited above or below.

Here is an example in Python 3 of a script that connects to a running DataLab
instance, adds a signal and an image, and then runs calculations (the cell
structure of the script make it convenient to be used in `Spyder`_ IDE):

.. literalinclude:: ../../remote_example.py

Here is a Python 2.7 reimplementation of this class:

.. literalinclude:: ../../remotecontrol_py27.py

Connection dialog
^^^^^^^^^^^^^^^^^

The DataLab package also provides a connection dialog that may be used
to connect to a running DataLab instance. It is exposed by the
:py:class:`cdl.widgets.connection.ConnectionDialog` class.

.. figure:: ../../images/shots/connect_dialog.png

    Screenshot of connection dialog (``cdl.widgets.connection.ConnectionDialog``)

Example of use:

.. literalinclude:: ../../../cdl/tests/features/control/connect_dialog.py

Public API: remote client
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cdl.core.remote.RemoteClient
    :inherited-members:

Public API: additional methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The remote control class methods (either using the proxy or the remote client)
may be completed with additional methods which are dynamically added at
runtime. This mechanism allows to access the methods of the "processor"
objects of DataLab.

.. automodule:: cdl.core.gui.processor.signal
    :members:

.. automodule:: cdl.core.gui.processor.image
    :members:


.. _XML-RPC: https://docs.python.org/3/library/xmlrpc.html

.. _Spyder: https://www.spyder-ide.org/