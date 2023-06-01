Remote controlling
==================

DataLab may be controlled remotely using the `XML-RPC`_ protocol which is
natively supported by Python. Remote controlling allows to access DataLab
main features from a separate process.

From an IDE
^^^^^^^^^^^

It is possible to run a Python script from an IDE (e.g. `Spyder`_ or any
other IDE, or even a Jupyter Notebook) that connects to a running DataLab
instance, adds a signal and an image, and then runs calculations. This is
the case of the `RemoteClient` class that is provided in module ``cdl.remote``.

From a third-party application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is also possible to connect to a running DataLab instance from a third-party
application, adds a signal and an image, and then runs calculations. This is
the case of the `RemoteClient` class that is provided in module ``cdl.remote``.

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

When using Python 3, you may directly use the `RemoteClient` class as in
examples cited above.

Here is an example in Python 3 of a script that connects to a running DataLab
instance, adds a signal and an image, and then runs calculations (the cell
structure of the script make it convenient to be used in `Spyder`_ IDE):

.. literalinclude:: ../../remote_example.py

Here is a Python 2.7 reimplementation of this class:

.. literalinclude:: ../../remotecontrol_py27.py

Public API: remote client
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: cdl.remotecontrol
    :members: RemoteClient

Public API: additional methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The remote control class methods may be completed with additional methods which
are dynamically added at runtime. This mechanism allows to access the methods
of the "processor" objects of DataLab.

.. automodule:: cdl.core.gui.processor.signal
    :members:

.. automodule:: cdl.core.gui.processor.image
    :members:


.. _XML-RPC: https://docs.python.org/3/library/xmlrpc.html

.. _Spyder: https://www.spyder-ide.org/