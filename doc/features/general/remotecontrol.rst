Remote controlling
==================

DataLab may be controlled remotely using the `XML-RPC`_ protocol which is
natively supported by Python. Remote controlling allows to access DataLab
main features from a separate process.

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

Some examples are provided to help implementing such a communication between your application and DataLab:
  - See module: ``cdl.tests.remoteclient_app``
  - See module: ``cdl.tests.remoteclient_unit``

.. figure:: /images/shots/remote_control_test.png

    Screenshot of remote client application test (``cdl.tests.remoteclient_app``)

When using Python 3, you may directly use the `RemoteClient` class as in
examples cited above.

Here is an example in Python 3 of a script that connects to a running DataLab
instance, adds a signal and an image, and then runs calculations (the cell
structure of the script make it convenient to be used in `Spyder`_ IDE):

.. literalinclude:: ../../remote_example.py

Here is a Python 2.7 reimplementation of this class:

.. literalinclude:: ../../remotecontrol_py27.py

.. _XML-RPC: https://docs.python.org/3/library/xmlrpc.html

.. _Spyder: https://www.spyder-ide.org/