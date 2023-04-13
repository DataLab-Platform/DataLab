Remote controlling
==================

CobraDataLab may be controlled remotely using the `XML-RPC`_ protocol which is
natively supported by Python. Remote controlling allows to access CobraDataLab
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

Some examples are provided to help implementing such a communication between your application and CobraDataLab:
  - See ``cdl.tests.remoteclient_app`` module
  - See ``cdl.tests.remoteclient_unit`` module

.. figure:: /images/shots/remote_control_test.png

    Screenshot of remote client application test (``cdl.tests.remoteclient_app``)

When using Python 3, you may directly use the `RemoteClient` class as in
examples cited above.

Here is a Python 2.7 reimplementation of this class:

.. literalinclude:: ../../cdl_remotecontrol_py27.py

.. _XML-RPC: https://docs.python.org/3/library/xmlrpc.html
