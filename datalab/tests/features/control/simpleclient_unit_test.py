# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Simple Remote client test
-------------------------

This test covers the simple remote client functionalities as provided by the
`sigima.client` module.

In Sigima, the `sigima.tests.common.client_unit_test` module provides a set of unit
tests for the client functionalities.

The purpose of these tests is to ensure that the client can correctly interact with
the real server, handling various scenarios and edge cases.

The tests include:

- A function comparing the list of methods implemented by the real server
  (DataLab's `RemoteServer` class) to those implemented by the stub server
  (Sigima's `DataLabStubServer` class).
- A function comparing the list of methods implemented by the full client (DataLab's
  `RemoteProxy` class) to those implemented by the simple client (Sigima's
  `SimpleRemoteProxy` class).
- A function that simply runs the `sigima.tests.common.client_unit_test` suite after
  having launched a real server instance (DataLab application, using the
  `datalab.tests.run_datalab_in_background` function), ensuring that this test passes
  with the stub server as well as with the real server.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: skip

import inspect

from guidata.env import execenv
from sigima.client.remote import SimpleRemoteProxy
from sigima.client.stub import DataLabStubServer
from sigima.tests.common.client_unit_test import RemoteClientTester

from datalab.control.baseproxy import AbstractDLControl
from datalab.control.proxy import RemoteProxy
from datalab.tests import run_datalab_in_background


def test_compare_server_methods() -> None:
    """Compare methods implemented by RemoteServer vs DataLabStubServer.

    This function compares the list of methods implemented by the real server
    (DataLab's RemoteServer class) to those implemented by the stub server
    (Sigima's DataLabStubServer class).
    """
    execenv.print("\n=== Comparing Server Methods ===")

    # Get methods from RemoteServer (the real DataLab server)
    # RemoteServer implements AbstractDLControl methods
    remote_server_methods = set(AbstractDLControl.get_public_methods())

    # Get methods from DataLabStubServer (the stub server)
    stub_server_methods = {
        name
        for name, method in inspect.getmembers(DataLabStubServer, inspect.isfunction)
        if not name.startswith("_")
    }

    execenv.print(f"RemoteServer methods: {len(remote_server_methods)}")
    execenv.print(f"DataLabStubServer methods: {len(stub_server_methods)}")

    # Methods in RemoteServer but not in DataLabStubServer
    missing_in_stub = remote_server_methods - stub_server_methods
    if missing_in_stub:
        execenv.print("\n⚠️  Methods in RemoteServer but missing in DataLabStubServer:")
        for method in sorted(missing_in_stub):
            execenv.print(f"  - {method}")

    # Methods in DataLabStubServer but not in RemoteServer
    extra_in_stub = stub_server_methods - remote_server_methods
    if extra_in_stub:
        execenv.print("\n🔍 Methods in DataLabStubServer but not in RemoteServer:")
        for method in sorted(extra_in_stub):
            execenv.print(f"  - {method}")

    # Common methods
    common_methods = remote_server_methods & stub_server_methods
    execenv.print(f"\n✅ Common methods: {len(common_methods)}")

    # Assert that stub server implements all required methods
    assert not missing_in_stub, (
        f"DataLabStubServer is missing {len(missing_in_stub)} methods "
        f"implemented by RemoteServer: {sorted(missing_in_stub)}"
    )

    execenv.print("✨ Server method comparison completed successfully!")


def test_compare_client_methods() -> None:
    """Compare methods implemented by RemoteProxy vs SimpleRemoteProxy.

    This function compares the list of methods implemented by the full client
    (DataLab's RemoteProxy class) to those implemented by the simple client
    (Sigima's SimpleRemoteProxy class).
    """
    execenv.print("\n=== Comparing Client Methods ===")

    # Get public methods from RemoteProxy (the full client)
    remote_proxy_methods = {
        name
        for name, method in inspect.getmembers(RemoteProxy, inspect.ismethod)
        if not name.startswith("_") and callable(method)
    }

    # Get public methods from SimpleRemoteProxy (the simple client)
    simple_proxy_methods = {
        name
        for name, method in inspect.getmembers(SimpleRemoteProxy, inspect.ismethod)
        if not name.startswith("_") and callable(method)
    }

    execenv.print(f"RemoteProxy methods: {len(remote_proxy_methods)}")
    execenv.print(f"SimpleRemoteProxy methods: {len(simple_proxy_methods)}")

    # Methods in SimpleRemoteProxy but not in RemoteProxy
    extra_in_simple = simple_proxy_methods - remote_proxy_methods
    if extra_in_simple:
        execenv.print(
            "\n🔍 Methods in SimpleRemoteProxy but not in RemoteProxy "
            "(expected, as SimpleRemoteProxy is a subset):"
        )
        for method in sorted(extra_in_simple):
            execenv.print(f"  - {method}")

    # Methods in RemoteProxy but not in SimpleRemoteProxy
    missing_in_simple = remote_proxy_methods - simple_proxy_methods
    if missing_in_simple:
        execenv.print(
            "\n📝 Methods in RemoteProxy but not in SimpleRemoteProxy "
            "(expected, as SimpleRemoteProxy is simpler):"
        )
        for method in sorted(missing_in_simple):
            execenv.print(f"  - {method}")

    # Common methods
    common_methods = remote_proxy_methods & simple_proxy_methods
    execenv.print(f"\n✅ Common methods: {len(common_methods)}")

    execenv.print("✨ Client method comparison completed successfully!")


def test_with_real_server() -> None:
    """Run sigima.tests.common.client_unit_test suite with real DataLab server.

    This function runs the comprehensive client unit test suite after launching
    a real DataLab instance in the background, ensuring that the tests pass
    with the real server.
    """
    execenv.print("\n=== Testing with Real DataLab Server ===")

    # Launch DataLab application in the background
    execenv.print("Launching DataLab in background...")
    run_datalab_in_background(wait_until_ready=True)

    # Import and run the comprehensive test from sigima
    execenv.print("Running comprehensive client tests with real server...")
    tester = RemoteClientTester()

    # Initialize connection to real DataLab server using the existing port
    if not tester.init_cdl():
        raise ConnectionRefusedError(
            "Failed to connect to DataLab server. "
            "Make sure DataLab is running and accessible."
        )

    try:
        # Run all tests
        tester.run_comprehensive_test()
        execenv.print("✨ All tests passed with real DataLab server!")
    finally:
        # Clean up
        tester.close_datalab()


if __name__ == "__main__":
    test_compare_server_methods()
    test_compare_client_methods()
    test_with_real_server()
