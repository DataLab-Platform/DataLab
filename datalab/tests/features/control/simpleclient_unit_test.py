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
import warnings

from guidata.env import execenv
from packaging.version import Version
from sigima.client.remote import SimpleRemoteProxy, __required_server_version__
from sigima.client.stub import DataLabStubServer
from sigima.tests.common.client_unit_test import RemoteClientTester

import datalab
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
    run_datalab_in_background()

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
    except Exception as exc:
        execenv.print("❌ Some tests failed with real DataLab server.")
        tester.close_datalab()
        raise exc

    # Clean up
    tester.close_datalab()


def test_version_compatibility() -> None:
    """Test that DataLab version is compatible with Sigima client.

    This test ensures that the current version of DataLab meets the minimum
    requirements of the Sigima client as defined in __required_server_version__.
    """
    execenv.print("\n=== Testing Version Compatibility ===")

    # Get DataLab version
    datalab_version = datalab.__version__
    execenv.print(f"DataLab version: {datalab_version}")
    execenv.print(f"Required version: {__required_server_version__}")

    # Test version comparison using Version class with edge cases
    execenv.print("\nTesting Version comparison:")
    test_cases = [
        ("1.0.0", "1.0.0", True, "Same version"),
        ("1.0.1", "1.0.0", True, "Newer patch version"),
        ("1.1.0", "1.0.0", True, "Newer minor version"),
        ("2.0.0", "1.0.0", True, "Newer major version"),
        ("0.9.9", "1.0.0", False, "Older version"),
        ("1.0.0a1", "1.0.0", False, "Alpha is pre-release (< 1.0.0)"),
        ("1.0.0b2", "1.0.0", False, "Beta is pre-release (< 1.0.0)"),
        ("1.0.0rc1", "1.0.0", False, "RC is pre-release (< 1.0.0)"),
        ("1.0.0", "1.0.0a1", True, "Release is newer than alpha"),
        ("1.0.1", "1.0.0a1", True, "Newer patch > alpha"),
        ("1.0.0a10", "1.0.0a2", True, "Later alpha version"),
    ]

    for ver1, ver2, expected, description in test_cases:
        result = Version(ver1) >= Version(ver2)
        status = "✓" if result == expected else "✗"
        execenv.print(
            f"  {status} Version('{ver1}') >= Version('{ver2}') = {result} "
            f"[expected: {expected}] - {description}"
        )
        assert result == expected, (
            f"Version comparison failed for {ver1} vs {ver2}: "
            f"expected {expected}, got {result}"
        )

    # Check if current DataLab version is compatible
    execenv.print(f"\nChecking DataLab {datalab_version} compatibility...")
    is_compatible = Version(datalab_version) >= Version(__required_server_version__)

    if is_compatible:
        execenv.print(
            f"✅ DataLab version {datalab_version} is compatible "
            f"(>= {__required_server_version__})"
        )
    else:
        execenv.print(
            f"⚠️  DataLab version {datalab_version} is NOT compatible "
            f"(< {__required_server_version__})"
        )

    # Note: During development (alpha/beta/rc versions), we allow the test to pass
    # even if the version comparison indicates incompatibility. In production,
    # both DataLab and Sigima should be at release versions.
    vdatalab = Version(datalab_version)

    # Allow pre-release versions if they're the same base version
    if vdatalab.is_prerelease and not is_compatible:
        # Check if the base version (without pre-release) would be compatible
        base_ver = f"{vdatalab.major}.{vdatalab.minor}.{vdatalab.micro}"
        if Version(base_ver) >= Version(__required_server_version__):
            execenv.print(
                f"ℹ️  Pre-release version {datalab_version} allowed for testing "
                f"(base version {base_ver} >= {__required_server_version__})"
            )
            is_compatible = True

    # This assertion ensures DataLab's version meets Sigima's requirements
    assert is_compatible, (
        f"DataLab version {datalab_version} is not compatible with "
        f"Sigima client requirements (>= {__required_server_version__}). "
        f"Please upgrade DataLab or downgrade the required version in Sigima."
    )

    # Test that warning is issued when connecting to incompatible version
    execenv.print("\nTesting version compatibility warning with stub server...")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a stub server that reports an old version
        stub = DataLabStubServer(port=0, verbose=False)
        # Temporarily patch get_version to return an old version
        original_get_version = stub.get_version

        def mock_old_version():
            return "0.9.0"  # Old version that should trigger warning

        stub.get_version = mock_old_version

        try:
            port = stub.start()

            # Try to connect - should trigger warning
            proxy = SimpleRemoteProxy(autoconnect=False)
            proxy.connect(port=str(port))

            # Check that a warning was issued
            version_warnings = [
                warning
                for warning in w
                if "may not be fully compatible" in str(warning.message)
            ]

            if version_warnings:
                execenv.print(
                    f"✅ Warning correctly issued for incompatible version: "
                    f"{version_warnings[0].message}"
                )
            else:
                execenv.print("⚠️  No warning issued for incompatible version")
                # Print all warnings for debugging
                if w:
                    execenv.print(f"   All warnings captured ({len(w)}):")
                    for warning in w:
                        execenv.print(f"     - {warning.message}")

            assert len(version_warnings) > 0, (
                "Expected warning for incompatible version but none was issued"
            )

        finally:
            # Restore original method and stop server
            stub.get_version = original_get_version
            stub.stop()

    execenv.print("✨ Version compatibility test completed successfully!")


if __name__ == "__main__":
    test_compare_server_methods()
    test_compare_client_methods()
    test_version_compatibility()
    test_with_real_server()
