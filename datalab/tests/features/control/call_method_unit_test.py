# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for call_method generic proxy feature
"""

# guitest: show

from __future__ import annotations

import xmlrpc.client

import numpy as np
from sigima.tests.data import get_test_image, get_test_signal

from datalab.tests import datalab_in_background_context


def test_call_method() -> None:
    """Test call_method generic proxy feature"""
    with datalab_in_background_context() as proxy:
        # Test 1: Add some test data
        signal = get_test_signal("paracetamol.txt")
        proxy.add_object(signal)
        image = get_test_image("flower.npy")
        proxy.add_object(image)

        # Test 2: Call remove_object without panel parameter (auto-detection)
        # Should find method on current panel (signal panel)
        proxy.set_current_panel("signal")
        titles_before = proxy.get_object_titles("signal")
        assert len(titles_before) == 1, "Should have one signal"

        # Remove the signal object using call_method (no panel parameter)
        # This should auto-detect and use the current panel
        proxy.call_method("remove_object", force=True)

        titles_after = proxy.get_object_titles("signal")
        assert len(titles_after) == 0, "Signal should be removed"

        # Test 3: Call method on specific panel
        proxy.set_current_panel("image")
        titles_before = proxy.get_object_titles("image")
        assert len(titles_before) == 1, "Should have one image"

        # Remove the image object using call_method with explicit panel parameter
        # This tests the panel parameter is correctly passed through XML-RPC
        proxy.call_method("remove_object", force=True, panel="image")

        titles_after = proxy.get_object_titles("image")
        assert len(titles_after) == 0, "Image should be removed"

        # Test 3b: Verify panel parameter works when calling from different panel
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        proxy.add_signal("Test Signal X", x, y)
        proxy.set_current_panel("signal")
        assert len(proxy.get_object_titles("signal")) == 1
        # Call remove_object on signal panel while being on signal panel
        proxy.call_method("remove_object", force=True, panel="signal")
        assert len(proxy.get_object_titles("signal")) == 0

        # Test 4: Test method resolution order (main window -> current panel)
        # First verify we can call a main window method
        current_panel = proxy.call_method("get_current_panel")
        assert current_panel in ["signal", "image"], "Should return valid panel name"

        # Test 5: Add more test data and test other methods
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        proxy.add_signal("Test Signal 1", x, y)
        proxy.add_signal("Test Signal 2", x, y * 2)

        # Test calling a method with positional and keyword arguments
        proxy.set_current_panel("signal")
        proxy.select_objects([1, 2])

        # Test delete_metadata through call_method
        proxy.call_method("delete_metadata", refresh_plot=False, keep_roi=True)

        # Test 5: Error handling - try to call a private method
        # Note: XML-RPC converts exceptions to xmlrpc.client.Fault
        try:
            proxy.call_method("__init__")
            assert False, "Should not allow calling private methods"
        except xmlrpc.client.Fault as exc:
            assert "private method" in exc.faultString.lower()

        # Test 6: Error handling - try to call non-existent method
        try:
            proxy.call_method("this_method_does_not_exist")
            assert False, "Should raise AttributeError for non-existent method"
        except xmlrpc.client.Fault as exc:
            assert "does not exist" in exc.faultString.lower()

        # Test 7: Call main window method (not panel method)
        panel_name = proxy.call_method("get_current_panel")
        assert panel_name == "signal", "Should return current panel name"

        print("✅ All call_method tests passed!")


if __name__ == "__main__":
    test_call_method()
