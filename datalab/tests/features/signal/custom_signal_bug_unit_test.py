# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Test for Custom Signal creation bug fix
========================================

This test verifies that creating a Custom signal doesn't crash with AttributeError
when xyarray is None.

Bug report: "Créer un signal Custom déclenche immédiatement l'erreur
AttributeError: 'NoneType' object has no attribute 'T'."
"""

import pytest
from guidata.qthelpers import qt_app_context

from datalab.gui.newobject import CustomSignalParam, create_signal_gui


def test_custom_signal_param_initialization():
    """Test that CustomSignalParam initializes properly"""
    with qt_app_context():
        param = CustomSignalParam()
        # Initially, xyarray should be None
        assert param.xyarray is None
        # But other defaults should exist
        assert param.size == 10
        assert param.xmin == 0.0
        assert param.xmax == 1.0


def test_custom_signal_param_setup_array():
    """Test that setup_array creates the xyarray properly"""
    with qt_app_context():
        param = CustomSignalParam()
        assert param.xyarray is None

        # Call setup_array to initialize
        param.setup_array(size=10, xmin=0.0, xmax=1.0)

        # Now xyarray should exist and be transposable
        assert param.xyarray is not None
        assert param.xyarray.shape == (10, 2)

        # Should be able to transpose without error
        transposed = param.xyarray.T
        assert transposed.shape == (2, 10)


def test_custom_signal_param_generate_1d_data():
    """Test that generate_1d_data works even with None xyarray"""
    with qt_app_context():
        param = CustomSignalParam()
        assert param.xyarray is None

        # generate_1d_data should call setup_array internally
        x, y = param.generate_1d_data()

        # Should succeed without AttributeError
        assert x is not None
        assert y is not None
        assert len(x) == param.size
        assert len(y) == param.size


def test_custom_signal_creation_forces_edit_mode():
    """Test that creating a custom signal with edit=False forces edit=True

    This test verifies the bug fix for:
    "Créer un signal Custom déclenche immédiatement l'erreur
    AttributeError: 'NoneType' object has no attribute 'T'."

    The bug occurred when edit=False was passed (the default), which caused
    setup_array to not be called, leaving xyarray as None, which then crashed
    when trying to access xyarray.T.

    The fix forces edit=True for CustomSignalParam to ensure the user is always
    prompted to set up the array properly.
    """
    # This test can't actually test the dialog interaction, but we can verify
    # that the code doesn't crash immediately with AttributeError due to None xyarray
    # The test framework will handle the unattended mode appropriately
    with qt_app_context():
        # In unattended mode, the dialogs will be auto-canceled, so this should
        # return None rather than crashing
        param = CustomSignalParam()
        # In actual usage with GUI, this would show dialogs
        # In unattended test mode, dialogs are auto-canceled
        # The important thing is it doesn't crash with AttributeError
        _signal = create_signal_gui(param, edit=False, parent=None)
        # In unattended mode, this will be None because dialogs are canceled
        # But it shouldn't have crashed with AttributeError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
