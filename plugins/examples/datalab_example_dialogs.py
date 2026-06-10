# -*- coding: utf-8 -*-

"""
Dialog methods plugin example
==============================

This example demonstrates how to use dialog methods in a DataLab plugin.

It shows:
- show_info(): Display information message
- show_warning(): Display warning message
- show_error(): Display error message
- ask_yesno(): Ask yes/no question
- edit_new_signal_parameters(): Edit signal parameters
- edit_new_image_parameters(): Edit image parameters

Usage
-----

1. Copy this file to your DataLab plugins directory
2. Restart DataLab or use "Plugins > Reload plugins"
3. Test each dialog type from the plugin menu

Available Dialog Methods
------------------------

**Message Dialogs:**

- ``self.show_info(message, title=None)``: Information (blue icon)
- ``self.show_warning(message, title=None)``: Warning (yellow icon)
- ``self.show_error(message, title=None)``: Error (red icon)

**Question Dialogs:**

- ``self.ask_yesno(question, title=None, cancelable=False)``:
  Returns ``True`` (Yes), ``False`` (No), or ``None`` (Cancel if cancelable)

**Parameter Dialogs:**

- ``self.edit_new_signal_parameters(title=None, size=None, stype=None)``:
  Returns ``NewSignalParam`` object or ``None`` if canceled

- ``self.edit_new_image_parameters(title=None, shape=None, dtype=None,
  hide_dtype=False, hide_type=False)``:
  Returns ``NewImageParam`` object or ``None`` if canceled

**Remote Proxy Access:**

Use ``self.proxy`` to control DataLab programmatically:

- ``self.proxy.add_object(obj)``: Add signal/image to current panel
- ``self.proxy.get_object()``: Get current selected object
- ``self.proxy.calc(operation, param)``: Run processing operation
- See DataLab remote control API documentation for full list

Example Workflow
----------------

1. Show dialog to get user input (parameters, confirmation, etc.)
2. Create or process data based on user input
3. Add results to DataLab using ``self.proxy.add_object()``
4. Show confirmation message with ``self.show_info()``

See Also
--------

- DataLab plugin documentation: https://datalab-platform.com/en/features/advanced/plugins.html
- DataLab remote control API: https://datalab-platform.com/en/features/general/remote.html
- ``datalab_example_empty.py``: Basic plugin structure
- ``datalab_example_imageproc.py``: Processing operations example
"""

import numpy as np
import sigima.objects

import datalab.plugins


class DialogMethodsExample(datalab.plugins.PluginBase):
    """DataLab Dialog Methods Example Plugin"""

    PLUGIN_INFO = datalab.plugins.PluginInfo(
        name="Dialog Methods (example)",
        version="1.0.0",
        description="Example plugin demonstrating dialog methods",
    )

    def demo_show_info(self) -> None:
        """Demonstrate show_info()"""
        self.show_info("This is an information message.")

    def demo_show_warning(self) -> None:
        """Demonstrate show_warning()"""
        self.show_warning("This is a warning message!")

    def demo_show_error(self) -> None:
        """Demonstrate show_error()"""
        self.show_error("This is an error message!")

    def demo_ask_yesno(self) -> None:
        """Demonstrate ask_yesno()"""
        result = self.ask_yesno(
            "Do you want to proceed?", title="Confirmation", cancelable=True
        )
        if result is True:
            self.show_info("You clicked Yes")
        elif result is False:
            self.show_info("You clicked No")
        else:  # None means Cancel
            self.show_info("You clicked Cancel")

    def demo_create_signal_with_dialog(self) -> None:
        """Demonstrate edit_new_signal_parameters()

        This shows how to:
        1. Display parameter dialog
        2. Handle user cancellation (returns None)
        3. Create object based on user input
        4. Add object to DataLab
        """
        # Ask user for signal parameters
        newparam = self.edit_new_signal_parameters(title="My New Signal", size=1000)

        if newparam is not None:
            # Create signal if user didn't cancel
            x = np.linspace(0, 10, newparam.size)
            y = np.sin(x)
            obj = sigima.objects.create_signal(newparam.title, x, y)
            self.proxy.add_object(obj)
            self.show_info(f"Created signal: {newparam.title}")

    def demo_create_image_with_dialog(self) -> None:
        """Demonstrate edit_new_image_parameters()

        Parameter dialog options:
        - title: Default title for new image
        - shape: Default (width, height) tuple
        - dtype: Default NumPy dtype
        - hide_dtype: Hide dtype selection
        - hide_type: Hide image type selection
        """
        # Ask user for image parameters
        newparam = self.edit_new_image_parameters(
            title="My New Image", shape=(512, 512), hide_type=True
        )

        if newparam is not None:
            # Create image if user didn't cancel
            data = np.random.random((newparam.height, newparam.width))
            obj = sigima.objects.create_image(newparam.title, data)
            self.proxy.add_object(obj)
            self.show_info(f"Created image: {newparam.title}")

    def create_actions(self) -> None:
        """Create actions demonstrating dialog methods"""
        # Signal Panel actions
        sah = self.signalpanel.acthandler
        with sah.new_menu(self.PLUGIN_INFO.name):
            with sah.new_menu("Message Dialogs"):
                sah.new_action(
                    "Show Info",
                    triggered=self.demo_show_info,
                    select_condition="always",
                )
                sah.new_action(
                    "Show Warning",
                    triggered=self.demo_show_warning,
                    select_condition="always",
                )
                sah.new_action(
                    "Show Error",
                    triggered=self.demo_show_error,
                    select_condition="always",
                )

            with sah.new_menu("Question Dialogs"):
                sah.new_action(
                    "Ask Yes/No",
                    triggered=self.demo_ask_yesno,
                    select_condition="always",
                )

            with sah.new_menu("Parameter Dialogs"):
                sah.new_action(
                    "Create Signal with Dialog",
                    triggered=self.demo_create_signal_with_dialog,
                    select_condition="always",
                )

        # Image Panel actions
        iah = self.imagepanel.acthandler
        with iah.new_menu(self.PLUGIN_INFO.name):
            with iah.new_menu("Message Dialogs"):
                iah.new_action(
                    "Show Info",
                    triggered=self.demo_show_info,
                    select_condition="always",
                )
                iah.new_action(
                    "Show Warning",
                    triggered=self.demo_show_warning,
                    select_condition="always",
                )
                iah.new_action(
                    "Show Error",
                    triggered=self.demo_show_error,
                    select_condition="always",
                )

            with iah.new_menu("Parameter Dialogs"):
                iah.new_action(
                    "Create Image with Dialog",
                    triggered=self.demo_create_image_with_dialog,
                    select_condition="always",
                )
