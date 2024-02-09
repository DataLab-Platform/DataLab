.. _tutorial_custom_func:

:octicon:`book;1em;sd-text-info` Prototyping a custom processing pipeline
=========================================================================

This example shows how to prototype a custom image processing pipeline using DataLab:

-   Define a custom processing function
-   Create a macro-command to apply the function to an image
-   Use the same code from an external IDE (e.g. Spyder) or a Jupyter notebook
-   Create a plugin to integrate the function in the DataLab GUI

Define a custom processing function
-----------------------------------

For illustrating the extensibility of DataLab, we will use a simple image processing
function that is not available in the standard DataLab distribution, and that
represents a typical use case for prototyping a custom processing pipeline.

The function that we will work on is a denoising filter that combines the ideas of
averaging and edge detection. This filter will average the pixel values in the
neighborhood, but with a twist: it will give less weight to pixels that are
significantly different from the central pixel, assuming they might be part of an
edge or noise.

Here is the code of the ``weighted_average_denoise`` function::

    def weighted_average_denoise(data: np.ndarray) -> np.ndarray:
        """Apply a custom denoising filter to an image.

        This filter averages the pixels in a 5x5 neighborhood, but gives less weight
        to pixels that significantly differ from the central pixel.
        """

        def filter_func(values: np.ndarray) -> float:
            """Filter function"""
            central_pixel = values[len(values) // 2]
            differences = np.abs(values - central_pixel)
            weights = np.exp(-differences / np.mean(differences))
            return np.average(values, weights=weights)

        return spi.generic_filter(data, filter_func, size=5)

For testing our processing function, we will use a generated image from a DataLab
plugin example (`plugins/examples/cdl_example_imageproc.py`). Before starting,
make sure that the plugin is installed in DataLab (see the first steps of the
tutorial :ref:`tutorial_blobs`).

.. figure:: ../../images/tutorials/custom_func/01.png

    To begin, we reorganize the window layout of DataLab to have the "Image Panel" on
    the left and the "Macro Panel" on the right.

.. figure:: ../../images/tutorials/custom_func/02.png

    We generate a new image using the
    "Plugins > Extract blobs (example) > Generate test image" menu.

.. figure:: ../../images/tutorials/custom_func/03.png

    We select a limited size for the image (e.g. 512x512 pixels) because our algorithm
    is quite slow, and click on "OK".

.. figure:: ../../images/tutorials/custom_func/04.png

    We can now see the generated image in the "Image Panel".

Create a macro-command
----------------------

Let's get back to our custom function. We can create a new macro-command that will
apply the function to the current image. To do so, we open the "Macro Panel" and
click on the "New macro" |libre-gui-add| button.

.. |libre-gui-add| image:: ../../../cdl/data/icons/libre-gui-add.svg
    :width: 24px
    :height: 24px

DataLab creates a new macro-command which is not empty: it contains a sample code
that shows how to create a new image and add it to the "Image Panel". We can remove
this code and replace it with our own code::

    # Import the necessary modules
    import numpy as np
    import scipy.ndimage as spi
    from cdl.proxy import RemoteProxy

    # Define our custom processing function
    def weighted_average_denoise(values: np.ndarray) -> float:
        """Apply a custom denoising filter to an image.

        This filter averages the pixels in a 5x5 neighborhood, but gives less weight
        to pixels that significantly differ from the central pixel.
        """
        central_pixel = values[len(values) // 2]
        differences = np.abs(values - central_pixel)
        weights = np.exp(-differences / np.mean(differences))
        return np.average(values, weights=weights)

    # Initialize the proxy to DataLab
    proxy = RemoteProxy()

    # Switch to the "Image Panel" and get the current image
    proxy.set_current_panel("image")
    image = proxy.get_object()
    if image is None:
        # We raise an explicit error if there is no image to process
        raise RuntimeError("No image to process!")

    # Get a copy of the image data, and apply the function to it
    data = np.array(image.data, copy=True)
    data = spi.generic_filter(data, weighted_average_denoise, size=5)

    # Add new image to the panel
    proxy.add_image("My custom filtered data", data)

In DataLab, macro-commands are simply Python scripts:

-   Macros are part of DataLab's **workspace**, which means that they are saved
    and restored when exporting and importing to/from an HDF5 file.

-   Macros are executed in a separate process, so we need to import the necessary
    modules and initialize the proxy to DataLab. The proxy is a special object that
    allows to communicate with DataLab.

-   As a consequence, **when defining a plugin or when controlling DataLab from an
    external IDE, we can use exactly the same code as in the macro-command**. This
    is a very important point, because it means that we can prototype our processing
    pipeline in DataLab, and then use the same code in a plugin or in an external IDE
    to develop it further.

.. note::

    The macro-command is executed in DataLab's Python environment, so we can use
    the modules that are available in DataLab. However, we can also use our own
    modules, as long as they are installed in DataLab's Python environment or in
    a Python distribution that is compatible with DataLab's Python environment.

    If your custom modules are not installed in DataLab's Python environment, and
    if they are compatible with DataLab's Python version, you can prepend the
    ``sys.path`` with the path to the Python distribution that contains your
    modules::

        import sys
        sys.path.insert(0, "/path/to/my/python/distribution")

    This will allow you to import your modules in the macro-command and mix them
    with the modules that are available in DataLab.

    .. warning::

        If you use this method, make sure that your modules are compatible with
        DataLab's Python version. Otherwise, you will get errors when importing
        them.

Now, let's execute the macro-command by clicking on the "Run macro"
|libre-camera-flash-on| button:

-   The macro-command is executed in a separate process, so we can continue to
    work in DataLab while the macro-command is running. And, if the macro-command
    takes too long to execute, we can stop it by clicking on the "Stop macro"
    |libre-camera-flash-off| button.

-   During the execution of the macro-command, we can see the progress in the
    "Macro Panel" window: the process standard output is displayed in the
    "Console" below the macro editor. We can see the following messages:

    - ``---[...]---[# ==> Running 'Untitled 01' macro...]``: the macro-command starts
    - ``Connecting to DataLab XML-RPC server...OK [...]``: the proxy is connected to DataLab
    - ``---[...]---[# <== 'Untitled 01' macro has finished]``: the macro-command ends

.. |libre-camera-flash-on| image:: ../../../cdl/data/icons/libre-camera-flash-on.svg
    :width: 24px
    :height: 24px

.. |libre-camera-flash-off| image:: ../../../cdl/data/icons/libre-camera-flash-off.svg
    :width: 24px
    :height: 24px

.. figure:: ../../images/tutorials/custom_func/05.png

    When the macro-command has finished, we can see the new image in the "Image Panel".
    Our filter has been applied to the image, and we can see that the noise has been
    reduced.

Prototyping with an external IDE
--------------------------------

Now that we have a working prototype of our processing pipeline, we can use the same
code in an external IDE to develop it further.

For example, we can use the Spyder IDE to debug our code. To do so, we need to
install Spyder but not necessarily in DataLab's Python environment (in the case
of the stand-alone version of DataLab, it wouldn't be possible anyway).

The only requirement is to install a DataLab client in Spyder's Python environment:

-   If you use the stand-alone version of DataLab or if you want or need to keep
    DataLab and Spyder in separate Python environments, you can install the
    `DataLab Simple Client <https://pypi.org/project/cdlclient/>`_ (``cdl-client``)
    using the ``pip`` package manager::

        pip install cdl-client

    Or you may also install the `DataLab Python package <https://pypi.org/project/cdl/>`_
    (``cdl``) which includes the client (but also other modules, so we don't recommend
    this method if you don't need all DataLab's features in this Python environment)::

        pip install cdl

-   If you use the DataLab Python package, you may run Spyder in the same Python
    environment as DataLab, so you don't need to install the client: it is already
    available in the main DataLab package (the ``cdl`` package).

Once the client is installed, we can start Spyder and create a new Python script:

.. literalinclude:: custom_func.py
    :language: python
    :linenos:

.. figure:: ../../images/tutorials/custom_func/06.png

    We go back to DataLab and select the first image in the "Image Panel".

.. figure:: ../../images/tutorials/custom_func/07.png

    Then, we execute the script in Spyder, step-by-step (using the defined cells), and
    we can see the result in DataLab.

.. figure:: ../../images/tutorials/custom_func/08.png

    We can see in DataLab that a new image has been added to the "Image Panel". This
    image is the result of the execution of the script in Spyder. Here we have used
    the script without any modification, but we could have modified it to test new
    ideas, and then use the modified script in DataLab.

Prototyping with a Jupyter notebook
-----------------------------------

We can also use a Jupyter notebook to prototype our processing pipeline. To do so,
we need to install Jupyter but not necessarily in DataLab's Python environment
(in the case of the stand-alone version of DataLab, it wouldn't be possible anyway).

The only requirement is to install a DataLab client in Jupyter's Python environment
(see the previous section for more details: that is exactly the same procedure as
for Spyder or any other IDE like Visual Studio Code, for example).

.. figure:: ../../images/tutorials/custom_func/nb.png

    Once the client is installed, we can start Jupyter and create a new notebook.

.. figure:: ../../images/tutorials/custom_func/06.png

    We go back to DataLab and select the first image in the "Image Panel".

.. figure:: ../../images/tutorials/custom_func/09.png

    Then, we execute the notebook in Jupyter, step-by-step (using the defined cells),
    and we can see the result in DataLab. Once again, we can see in DataLab that a new
    image has been added to the "Image Panel". This image is the result of the
    execution of the notebook in Jupyter. As for the script in Spyder, we could have
    modified the notebook to test new ideas, and then use the modified notebook in
    DataLab.

Creating a plugin
-----------------

Now that we have a working prototype of our processing pipeline, we can create a
plugin to integrate it in DataLab's GUI. To do so, we need to create a new Python
module that will contain the plugin code. We can use the same code as in the
macro-command, but we need to make some changes.

.. seealso::

    The plugin system is described in the :ref:`about_plugins` section.

Apart from integrating the feature to DataLab's GUI which is more convenient for
the user, the advantage of creating a plugin is that we can take benefit of the
DataLab infrastructure, if we encapsulate our processing function in a certain
way (see below):

-   Our function will be executed in a separate process, so we can interrupt it if it
    takes too long to execute.

-   Warnings and errors will be handled by DataLab, so we don't need to handle them
    ourselves.

The most significant change is that we need to define a function that will be
operating on DataLab's native image objects (:class:`cdl.obj.ImageObj`), instead of
operating on NumPy arrays. So we need to find a way to call our custom function
``weighted_average_denoise`` with a :class:`cdl.obj.ImageObj` as input and output.
To avoid writing a lot of boilerplate code, we can use the function wrapper provided
by DataLab: :class:`cdl.core.computation.image.Wrap11Func`.

Besides we need to define a class that describes our plugin, which must inherit
from :class:`cdl.plugins.PluginBase` and name the Python script that contains the
plugin code with a name that starts with ``cdl_`` (e.g. ``cdl_custom_func.py``), so
that DataLab can discover it at startup.

Moreover, inside the plugin code, we want to add an entry in the "Plugins" menu, so
that the user can access our plugin from the GUI.

Here is the plugin code:

.. literalinclude:: ../../../plugins/examples/cdl_custom_func.py
    :language: python
    :linenos:

To test it, we have to add the plugin script to one of the plugin directories that
are discovered by DataLab at startup (see the :ref:`about_plugins` section for more
details, or the :ref:`tutorial_blobs` for an example).

.. figure:: ../../images/tutorials/custom_func/10.png

    We restart DataLab and we can see that the plugin has been loaded.

.. figure:: ../../images/tutorials/custom_func/11.png

    We generate again our test image using (see the first steps of the tutorial),
    and we process it using the plugin: "Plugins > My custom filters > Weighted average denoise".
