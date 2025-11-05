.. _tutorial_extensibility:

:octicon:`video;1em;sd-text-info` Add your own features
=======================================================

.. meta::
    :description: Tutorial video on how to add your own features to DataLab, the open-source platform for scientific data analysis
    :keywords: DataLab, scientific data analysis, Python, macro-commands, remote control, plugins

.. only:: html and not latex

    .. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/s_z9SUegYJQ"
        frameborder="0" allowfullscreen></iframe>

.. only:: latex and not html

    .. image:: https://img.youtube.com/vi/s_z9SUegYJQ/0.jpg
        :target: https://www.youtube.com/watch?v=s_z9SUegYJQ

.. warning::

    This video shows an older version of DataLab. Some features may have changed
    since then. Please refer to the documentation for the most up-to-date
    information.

In this tutorial, we will show how to add your own features to DataLab using three
different approaches:

1. Macro-commands, using the integrated macro manager
2. Remote control of DataLab from an external IDE (e.g. Spyder) or a Jupyter notebook
3. Plugins

The first common point between these three approaches is that they all rely on DataLab's
high-level API, which allow to interact with almost every aspect of the software.
This API is here accessed using Python scripts, but it may also be accessed using
any other language when using the remote control approach (because it relies on
a standard communication protocol, XML-RPC).

The second common point is that they all use Python code, and compatible proxy objects,
so that the same code can be at least partially reused in the three approaches.
