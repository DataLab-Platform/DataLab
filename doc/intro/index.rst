Getting started
===============

.. meta::
    :description: Getting started with DataLab, the open platform for signal and image processing
    :keywords: DataLab, signal processing, image processing, open platform, scientific data, industrial data, industrial-grade software

DataLab is an open platform for signal and image processing, designed to be used by
scientists, engineers, and researchers in academia and industry, while offering the
reliability of industrial-grade software. It is a versatile software that can be used
for a wide range of applications, from simple data analysis to complex signal processing
and image analysis tasks.

DataLab integrates seemlessly into your workflow thanks to three main operating modes:

.. list-table::
    :header-rows: 0

    * - |appmode|
      - **Stand-alone application**, with a graphical user interface that allows you to interact with your data and visualize the results of your analysis in real time.

    * - |libmode|
      - **Python library**, allowing you to integrate DataLab functions (or graphical user interfaces) into your own Python scripts and programs or Jupyter notebooks.

    * - |remotemode|
      - **Remotely controlled** from your own software, or from an IDE (e.g., Spyder) or a Jupyter notebook, using the DataLab API.

.. |appmode| image:: ../../resources/DataLab-app.svg
    :width: 64px
    :height: 64px
    :class: dark-light no-scaled-link

.. |libmode| image:: ../../resources/DataLab-lib.svg
    :width: 64px
    :height: 64px
    :class: dark-light no-scaled-link

.. |remotemode| image:: ../../resources/DataLab-remote.svg
    :width: 64px
    :height: 64px
    :class: dark-light no-scaled-link

DataLab leverages the power of Python and its scientific ecosystem, through the use of
the following libraries:

- `NumPy <https://numpy.org/>`_ for numerical computing (arrays, linear algebra, etc.)
- `SciPy <https://www.scipy.org/>`_ for scientific computing (interpolation, special functions, etc.)
- `scikit-image <https://scikit-image.org/>`_ and `OpenCV <https://opencv.org/>`_ for image processing
- `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ for wavelet transform
- `PlotPyStack <https://github.com/PlotPyStack/>`_ for Qt-based interactive data visualization

.. only:: html and not latex

    .. grid:: 2 2 4 4
        :gutter: 1 2 3 4

        .. grid-item-card:: :octicon:`package;1em;sd-text-info`  Installation
            :link: installation
            :link-type: doc

            How to install DataLab on your computer

        .. grid-item-card:: :octicon:`gift;1em;sd-text-info`  Introduction
            :link: introduction
            :link-type: doc

            Use cases and key strengths of DataLab

        .. grid-item-card:: :octicon:`star;1em;sd-text-info`  Key features
            :link: keyfeatures
            :link-type: doc

            Feature matrix of DataLab

        .. grid-item-card:: :octicon:`mortar-board;1em;sd-text-info`  Tutorials
            :link: tutorials/index
            :link-type: doc

            Tutorials to learn how to use DataLab

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   introduction
   keyfeatures
   tutorials/index
