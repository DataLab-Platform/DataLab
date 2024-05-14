Getting started
===============

.. meta::
    :description: Getting started with DataLab, the open platform for signal and image processing
    :keywords: DataLab, signal processing, image processing, open platform, scientific data, industrial data, industrial-grade software

DataLab is an open platform for signal and image processing. Its functional scope is
intentionally broad. With its many functions, some of them technically advanced, DataLab
enables the processing and visualization of all types of scientific data. As a result,
scientific, industrial, and innovation stakeholders can have access to an easy-to-use
tool that seemlessly integrates into their workflow and offers the reliability of
industrial-grade software.

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
