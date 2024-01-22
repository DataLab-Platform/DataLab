DataLab
=======

DataLab is an **open-source platform for scientific and technical data processing
and visualization** with unique features designed to meet industrial requirements,
extendable with :bdg-ref-success-line:`Plugins <about_plugins>` and working
seamlessly with :ref:`your IDE <use_cases>` or :ref:`Jupyter notebooks <use_cases>`, ...
(see our :bdg-ref-success-line:`Tutorials <tutorials>` for illustrated examples).

DataLab leverages the richness of the scientific Python ecosystem (NumPy, SciPy,
scikit-image, OpenCV, `PlotPyStack`_,...) and Qt graphical user interfaces.

.. figure:: images/DataLab-Overview.png
    :class: dark-light

    Signal and image visualization in DataLab


.. only:: html and not latex

    .. grid:: 2 2 4 4
        :gutter: 1 2 3 4

        .. grid-item-card:: :octicon:`rocket;1em;sd-text-info`  Getting started
            :link: intro/index
            :link-type: doc

            Installation, use cases, key strengths, ...

        .. grid-item-card:: :octicon:`tools;1em;sd-text-info`  Features
            :link: features/index
            :link-type: doc

            In-depth description of DataLab features

        .. grid-item-card:: :octicon:`book;1em;sd-text-info`  API
            :link: api/index
            :link-type: doc

            Reference documentation of DataLab API

        .. grid-item-card:: :octicon:`gear;1em;sd-text-info`  Development
            :link: dev/index
            :link-type: doc

            Roadmap, contributing, ...

With its user-friendly experience and versatile :ref:`usage_modes`, DataLab enables
efficient development of your data processing and visualization applications while
benefiting from an industrial-grade technological platform.

.. figure:: _static/plotpy-stack-powered.png
    :align: center
    :width: 300 px

    DataLab is powered by `PlotPyStack <https://github.com/PlotPyStack>`_,
    the scientific Python-Qt visualization and graphical user interface stack.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro/index
   features/index
   api/index
   dev/index

.. note:: DataLab was created by `Codra`_/`Pierre Raybaut`_ in 2023. It is
          developed and maintained by DataLab open-source project team with
          the support of `Codra`_.

.. _PlotPyStack: https://github.com/PlotPyStack
.. _Codra: https://codra.net/
.. _Pierre Raybaut: https://github.com/PierreRaybaut/
