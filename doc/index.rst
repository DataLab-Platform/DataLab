DataLab User Guide
==================

DataLab is a **generic signal and image processing software**
with unique features designed to meet industrial requirements
(see :ref:`key_strengths`: Extensibility, Interoperability, ...).
It is based on Python scientific libraries (such as NumPy,
SciPy or scikit-image) and Qt graphical user interfaces (thanks to
the powerful `PlotPyStack`_ - mostly the `guidata`_ and `PlotPy`_ libraries).

With its user-friendly experience and versatile :ref:`usage_modes`,
DataLab enables efficient development of your data processing and
visualization applications while benefiting from an industrial-grade
technological platform.

.. figure:: images/DataLab-Overview.png
    :class: dark-light

    Signal and image visualization in DataLab

DataLab :ref:`main_features` are available not only using the
**stand-alone application**
(easily installed thanks to the Windows installer or the Python package)
but also by **embedding it into your own application**
(see the "embedded tests" for detailed examples of how to do so).

.. figure:: _static/plotpy-stack-powered.png
    :align: center
    :width: 300 px

    DataLab is powered by `PlotPyStack <https://github.com/PlotPyStack>`_,
    the Python-Qt visualization and scientific graphical user interface stack.

External resources:
    .. list-table::
        :widths: 20, 80

        * - `Home`_
          - DataLab home page
        * - `PyPI`_
          - Python Package Index
        * - `GitHub`_
          - Bug reports and feature requests

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro/index
   features/general/index
   features/signal/index
   features/image/index
   dev/index
   changelog

Copyrights and licensing
------------------------

- Copyright © 2023 `Codra`_, Pierre Raybaut
- Licensed under the terms of the `BSD 3-Clause`_

.. _PlotPyStack: https://github.com/PlotPyStack
.. _guidata: https://pypi.python.org/pypi/guidata
.. _PlotPy: https://pypi.python.org/pypi/PlotPy
.. _PyPI: https://pypi.python.org/pypi/CDL
.. _Home: https://codra-ingenierie-informatique.github.io/DataLab/
.. _GitHub: https://github.com/Codra-Ingenierie-Informatique/DataLab
.. _Codra: https://codra.net/
.. _BSD 3-Clause: https://github.com/Codra-Ingenierie-Informatique/DataLab/blob/master/LICENSE
