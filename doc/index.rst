CobraDataLab user guide
=======================

CobraDataLab is a **generic signal and image processing software**.
It is based on Python scientific libraries (such as NumPy,
SciPy or scikit-image) and Qt graphical user interfaces
(thanks to `guidata`_ and `guiqwt`_ libraries).

.. figure:: images/panorama.png

    Signal and image visualization in CobraDataLab

CobraDataLab features are available not only using the
**stand-alone application**
(easily installed thanks to the Windows installer or the Python package)
but also by **embedding it into your own application**
(see the "embedded tests" for detailed examples of how to do so).

External resources:
    .. list-table::
        :widths: 20, 80

        * - `Home`_
          - CobraDataLab home page
        * - `PyPI`_
          - Python Package Index
        * - `GitHub`_
          - Bug reports and feature requests

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   overview
   roadmap
   features/general/index
   features/signal/index
   features/image/index

Copyrights and licensing
------------------------

- Copyright © 2022 `CEA`_ - `Codra`_, Pierre Raybaut
- Licensed under the terms of the `BSD / CeCILL-B License`_

.. _guidata: https://pypi.python.org/pypi/guidata
.. _guiqwt: https://pypi.python.org/pypi/guiqwt
.. _PyPI: https://pypi.python.org/pypi/cdl
.. _Home: https://codra-ingenierie-informatique.github.io/CobraDataLab/
.. _GitHub: https://github.com/CODRA-Ingenierie-Informatique/CobraDataLab
.. _CEA: http://www.cea.fr
.. _Codra: https://codra.net/
.. _BSD / CeCILL-B License: https://github.com/CODRA-Ingenierie-Informatique/CobraDataLab/blob/master/LICENSE
