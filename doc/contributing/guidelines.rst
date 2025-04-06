.. _guidelines:

Coding guidelines
=================

Generic coding guidelines
-------------------------

We follow the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ coding style.

In particular, we are especially strict about the following guidelines:

- Limit all lines to a maximum of 79 characters.
- Respect the naming conventions (classes, functions, variables, etc.).
- Use specific exceptions instead of the generic :class:`Exception`.

To enforce these guidelines, the following tools are mandatory:

- `ruff <https://pypi.org/project/ruff/>`_ for code formatting and static code analysis.
- `pylint <https://pypi.org/project/pylint/>`_ for static code analysis.

ruff
^^^^

If you are using `Visual Studio Code <https://code.visualstudio.com/>`_,
the project settings will automatically format your code with `ruff` on save
(you may also run the task "Run Ruff" to run `ruff` on the project).

To run `ruff`, run the following command::

    ruff check

pylint
^^^^^^

To run `pylint`, run the following command::

    pylint datalab

If you are using `Visual Studio Code <https://code.visualstudio.com/>`_
on Windows, you may run the task "Run Pylint" to run `pylint` on the project.

.. note::

    A `pylint` rating greater than 9/10 is required to merge a pull request.

Specific coding guidelines
--------------------------

In addition to the generic coding guidelines, we have the following specific
guidelines:

- Write docstrings for all classes, methods and functions. The docstrings
  should follow the `Google style <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_.

- Add typing annotations for all functions and methods. The annotations should
  use the future syntax (``from __future__ import annotations``)

- Try to keep the code as simple as possible. If you have to write a complex
  piece of code, try to split it into several functions or classes.

- Add as many comments as possible. The code should be self-explanatory, but
  it is always useful to add some comments to explain the general idea of the
  code, or to explain some tricky parts.

- Do not use ``from module import *`` statements, even in the ``__init__``
  module of a package.

- Avoid using mixins (multiple inheritance) when possible. It is often
  possible to use composition instead of inheritance.

- Avoid using ``__getattr__`` and ``__setattr__`` methods. They are often used
  to implement lazy initialization, but this can be done in a more explicit
  way.
