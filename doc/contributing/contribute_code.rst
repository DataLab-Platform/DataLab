.. _contribute_code:

Contribute code
===============

This page explains how you can contribute code to the project.

.. seealso::

    The :ref:`guidelines` page presents the coding guidelines that you
    should follow when contributing to the project.

Fork the project
----------------

The first step is to fork the project on GitHub.
You can do that by visiting `DataLab GitHub project <https://github.com/DataLab-Platform/DataLab/>`_
and clicking on the "Fork" button on the top right corner of the page.

Once you have forked the project, you will have a copy of the project
in your own GitHub account. Then you can clone the project on your computer
and start working on it.

Submit a pull request
---------------------

Once you have made some changes, you can submit a pull request to the
original project. To do that, go to your forked project on GitHub and
click on the "Pull request" button on the top right corner of the page.

Then you will have to fill a form to describe your pull request. Once
you have submitted the pull request, the project maintainers will review
your changes and merge them if they are satisfied.

During the review process, the project maintainers will check that your
code follows the coding guidelines and that it does not break the existing
tests. If your code does not follow the coding guidelines, you will have
to fix it before your pull request can be merged.

Maintainer tasks
----------------

A few maintenance tasks are intentionally **not** run by the release CI and must be performed locally on a Windows workstation by a project maintainer. They produce committed assets that the CI then consumes as-is.

Refreshing documentation screenshots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The PNG files under ``doc/images/`` are versioned. The screenshot generator renders actual Qt windows, dialogs and menus offscreen: it does not capture the desktop, drive the mouse or require Computer Use. Run it from a source checkout with the development environment configured in ``.env``, including Sigima's test data, Qt and OpenCV. The reference captures are generated on Windows with the Segoe UI font.

The screenshot pipeline produces **per-language** files (``foo.fr.png`` / ``foo.en.png``) thanks to Sphinx's ``figure_language_filename`` configuration. RST sources reference the unsuffixed name (``foo.png``); each language build automatically picks the matching suffix when available and falls back to the unsuffixed file otherwise.

Re-run this whenever a change affects the rendered UI (menus, dialogs, panels, themes):

.. code-block:: bat

    scripts\update_screenshots.bat

or via the **🖼️ Refresh doc screenshots** VS Code task. Each language runs in its own process with a disposable home directory and in-memory default settings: personal preferences and installed plugins are not changed. The theme, scale, window sizes and tutorial data are fixed. Native window decorations are intentionally absent; popup menus are rendered by Qt and composited over their parent window.

To update only the interface and a tutorial, or render previews outside the source tree:

.. code-block:: bat

    scripts\update_screenshots.bat --group interface spectrum
    scripts\update_screenshots.bat --group blobs --language en --output-dir build\screenshot-preview

Available groups are ``interface`` (startup, settings and History), ``gallery`` (the existing demonstration scenes), ``fabry-perot``, ``spectrum`` and ``blobs``. All groups and both languages are selected by default. Tutorial figures remain shared English PNGs under ``doc/images/tutorials/``; interface figures are localized under ``doc/images/shots/``. The blob tutorial executes the example plugin with seeded random data; its plugin-directory figure uses illustrative paths rather than personal or temporary directories.

The scenarios are in ``datalab/tests/scenarios/tutorial_screenshots.py``. They check that annotations are stored separately from metadata, peak detection creates XY markers rather than another signal, and the blob detector's color filter is disabled. Generation stops on errors; the focused regression tests also check PNG output and protection of an existing configuration.

Inspect the resulting figures and build the English and French documentation before committing the updated PNGs. Once a ``foo.fr.png`` + ``foo.en.png`` pair exists for a given asset, the unsuffixed ``foo.png`` becomes a pure fallback and can be removed in a follow-up cleanup commit.

Regenerating graphical resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The application icon (``resources/DataLab.ico``) and the WiX installer bitmaps (``wix/dialog.bmp``, ``wix/banner.bmp``) are generated from SVG sources. Re-run only when the SVG sources change:

.. code-block:: bat

    scripts\build_resources.bat

or via the **🎨 Generate resources** VS Code task.

The images displayed by the application (splash screen, logos, plot watermark) are generated in ``datalab/data/logo/`` together with their ``@2x`` high-resolution variants, used on high-DPI displays: to regenerate only these images, run ``scripts\build_resources.bat logo``.
