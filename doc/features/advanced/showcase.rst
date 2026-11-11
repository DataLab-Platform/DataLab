.. _ref-to-showcase:

Showcase mode
=============

.. meta::
    :description: DataLab showcase mode: live and unattended demos for exhibitions and booths
    :keywords: DataLab, showcase, demo, exhibition, booth, conference, kiosk

The showcase mode plays live demos of DataLab features, without any user
interaction: it is designed for exhibitions, conference booths and kiosks, where
DataLab presents itself on a screen, in a loop.

Each demo drives the real user interface, as a user would do: a title card
introduces the demo, then captions explain each step while the mouse pointer
moves, menus open and the relevant parts of the window are highlighted. The
captions follow the language of the user interface. At the end of the demos, an
end card invites the audience to try DataLab.

.. warning::

    The showcase mode clears the workspace (signals, images, macros and history)
    before each demo.

The following demos are available:

.. list-table::
    :header-rows: 1
    :widths: 15 85

    * - Identifier
      - Demo
    * - ``d1``
      - Full traceability: history, provenance, cascade recompute
    * - ``d2``
      - Reproducibility: provenance capsules (RO-Crate)
    * - ``d3``
      - Advanced, interactive image analysis
    * - ``d4``
      - Automation and openness: macros and remote control

Starting the showcase mode
--------------------------

From the menu
^^^^^^^^^^^^^

Select **? > Showcase mode...**: a dialog lets you choose

- the demos to play,
- whether to play them in a loop,
- the pace (slow, normal or fast),
- whether to show DataLab in full screen,
- whether to include DataLab Web, and its URL (see
  :ref:`ref-to-showcase-web` below),
- the screen showing DataLab, when several screens are available.

These choices are remembered for the next time.

From the command line
^^^^^^^^^^^^^^^^^^^^^

The showcase mode may also be started with command-line options, e.g. to start a
booth computer directly in showcase mode, with the standalone version of DataLab:

.. code-block:: console

    DataLab.exe --showcase --showcase-loop --showcase-fullscreen

With the Python package, the ``datalab-showcase`` command is equivalent to
``datalab --showcase --showcase-loop --showcase-fullscreen``.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Option
      - Description
    * - ``--showcase [IDS]``
      - Play the showcase once the main window is shown: all the demos, or the
        demos whose comma-separated identifiers are given (e.g. ``d1,d3``)
    * - ``--showcase-loop``
      - Play the demos in a loop, until the operator stops the showcase
    * - ``--showcase-speed SPEED``
      - Delay factor: 1 by default, slower above (e.g. 1.5), faster below (e.g.
        0.7)
    * - ``--showcase-screen N``
      - Index of the screen showing DataLab (0 for the first screen)
    * - ``--showcase-fullscreen``
      - Show DataLab in full screen during the showcase
    * - ``--showcase-web [URL]``
      - End each loop with one pass of the DataLab Web showcase, in a browser
        window (see :ref:`ref-to-showcase-web` below): DataLab Web at URL, or
        its public deployment if no URL is given

Operator hotkeys
----------------

While the showcase mode plays, the operator may use the following keys:

- :kbd:`F9`: pause or resume the showcase,
- :kbd:`Shift+F12`, or :kbd:`Esc` twice quickly: stop the showcase after the
  current demo.

Once the showcase is stopped, or once all the demos have been played (without
loop), DataLab is given back for a live demonstration: the window gets its
previous state back, and the dialogs are no longer accepted automatically.

The showcase mode is designed to run unattended for hours: if a demo fails, the
showcase moves on to the next one, and the dialogs left open are accepted after a
while.

.. _ref-to-showcase-web:

DataLab Web on the same booth
-----------------------------

`DataLab Web <https://datalab-platform.com/web/>`_, the version of DataLab running
in a web browser, has its own showcase mode, playing demos of its features.

With two screens, run both showcases side by side: DataLab on one screen (see the
**Screen** option of the dialog, or the ``--showcase-screen`` option), and DataLab
Web on the other screen, in a browser window, as described in the documentation of
DataLab Web.

With a single screen, DataLab may include DataLab Web: each loop then ends with an
interlude, during which DataLab Web takes over in a browser window for one pass of
its own demos, in the language of DataLab. Once the browser window closes, DataLab
shows up again and the showcase goes on. To include DataLab Web:

- check **Include DataLab Web** in the **Showcase mode** dialog, and enter the URL
  of DataLab Web (the public deployment is used if no URL is entered),
- or use the ``--showcase-web [URL]`` command-line option, e.g.:

.. code-block:: console

    DataLab.exe --showcase --showcase-loop --showcase-fullscreen --showcase-web

The DataLab Web interlude needs a Chromium-based web browser: Google Chrome (the
preferred one), Microsoft Edge (always available on Windows) or Chromium. With
Microsoft Edge, DataLab Web may run slower when the *Enhance your security on the
web* setting of Edge applies to it. The browser runs with a profile dedicated to
the showcase, stored in the DataLab configuration folder: the browser window does
not show the bookmarks nor the tabs of the usual browser profile, and the Python
runtime of DataLab Web, downloaded on the first interlude, is kept for the next
ones. The browser window is shown in full screen when DataLab is.

The identifier of the interlude is ``web``: give it among the identifiers of the
demos to play it elsewhere than at the end of the loop (e.g. ``--showcase
d1,web,d2``). To stop the showcase during the interlude, close the browser window
(e.g. with :kbd:`Alt+F4`): DataLab shows up again, then press :kbd:`Shift+F12`.

The DataLab Web interlude lasts 15 minutes at most: the browser window is closed
afterwards. This duration and the DataLab Web demos to play (all of them, by
default) may be changed in the ``[main]`` section of the DataLab configuration file,
with the ``showcase_web_timeout`` option (in minutes) and the ``showcase_web_ids``
option (comma-separated identifiers of the DataLab Web demos).

Demos of the installed applications
-----------------------------------

Besides the demos of DataLab itself, each installed application plugin (see
:doc:`plugins`) takes part in the showcase with a demo presenting the application
in the **Applications** panel. The demos of DataLab and those of the applications
alternate. The identifier of the demo of an application is ``app:`` followed by
the plugin identifier (e.g. ``app:org.example.my-application``).

The demo of an application is built from what the application declares, without
any code specific to the showcase:

- The application is presented with its name and its description.
- Its first two examples designed for a method are tried with their **Try with
  this example** command: the example is opened, the parameters of the method are
  shown, then the method runs. Its main result is shown, with its table of
  results, if any. When the example is designed for several methods, the next one
  may then run on this result (e.g. compute an absorption spectrum, then
  normalize it).
- An application without such an example opens its first tool which needs no
  selection, an instrument first: the instrument window shows its live view,
  acquires a few times (with and without the **New group** option), then is
  closed, which disconnects the instrument.
- The demo ends with an invitation to discover more applications in the plugin
  catalog.

An application without example nor tool is only presented. The options changed
by the demo (method selected in the Applications panel, recent applications of
the welcome page, **New group** option of the instruments) are restored at the
end of the demo.
