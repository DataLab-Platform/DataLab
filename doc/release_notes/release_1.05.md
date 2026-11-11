# Version 1.5 #

## DataLab Version 1.5.0 ##

### Requirements ###

* Runtime requirements are unchanged since version 1.4.1.

### ✨ New Features ###

**Showcase mode:**

* The new **Showcase mode** plays live demos of DataLab features, without any user interaction, for exhibitions, conference booths and demonstrations. Each demo drives the real user interface: captions, in the language of the user interface, explain each step while the mouse pointer moves, menus open and the relevant parts of the window are highlighted. The showcase mode clears the workspace before each demo (see [Showcase mode](../features/advanced/showcase.rst)).
* **? > Showcase mode...** opens a dialog to choose the demos to play, whether to play them in a loop, the pace, full screen display and the screen showing DataLab; these choices are remembered.
* Four demos are built in: full traceability (history, provenance and cascade recompute), reproducibility with provenance capsules (RO-Crate), advanced and interactive image analysis, and automation with macros and remote control.
* Each installed application plugin also takes part in the showcase, with a demo generated from its examples: they are tried on the methods they are designed for, with their results. An application without such an example opens its first tool instead, e.g. an instrument with its live view. Each of these demos ends with an invitation to discover more applications in the plugin catalog.
* Command-line options start a booth computer directly in unattended showcase mode, e.g. `DataLab.exe --showcase --showcase-loop --showcase-fullscreen`: `--showcase [IDS]` plays all the demos or the given ones (e.g. `d1,d3`), and `--showcase-loop`, `--showcase-speed`, `--showcase-screen` and `--showcase-fullscreen` set the loop, the pace, the screen and the full screen display. With the Python package, the new `datalab-showcase` command plays all the demos in a loop, in full screen.
* On a single-screen booth, each loop may end with one pass of the DataLab Web showcase, in a browser window (Google Chrome, Microsoft Edge or Chromium): check **Include DataLab Web** in the dialog, or use the `--showcase-web [URL]` option.
* The operator keeps control with hotkeys: **F9** pauses or resumes the showcase, and **Shift+F12** stops it after the current demo, giving DataLab back for a live demonstration. A failing demo does not stop the showcase, which may run unattended for hours.
