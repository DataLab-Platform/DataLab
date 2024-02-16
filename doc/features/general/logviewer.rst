.. _ref-to-logviewer:

Log viewer
==========

.. meta::
  :description: DataLab's log viewer
  :keywords: DataLab, log, viewer, crash, bug, report

Despite countless efforts (unit testing, test coverage, ...),
DataLab might crash or behave unexpectedly.

For those situations, DataLab provides two logs (located in your home directory):
  - "Traceback log", for Python exceptions
  - "Faulthandler log", for system failures (e.g. Qt-related crash)

.. figure:: /images/shots/logviewer.png

    DataLab log viewer (see "?" menu)

If DataLab crashed or if any Python exception is raised during
its execution, those log files will be updated accordingly.
DataLab will even notify that new informations are available in
log files at next startup. This is an invitation to submit a bug report.

Reporting unexpected behavior or any other bug on `GitHub Issues`_
will be greatly appreciated, especially if those log file contents
are attached to the report (as information on your installation
configuration, see :ref:`ref-to-instviewer`).

.. _GitHub Issues: https://github.com/DataLab-Platform/DataLab/issues/new/choose
