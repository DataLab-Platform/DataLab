# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Log viewer test
"""

from codraft.app import run
from codraft.tests import logview_error
from codraft.utils.tests import exec_script

SHOW = True  # Show test in GUI-based test launcher


if __name__ == "__main__":
    exec_script(logview_error.__file__)
    run()
