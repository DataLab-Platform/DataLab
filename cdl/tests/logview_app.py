# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Log viewer test
"""

from cdl.app import run
from cdl.tests import logview_error
from cdl.utils.tests import exec_script

SHOW = True  # Show test in GUI-based test launcher


if __name__ == "__main__":
    exec_script(logview_error.__file__)
    run()
