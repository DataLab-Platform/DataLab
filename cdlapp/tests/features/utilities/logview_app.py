# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Log viewer test
"""

# guitest: show

from cdlapp.app import run
from cdlapp.tests.features.utilities import logview_error
from cdlapp.utils.tests import exec_script

if __name__ == "__main__":
    exec_script(logview_error.__file__)
    run()
