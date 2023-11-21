# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Log viewer test
"""

# guitest: show

from cdl.app import run
from cdl.tests.features.utilities import logview_error
from cdl.utils.tests import exec_script


def test_logviewer_app():
    """Test log viewer"""
    exec_script(logview_error.__file__)
    run()


if __name__ == "__main__":
    test_logviewer_app()
