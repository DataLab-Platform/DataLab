# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Log viewer test
"""

# guitest: show

from cdl.app import run
from cdl.tests.features.utilities import logview_error
from sigima_.tests.helpers import exec_script


def test_logviewer_app():
    """Test log viewer"""
    exec_script(logview_error.__file__)
    run()


if __name__ == "__main__":
    test_logviewer_app()
