# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...

import pytest

from sigima_.param import ArithmeticParam
from sigima_.tests.env import execenv


@pytest.mark.gui
def test_arithmetic_param_interactive():
    """Arithmetic parameters interactive test."""
    # pylint: disable=import-outside-toplevel
    from guidata.qthelpers import qt_app_context

    with qt_app_context():
        param = ArithmeticParam()
        if param.edit():
            execenv.print(param)


if __name__ == "__main__":
    test_arithmetic_param_interactive()
