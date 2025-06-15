# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Arithmetic parameters unit test.
"""

# pylint: disable=invalid-name  # Allows short reference names like x, y, ...
# guitest: show

from guidata.qthelpers import qt_app_context

from cdl.env import execenv
from sigima_.param import ArithmeticParam


def test_arithmetic_param_interactive():
    """Arithmetic parameters interactive test."""
    with qt_app_context():
        param = ArithmeticParam()
        if param.edit():
            execenv.print(param)


if __name__ == "__main__":
    test_arithmetic_param_interactive()
