# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdl/LICENSE for details)

"""
Multiprocessing unit test
-------------------------

Just test if it's possible to run the multiprocessing test from another module. This
may be an issue with the Pool object being global.
"""

from cdl.tests.multiprocess1_unit import test

SHOW = True  # Show test in GUI-based test launcher


if __name__ == "__main__":
    test()
