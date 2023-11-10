# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause
# (see cdlapp/LICENSE for details)

"""
Process isolation unit test
---------------------------

Just test if it's possible to run the process isolation test from another module. This
may be an issue with the Pool object being global.
"""

# guitest: show

from cdlapp.tests.backbone.procisolation1_unit import test

if __name__ == "__main__":
    test()
