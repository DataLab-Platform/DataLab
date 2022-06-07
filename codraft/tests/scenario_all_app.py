# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Unit test scenario: all features

Maximizing test coverage.
"""

# pylint: disable=duplicate-code

from codraft.tests import codraft_app_context
from codraft.tests.scenario_ima_app import test_image_features
from codraft.tests.scenario_sig_app import test_signal_features


def test():
    """Run all unit test scenarios"""
    with codraft_app_context() as win:
        test_signal_features(win)
        test_image_features(win)


if __name__ == "__main__":
    test()
