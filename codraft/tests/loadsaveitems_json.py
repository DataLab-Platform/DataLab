# -*- coding: utf-8 -*-
#
# Licensed under the terms of the CECILL License
# (see codraft/__init__.py for details)

"""
Unit test for plot items <--> JSON serialization/deserialization

How to save/restore items to/from a JSON string?

    # Plot items --> JSON:
    writer = JSONWriter(None)
    save_items(writer, items)
    text = writer.get_json()

    # JSON --> Plot items:
    items = load_items(JSONReader(text))

"""

# TODO: [P3] Open-source contribution: Refactor this module in guiqwt/tests
# (the "run" method override has to be removed at this occasion)

import os
import os.path as osp

from guiqwt.tests.loadsaveitems_pickle import IOTest

from codraft.utils.jsonio import JSONReader, JSONWriter
from codraft.utils.qthelpers import exec_dialog, qt_app_context

SHOW = True  # Show test in GUI-based test launcher


class JSONTest(IOTest):
    """Class for JSON I/O testing"""

    FNAME = osp.join(osp.dirname(__file__), "loadsavecanvas.json")

    def run(self):
        """Run test"""
        #  Overrides IOTest method to add "unattended mode" support (see `exec_dialog`)
        self.create_dialog()
        self.add_items()
        exec_dialog(self.dlg)
        print("Saving items...", end=" ")
        self.save_items()
        print("OK")

    def restore_items(self):
        """Restore plot items"""
        # jsontext = open(JSONTest.FNAME).read()
        self.plot.deserialize(JSONReader(self.FNAME))

    def save_items(self):
        """Save plot items"""
        writer = JSONWriter(self.FNAME)
        self.plot.serialize(writer)
        # print(len(writer.get_json()))
        writer.save()


def remove_test_file():
    """Remove JSON test file"""
    if osp.isfile(test.FNAME):
        os.remove(test.FNAME)


if __name__ == "__main__":
    with qt_app_context():
        test = JSONTest()
        remove_test_file()
        test.run()  # Build items, save items on close
        test.run()  # Restore items
        remove_test_file()
