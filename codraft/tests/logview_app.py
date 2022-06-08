# -*- coding: utf-8 -*-
#
# Licensed under the terms of the BSD 3-Clause or the CeCILL-B License
# (see codraft/__init__.py for details)

"""
Log viewer test
"""

import os.path as osp
import subprocess
import sys

from codraft.app import run
from codraft.utils.env import execenv

SHOW = True  # Show test in GUI-based test launcher


def exec_script(path):
    """Run test script"""
    command = [sys.executable, '"' + path + '"']
    stderr = subprocess.DEVNULL if execenv.unattended else None
    with subprocess.Popen(" ".join(command), shell=True, stderr=stderr) as proc:
        proc.wait()


if __name__ == "__main__":
    exec_script(osp.join(osp.dirname(__file__), "logview_error.py"))
    run()
