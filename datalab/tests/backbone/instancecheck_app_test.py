# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Integration test: DataLab concurrent instance detection
========================================================

Verifies that the DataLab application properly detects concurrent instances
using the lock file mechanism.

Scenario: DataLab instance A is already open (lock file with live PID).
User launches DataLab instance B.  B detects A and warns the user.
If the user refuses, B closes.  If the user accepts, B continues.
"""

# guitest: skip

# pylint: disable=redefined-outer-name

from __future__ import annotations

import os
import subprocess
import sys
import time
from unittest import mock

import pytest

from datalab.utils import instancecheck
from datalab.utils.instancecheck import (
    LOCK_FILENAME,
    create_lock_file,
    is_another_instance_running,
)


@pytest.fixture()
def running_datalab(tmp_path):
    """Simulate a running DataLab instance via a subprocess holding a lock.

    The subprocess writes its PID into a lock file and stays alive for the
    duration of the test, mimicking a DataLab instance that is open.
    """
    lock_path = str(tmp_path / LOCK_FILENAME)

    script = tmp_path / "_fake_running_datalab.py"
    script.write_text(
        "import os, sys, time\n"
        f"with open({lock_path!r}, 'w') as f:\n"
        "    f.write(str(os.getpid()))\n"
        "print(os.getpid(), flush=True)\n"
        "time.sleep(60)\n"  # Stay alive — simulates DataLab being open
    )

    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for child to write its PID (readline blocks until newline)
    child_pid = int(proc.stdout.readline().decode().strip())
    time.sleep(0.3)  # Ensure lock file is flushed

    registry = instancecheck.ApplicationInstanceRegistry(app_name="TestDataLab")
    with (
        mock.patch.object(instancecheck, "DEFAULT_REGISTRY", registry),
        mock.patch.object(registry, "_get_lock_path", return_value=lock_path),
    ):
        yield lock_path, child_pid, proc

    proc.terminate()
    proc.wait(timeout=5)


class TestDataLabConcurrentInstances:
    """Integration: DataLab detects when another instance is already open."""

    def test_running_instance_detection_and_force_flow(self, running_datalab):
        """A running instance is detected, preserved, then extended with force."""
        lock_path, child_pid, _proc = running_datalab

        assert is_another_instance_running() == child_pid

        with open(lock_path, encoding="utf-8") as f:
            assert int(f.read().strip()) == child_pid

        create_lock_file(force=True)

        pids = instancecheck._read_lock_pids(lock_path)
        assert os.getpid() in pids
        assert child_pid in pids

        assert is_another_instance_running() == child_pid

    def test_stale_lock_from_crashed_instance(self, tmp_path):
        """DataLab A crashed → stale lock is cleaned on B's startup."""
        lock_path = str(tmp_path / LOCK_FILENAME)

        # Simulate DataLab A that crashed (exits immediately)
        script = tmp_path / "_crashed_datalab.py"
        script.write_text(
            "import os, sys\n"
            f"with open({lock_path!r}, 'w') as f:\n"
            "    f.write(str(os.getpid()))\n"
            "sys.stdout.write(str(os.getpid()))\n"
            "sys.stdout.flush()\n"
        )
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dead_pid = int(result.stdout.strip())
        assert os.path.exists(lock_path)

        registry = instancecheck.ApplicationInstanceRegistry(app_name="TestDataLab")
        with (
            mock.patch.object(instancecheck, "DEFAULT_REGISTRY", registry),
            mock.patch.object(registry, "_get_lock_path", return_value=lock_path),
        ):
            # DataLab B starts up and detects the stale lock
            detected = is_another_instance_running()
            assert detected is None, (
                f"Stale lock (PID {dead_pid}) should be cleaned, got {detected}"
            )

        # Lock file was removed automatically
        assert not os.path.exists(lock_path)
