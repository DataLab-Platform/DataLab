# Copyright (c) DataLab Platform Developers, BSD 3-Clause license, see LICENSE file.

"""
Worker and WorkerStateMachine Unit Tests
========================================

This module contains comprehensive unit tests for the Worker class and
WorkerStateMachine class in DataLab's processor module.

Tests cover:
1. State machine functionality independently from Worker class
2. Worker class nominal operations
3. Exception handling in computations
4. Process pool restart mechanism for long-running/cancelled computations
"""

# pylint: disable=protected-access

import time
from unittest.mock import Mock, patch

import pytest
from guidata.qthelpers import qt_app_context
from qtpy import QtWidgets as QW

import datalab.gui.processor.base as base_module
from datalab.gui.processor.base import Worker, WorkerState, WorkerStateMachine
from datalab.gui.processor.catcher import CompOut


class TestWorkerStateMachine:
    """Test suite for WorkerStateMachine class - independent from Worker class."""

    def test_initial_state(self):
        """Test that state machine starts in IDLE state."""
        sm = WorkerStateMachine()
        assert sm.current_state == WorkerState.IDLE

    def test_valid_transitions(self):
        """Test all valid state transitions."""
        sm = WorkerStateMachine()

        # IDLE -> STARTING
        sm.transition_to(WorkerState.STARTING)
        assert sm.current_state == WorkerState.STARTING

        # STARTING -> RUNNING
        sm.transition_to(WorkerState.RUNNING)
        assert sm.current_state == WorkerState.RUNNING

        # RUNNING -> FINISHED
        sm.transition_to(WorkerState.FINISHED)
        assert sm.current_state == WorkerState.FINISHED

        # FINISHED -> IDLE (complete cycle)
        sm.transition_to(WorkerState.IDLE)
        assert sm.current_state == WorkerState.IDLE

    def test_same_state_transition_allowed(self):
        """Test that transitioning to the same state is a no-op."""
        sm = WorkerStateMachine()

        # Should not raise exception
        sm.transition_to(WorkerState.IDLE)
        assert sm.current_state == WorkerState.IDLE

        # Same for other states
        sm.transition_to(WorkerState.STARTING)
        sm.transition_to(WorkerState.STARTING)
        assert sm.current_state == WorkerState.STARTING

    def test_invalid_transitions(self):
        """Test that invalid transitions raise ValueError."""
        sm = WorkerStateMachine()

        # IDLE can only go to STARTING
        with pytest.raises(
            ValueError,
            match="Invalid transition from WorkerState.IDLE to WorkerState.RUNNING",
        ):
            sm.transition_to(WorkerState.RUNNING)

        with pytest.raises(
            ValueError,
            match="Invalid transition from WorkerState.IDLE to WorkerState.FINISHED",
        ):
            sm.transition_to(WorkerState.FINISHED)

        # STARTING can only go to RUNNING
        sm.transition_to(WorkerState.STARTING)
        with pytest.raises(
            ValueError,
            match="Invalid transition from WorkerState.STARTING to WorkerState.IDLE",
        ):
            sm.transition_to(WorkerState.IDLE)

        with pytest.raises(ValueError, match="Invalid transition.*STARTING.*FINISHED"):
            sm.transition_to(WorkerState.FINISHED)

    def test_reset_to_idle_from_any_state(self):
        """Test that reset_to_idle works from any state."""
        sm = WorkerStateMachine()

        # Test from each state
        states_to_test = [
            WorkerState.IDLE,
            WorkerState.STARTING,
            WorkerState.RUNNING,
            WorkerState.FINISHED,
        ]

        for state in states_to_test:
            # Force state (bypassing transition validation for test)
            sm._current_state = state
            sm.reset_to_idle()
            assert sm.current_state == WorkerState.IDLE


# Module-level functions for pickle compatibility in multiprocessing
def dummy_successful_computation(x: int, y: int) -> int:
    """Simple computation that returns sum of two numbers."""
    return x + y


def dummy_failing_computation() -> None:
    """Computation that raises an exception."""
    raise ValueError("Test exception from computation")


def dummy_infinite_computation() -> None:
    """Computation that runs forever (until process is killed)."""
    while True:
        time.sleep(0.1)  # Simulate work


class TestWorker:
    """Test suite for Worker class functionality."""

    def setup_method(self):
        """Setup for each test method - ensure clean pool state."""
        # Clean up any existing pool
        Worker.terminate_pool(wait=False)
        Worker.create_pool()

    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up pool to avoid interference between tests
        Worker.terminate_pool(wait=False)

    def test_worker_initial_state(self):
        """Test worker starts in correct initial state."""
        worker = Worker()
        assert worker.state_machine.current_state == WorkerState.IDLE
        assert worker.asyncresult is None

    def test_worker_pool_management(self):
        """Test pool creation and termination."""
        # Test static methods work
        Worker.terminate_pool()
        Worker.create_pool()

        # Verify pool exists by checking global POOL variable
        assert base_module.POOL is not None

        # Test termination
        Worker.terminate_pool()
        # Note: POOL global variable is set to None in terminate_pool

    def test_nominal_computation_case(self):
        """Test Case 2a: Nominal case - start computation, wait, get result."""
        with qt_app_context():
            worker = Worker()

            # Start computation
            worker.run(dummy_successful_computation, (10, 5))
            assert worker.state_machine.current_state == WorkerState.RUNNING

            # Wait for completion
            while not worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.001)  # Small sleep to avoid busy waiting

            # Should be finished
            assert worker.state_machine.current_state == WorkerState.FINISHED
            assert worker.has_result_available()

            # Get result
            result = worker.get_result()
            assert isinstance(result, CompOut)
            assert result.result == 15  # 10 + 5
            assert result.error_msg is None
            assert worker.state_machine.current_state == WorkerState.IDLE

    def test_computation_exception_case(self):
        """Test Case 2b: Computation raises an exception."""
        # Note: In unattended test mode, exceptions are re-raised by wng_err_func
        # This test validates that the worker can handle exceptions that bubble up
        with qt_app_context():
            worker = Worker()

            # Start computation that will fail
            worker.run(dummy_failing_computation, ())
            assert worker.state_machine.current_state == WorkerState.RUNNING

            # In unattended mode, the exception will be raised when getting result
            # Wait for computation to "complete" (which means ready to raise exception)
            while not worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.001)

            # Should be marked as finished (even though it will raise on get_result)
            assert worker.state_machine.current_state == WorkerState.FINISHED
            assert worker.has_result_available()

            # In unattended mode, get_result will raise the original exception
            with pytest.raises(ValueError, match="Test exception from computation"):
                worker.get_result()

            # After exception, should still be back to IDLE (cleanup happened)
            assert worker.state_machine.current_state == WorkerState.IDLE

    def test_computation_cancellation_and_pool_restart(self):
        """Test Case 2c: Long computation with cancellation and pool restart."""
        with qt_app_context():
            worker = Worker()

            # Start infinite computation
            worker.run(dummy_infinite_computation, ())
            assert worker.state_machine.current_state == WorkerState.RUNNING

            # Let it run briefly to ensure it's actually started
            start_time = time.time()
            while time.time() - start_time < 0.1:  # 100ms
                QW.QApplication.processEvents()
                time.sleep(0.001)
                # Verify it's still running (not finished)
                if worker.is_computation_finished():
                    pytest.fail("Infinite computation finished unexpectedly")

            # Simulate user cancellation (like progress dialog cancel)
            worker.restart()
            assert worker.state_machine.current_state == WorkerState.IDLE
            assert worker.asyncresult is None

            # Verify pool was restarted by running a normal computation
            worker.run(dummy_successful_computation, (20, 22))
            assert worker.state_machine.current_state == WorkerState.RUNNING

            # Wait for this computation to complete
            while not worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.001)

            # Should complete successfully, proving pool restart worked
            assert worker.state_machine.current_state == WorkerState.FINISHED
            result = worker.get_result()
            assert result.result == 42  # 20 + 22
            assert result.error_msg is None

    def test_worker_restart_from_various_states(self):
        """Test worker.restart() behavior from different states."""
        with qt_app_context():
            worker = Worker()

            # Test restart from IDLE (should be no-op)
            assert worker.state_machine.current_state == WorkerState.IDLE
            worker.restart()
            assert worker.state_machine.current_state == WorkerState.IDLE

            # Test restart from STARTING state
            with patch.object(worker.state_machine, "transition_to"):
                # Force STARTING state
                worker.state_machine._current_state = WorkerState.STARTING
                worker.asyncresult = Mock()  # Simulate asyncresult exists

                worker.restart()

                assert worker.asyncresult is None
                assert worker.state_machine.current_state == WorkerState.IDLE

            # Test restart from FINISHED state
            worker.state_machine._current_state = WorkerState.FINISHED
            worker.asyncresult = Mock()

            worker.restart()
            assert worker.asyncresult is None
            assert worker.state_machine.current_state == WorkerState.IDLE

    def test_worker_close(self):
        """Test worker.close() method."""
        worker = Worker()

        # Test close with no active computation
        worker.close()  # Should not raise

        # Test close with active computation (simulate)
        worker.asyncresult = Mock()
        worker.close()  # Should not raise

    def test_error_conditions(self):
        """Test various error conditions."""
        worker = Worker()

        # Test starting computation when not in IDLE state
        worker.state_machine._current_state = WorkerState.RUNNING
        with pytest.raises(
            ValueError, match="Cannot start computation from WorkerState.RUNNING state"
        ):
            worker.run(dummy_successful_computation, (1, 2))

        # Reset to proper state
        worker.state_machine.reset_to_idle()

        # Test getting result when not in FINISHED state
        with pytest.raises(
            ValueError, match="Cannot get result from WorkerState.IDLE state"
        ):
            worker.get_result()

        # Test has_result_available in various states
        assert not worker.has_result_available()  # IDLE

        worker.state_machine._current_state = WorkerState.RUNNING
        assert not worker.has_result_available()  # RUNNING

        worker.state_machine._current_state = WorkerState.FINISHED
        assert worker.has_result_available()  # FINISHED

    def test_restart_pool_method_integration(self):
        """Test that restart_pool is properly integrated with restart method."""
        with qt_app_context():
            worker = Worker()

            # Start an infinite computation
            worker.run(dummy_infinite_computation, ())

            # Let it run briefly
            start_time = time.time()
            while time.time() - start_time < 0.05:  # 50ms
                QW.QApplication.processEvents()
                time.sleep(0.001)

            # Force RUNNING state and call restart (which should use restart_pool)
            assert worker.state_machine.current_state == WorkerState.RUNNING
            worker.restart()

            # Should be back to IDLE
            assert worker.state_machine.current_state == WorkerState.IDLE

            # Pool should be functional - test with a simple computation
            worker.run(dummy_successful_computation, (100, 200))
            while not worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.001)

            result = worker.get_result()
            assert result.result == 300


# Module-level function for pickle compatibility
def simple_multiplication(a: int, b: int) -> int:
    """Simple computation that multiplies two numbers."""
    return a * b


def test_worker_and_state_machine_integration():
    """Integration test ensuring Worker and WorkerStateMachine work together."""
    with qt_app_context():
        # Ensure clean pool
        Worker.terminate_pool(wait=False)
        Worker.create_pool()

        try:
            worker = Worker()

            # Test complete workflow
            assert worker.state_machine.current_state == WorkerState.IDLE

            # Start computation
            worker.run(simple_multiplication, (6, 7))
            assert worker.state_machine.current_state == WorkerState.RUNNING

            # Wait and check states
            while not worker.is_computation_finished():
                QW.QApplication.processEvents()
                time.sleep(0.001)

            assert worker.state_machine.current_state == WorkerState.FINISHED

            # Get result and verify state transition
            result = worker.get_result()
            assert result.result == 42  # 6 * 7
            assert worker.state_machine.current_state == WorkerState.IDLE

        finally:
            # Cleanup
            Worker.terminate_pool(wait=False)


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__, "-v"])
