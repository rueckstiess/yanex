"""
Tests for experiment public API module - complete conversion to utilities.

This file replaces test_api.py with equivalent functionality using the new test utilities.
All test logic and coverage is preserved while reducing setup duplication.
"""

import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import yanex
from tests.test_utils import TestDataFactory, create_isolated_manager


class TestThreadLocalState:
    """Test thread-local state management - minimal changes needed."""

    def test_no_active_context_standalone_mode(self):
        """Test standalone mode when no active experiment context."""
        # In standalone mode, get_params() should return empty dict
        params = yanex.get_params()
        assert params == {}

        # And is_standalone() should return True
        assert yanex.is_standalone() is True
        assert yanex.has_context() is False

        # get_experiment_dir() should return None in standalone mode
        exp_dir = yanex.get_experiment_dir()
        assert exp_dir is None

    def test_get_current_experiment_id_no_context(self):
        """Test getting experiment ID without context returns None in standalone mode."""
        # In standalone mode, should return None instead of raising exception
        experiment_id = yanex._get_current_experiment_id()
        assert experiment_id is None

    def test_set_and_get_experiment_id(self):
        """Test setting and getting experiment ID."""
        test_id = "test12345"
        yanex._set_current_experiment_id(test_id)

        assert yanex._get_current_experiment_id() == test_id

        # Clean up
        yanex._clear_current_experiment_id()

    def test_clear_experiment_id(self):
        """Test clearing experiment ID."""
        yanex._set_current_experiment_id("test12345")
        yanex._clear_current_experiment_id()

        # After clearing, should return None (standalone mode)
        experiment_id = yanex._get_current_experiment_id()
        assert experiment_id is None

    def test_thread_isolation(self):
        """Test that thread-local state is isolated between threads."""
        results = {}

        def worker(thread_id):
            exp_id = f"exp{thread_id}"
            yanex._set_current_experiment_id(exp_id)
            time.sleep(0.1)  # Allow other threads to run
            results[thread_id] = yanex._get_current_experiment_id()
            yanex._clear_current_experiment_id()

        # Start multiple threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Each thread should have seen its own experiment ID
        assert results[0] == "exp0"
        assert results[1] == "exp1"
        assert results[2] == "exp2"


class TestExperimentAPI:
    """Test experiment API functions - major improvements with utilities."""

    def setup_method(self):
        """Set up test experiment context using utilities."""
        # NEW: Much cleaner setup using utilities
        self.manager = create_isolated_manager()
        self.experiment_id = "api12345"

        # Create experiment directory
        exp_dir = self.manager.storage.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        # NEW: Use factory for standardized metadata
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=self.experiment_id, status="running", name="test-experiment"
        )
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        # NEW: Use factory for config with custom overrides
        config = TestDataFactory.create_experiment_config(
            "ml_training",
            learning_rate=0.01,  # Override default
            epochs=10,
            model={
                "architecture": "resnet",
                "layers": 18,
                "optimizer": {"type": "adam", "lr": 0.001, "weight_decay": 1e-4},
            },
            data={"batch_size": 32, "augmentation": True},
        )
        self.manager.storage.save_config(self.experiment_id, config)

        # Set current experiment
        yanex._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        yanex._clear_current_experiment_id()

    @patch("yanex.api._get_experiment_manager")
    def test_get_params(self, mock_get_manager):
        """Test getting experiment parameters."""
        mock_get_manager.return_value = self.manager

        params = yanex.get_params()
        assert params["learning_rate"] == 0.01
        assert params["epochs"] == 10

    @patch("yanex.api._get_experiment_manager")
    def test_get_param_simple(self, mock_get_manager):
        """Test getting simple (non-nested) parameters."""
        mock_get_manager.return_value = self.manager

        # Get existing parameter
        lr = yanex.get_param("learning_rate")
        assert lr == 0.01

        # Get non-existent parameter with default
        dropout = yanex.get_param("dropout", 0.1)
        assert dropout == 0.1

        # Get non-existent parameter without default
        optimizer = yanex.get_param("nonexistent")
        assert optimizer is None

    @patch("yanex.api._get_experiment_manager")
    def test_get_param_dot_notation(self, mock_get_manager):
        """Test getting nested parameters with dot notation."""
        mock_get_manager.return_value = self.manager

        # Test single-level nesting
        arch = yanex.get_param("model.architecture")
        assert arch == "resnet"

        layers = yanex.get_param("model.layers")
        assert layers == 18

        batch_size = yanex.get_param("data.batch_size")
        assert batch_size == 32

        augmentation = yanex.get_param("data.augmentation")
        assert augmentation is True

        # Test double-level nesting
        opt_type = yanex.get_param("model.optimizer.type")
        assert opt_type == "adam"

        opt_lr = yanex.get_param("model.optimizer.lr")
        assert opt_lr == 0.001

        weight_decay = yanex.get_param("model.optimizer.weight_decay")
        assert weight_decay == 1e-4

    @patch("yanex.api._get_experiment_manager")
    def test_get_param_dot_notation_with_defaults(self, mock_get_manager):
        """Test dot notation with non-existent keys using defaults."""
        mock_get_manager.return_value = self.manager

        # Non-existent nested parameter with default
        momentum = yanex.get_param("model.optimizer.momentum", 0.9)
        assert momentum == 0.9

        # Non-existent top-level with nested key
        scheduler_type = yanex.get_param("scheduler.type", "cosine")
        assert scheduler_type == "cosine"

        # Non-existent middle-level parameter
        hidden_size = yanex.get_param("model.transformer.hidden_size", 512)
        assert hidden_size == 512

        # Completely non-existent nested path
        unknown = yanex.get_param("unknown.nested.path", "default")
        assert unknown == "default"

    @patch("yanex.api._get_experiment_manager")
    def test_get_param_dot_notation_edge_cases(self, mock_get_manager):
        """Test edge cases for dot notation."""
        mock_get_manager.return_value = self.manager

        # Test accessing non-dict as if it were dict
        invalid = yanex.get_param("learning_rate.nested", "default")
        assert invalid == "default"

    @patch("yanex.api._get_experiment_manager")
    def test_get_status(self, mock_get_manager):
        """Test getting experiment status."""
        mock_get_manager.return_value = self.manager

        status = yanex.get_status()
        assert status == "running"

    def test_get_experiment_id(self):
        """Test getting current experiment ID."""
        exp_id = yanex.get_experiment_id()
        assert exp_id == self.experiment_id

    @patch("yanex.api._get_experiment_manager")
    def test_get_experiment_dir(self, mock_get_manager):
        """Test getting current experiment directory."""
        mock_get_manager.return_value = self.manager

        exp_dir = yanex.get_experiment_dir()
        expected_dir = self.manager.storage.experiments_dir / self.experiment_id
        assert exp_dir == expected_dir
        assert exp_dir.exists()
        assert exp_dir.is_dir()

    @patch("yanex.api._get_experiment_manager")
    def test_get_metadata(self, mock_get_manager):
        """Test getting experiment metadata."""
        mock_get_manager.return_value = self.manager

        metadata = yanex.get_metadata()
        assert metadata["id"] == self.experiment_id
        assert metadata["status"] == "running"
        assert metadata["name"] == "test-experiment"

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_basic(self, mock_get_manager):
        """Test basic metrics logging."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95, "loss": 0.05}
        yanex.log_metrics(results)

        # Verify results were saved
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        assert saved_results[0]["accuracy"] == 0.95
        assert saved_results[0]["loss"] == 0.05
        assert "step" in saved_results[0]

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_with_step(self, mock_get_manager):
        """Test metrics logging with specific step."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.90}
        yanex.log_metrics(results, step=5)

        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert saved_results[0]["step"] == 5

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_merge_info(self, mock_get_manager):
        """Test merging metrics with existing step."""
        mock_get_manager.return_value = self.manager

        # Log first result
        yanex.log_metrics({"accuracy": 0.90}, step=1)

        # Log second result with same step (should merge)
        yanex.log_metrics({"loss": 0.05}, step=1)

        # Verify both metrics are present in the same step
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        assert saved_results[0]["accuracy"] == 0.90
        assert saved_results[0]["loss"] == 0.05

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_merge_behavior(self, mock_get_manager):
        """Test comprehensive metrics merging behavior."""
        mock_get_manager.return_value = self.manager

        # Log initial metrics for step 3
        yanex.log_metrics({"foo": 1}, step=3)

        # Log additional metrics for same step (should merge)
        yanex.log_metrics({"bar": 2}, step=3)

        # Log updated value for existing metric (should overwrite)
        yanex.log_metrics({"foo": 10}, step=3)

        # Add one more metric
        yanex.log_metrics({"baz": 3}, step=3)

        # Verify all metrics are present with correct values
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        result = saved_results[0]
        assert result["step"] == 3
        assert result["foo"] == 10  # Updated value
        assert result["bar"] == 2  # Original value preserved
        assert result["baz"] == 3  # New metric added
        assert "timestamp" in result  # Original timestamp preserved
        assert "last_updated" in result  # Last updated timestamp added

    @patch("yanex.api._get_experiment_manager")
    def test_log_results_deprecation_warning(self, mock_get_manager):
        """Test that log_results shows deprecation warning."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95, "loss": 0.05}

        with pytest.warns(DeprecationWarning, match="log_results\\(\\) is deprecated"):
            yanex.log_results(results)

        # Verify results were still saved (through log_metrics)
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        assert saved_results[0]["accuracy"] == 0.95

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_with_string_step_raises_type_error(self, mock_get_manager):
        """Test that log_metrics raises TypeError when step is a string."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95}

        with pytest.raises(TypeError, match="step parameter must be an int or None"):
            yanex.log_metrics(results, step="0")

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_with_float_step_raises_type_error(self, mock_get_manager):
        """Test that log_metrics raises TypeError when step is a float."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95}

        with pytest.raises(TypeError, match="step parameter must be an int or None"):
            yanex.log_metrics(results, step=1.0)

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_with_list_step_raises_type_error(self, mock_get_manager):
        """Test that log_metrics raises TypeError when step is a list."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95}

        with pytest.raises(TypeError, match="step parameter must be an int or None"):
            yanex.log_metrics(results, step=[1])

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_with_dict_step_raises_type_error(self, mock_get_manager):
        """Test that log_metrics raises TypeError when step is a dict."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95}

        with pytest.raises(TypeError, match="step parameter must be an int or None"):
            yanex.log_metrics(results, step={"step": 1})

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_mixed_types_scenario(self, mock_get_manager):
        """Test the bug scenario: string step followed by None step."""
        mock_get_manager.return_value = self.manager

        # This should raise TypeError immediately, preventing the sorting bug
        with pytest.raises(TypeError, match="step parameter must be an int or None"):
            yanex.log_metrics({"loss": 0.5}, step="0")

        # Verify no results were saved (error occurred before storage)
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 0

    @patch("yanex.api._get_experiment_manager")
    def test_log_metrics_valid_types_work(self, mock_get_manager):
        """Test that log_metrics works correctly with valid types."""
        mock_get_manager.return_value = self.manager

        # Valid: explicit int step
        yanex.log_metrics({"loss": 0.5}, step=0)

        # Valid: None step (auto-incremented)
        yanex.log_metrics({"accuracy": 0.9})

        # Valid: another explicit int step
        yanex.log_metrics({"f1_score": 0.85}, step=2)

        # Verify all results were saved correctly
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 3
        assert saved_results[0]["step"] == 0
        assert saved_results[1]["step"] == 1
        assert saved_results[2]["step"] == 2

    @patch("yanex.api._get_experiment_manager")
    def test_log_artifact(self, mock_get_manager):
        """Test artifact logging."""
        mock_get_manager.return_value = self.manager

        # Create a test file in the isolated environment
        test_file = self.manager.storage.experiments_dir / "test_artifact.txt"
        test_file.write_text("test content")

        yanex.log_artifact("test.txt", test_file)

        # Verify artifact was saved
        artifact_path = (
            self.manager.storage.experiments_dir
            / self.experiment_id
            / "artifacts"
            / "test.txt"
        )
        assert artifact_path.exists()
        assert artifact_path.read_text() == "test content"

    @patch("yanex.api._get_experiment_manager")
    def test_log_text(self, mock_get_manager):
        """Test text artifact logging."""
        mock_get_manager.return_value = self.manager

        content = "This is test content"
        yanex.log_text(content, "output.txt")

        # Verify text artifact was saved
        artifact_path = (
            self.manager.storage.experiments_dir
            / self.experiment_id
            / "artifacts"
            / "output.txt"
        )
        assert artifact_path.exists()
        assert artifact_path.read_text() == content

    @patch("yanex.api._get_experiment_manager")
    def test_log_matplotlib_figure(self, mock_get_manager):
        """Test matplotlib figure logging."""
        del mock_get_manager  # Unused parameter
        # Skip this test for now due to complex mocking requirements
        pytest.skip("Complex matplotlib mocking - tested in integration")

    def test_log_matplotlib_figure_no_matplotlib(self):
        """Test matplotlib figure logging without matplotlib installed."""
        # Mock the import inside the function to raise ImportError
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "matplotlib.pyplot":
                raise ImportError("No module named 'matplotlib'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="matplotlib is required"):
                yanex.log_matplotlib_figure(None, "plot.png")


class TestManualExperimentControl:
    """Test manual experiment control functions - improved with utilities."""

    def setup_method(self):
        """Set up test experiment context using utilities."""
        # NEW: Much cleaner setup
        self.manager = create_isolated_manager()
        self.experiment_id = "control123"

        # Create experiment directory
        exp_dir = self.manager.storage.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Use factory for metadata
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=self.experiment_id, status="running"
        )
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        yanex._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        yanex._clear_current_experiment_id()

    @patch("yanex.api._get_experiment_manager")
    def test_completed(self, mock_get_manager):
        """Test manual experiment completion."""
        mock_get_manager.return_value = self.manager

        with pytest.raises(yanex._ExperimentCompletedException):
            yanex.completed()

        # Verify experiment was marked as completed
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

    @patch("yanex.api._get_experiment_manager")
    def test_fail(self, mock_get_manager):
        """Test manual experiment failure."""
        mock_get_manager.return_value = self.manager

        error_msg = "Something went wrong"
        with pytest.raises(yanex._ExperimentFailedException):
            yanex.fail(error_msg)

        # Verify experiment was marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert metadata["error_message"] == error_msg

    @patch("yanex.api._get_experiment_manager")
    def test_cancel(self, mock_get_manager):
        """Test manual experiment cancellation."""
        mock_get_manager.return_value = self.manager

        cancel_msg = "User requested cancellation"
        with pytest.raises(yanex._ExperimentCancelledException):
            yanex.cancel(cancel_msg)

        # Verify experiment was marked as cancelled
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "cancelled"
        assert metadata["cancellation_reason"] == cancel_msg


class TestExperimentContext:
    """Test ExperimentContext context manager - improved with utilities."""

    def setup_method(self):
        """Set up test environment using utilities."""
        # NEW: Cleaner setup with utilities
        self.manager = create_isolated_manager()
        self.experiment_id = "context123"

        # Create experiment directory
        exp_dir = self.manager.storage.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Use factory for 'created' status metadata
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=self.experiment_id, status="created", started_at=None
        )
        self.manager.storage.save_metadata(self.experiment_id, metadata)

    @patch("yanex.api._get_experiment_manager")
    def test_context_manager_normal_exit(self, mock_get_manager):
        """Test normal context manager execution."""
        mock_get_manager.return_value = self.manager

        context = yanex.ExperimentContext(self.experiment_id)

        with context:
            # Verify experiment is running and context is set
            assert yanex._get_current_experiment_id() == self.experiment_id
            status = self.manager.get_experiment_status(self.experiment_id)
            assert status == "running"

        # After exit, should be completed and context cleared
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

        # After context exit, should return to standalone mode
        experiment_id = yanex._get_current_experiment_id()
        assert experiment_id is None

    @patch("yanex.api._get_experiment_manager")
    def test_context_manager_with_exception(self, mock_get_manager):
        """Test context manager with unhandled exception."""
        mock_get_manager.return_value = self.manager

        context = yanex.ExperimentContext(self.experiment_id)

        with pytest.raises(ValueError):
            with context:
                raise ValueError("Test error")

        # Should be marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert "ValueError: Test error" in metadata["error_message"]

    @patch("yanex.api._get_experiment_manager")
    def test_context_manager_keyboard_interrupt(self, mock_get_manager):
        """Test context manager with KeyboardInterrupt."""
        mock_get_manager.return_value = self.manager

        context = yanex.ExperimentContext(self.experiment_id)

        with pytest.raises(KeyboardInterrupt):
            with context:
                raise KeyboardInterrupt()

        # Should be marked as cancelled
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "cancelled"
        assert "Interrupted by user" in metadata["cancellation_reason"]

    @patch("yanex.api._get_experiment_manager")
    def test_context_manager_manual_completion(self, mock_get_manager):
        """Test context manager with manual completion."""
        mock_get_manager.return_value = self.manager

        context = yanex.ExperimentContext(self.experiment_id)

        with context:
            # Manually complete the experiment
            yanex.completed()

        # Should be marked as completed, not completed again
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

    @patch("yanex.api._get_experiment_manager")
    def test_context_manager_manual_failure(self, mock_get_manager):
        """Test context manager with manual failure."""
        mock_get_manager.return_value = self.manager

        context = yanex.ExperimentContext(self.experiment_id)

        with context:
            yanex.fail("Manual failure")

        # Should be marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert metadata["error_message"] == "Manual failure"


class TestExperimentCreation:
    """Test experiment creation functions - improved with utilities."""

    def setup_method(self):
        """Set up test environment using utilities."""
        # NEW: Use isolated manager for clean environment
        self.manager = create_isolated_manager()

    @patch("yanex.api._get_experiment_manager")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_create_experiment(self, mock_capture_env, mock_git_info, mock_get_manager):
        """Test create_experiment function."""
        # Setup mocks using utilities
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        mock_get_manager.return_value = self.manager

        script_path = Path(__file__)
        context = yanex.create_experiment(
            script_path=script_path,
            name="test-experiment",
            config={"learning_rate": 0.01},
            tags=["ml", "test", "unit-tests"],
            description="Test experiment",
        )

        # Should return ExperimentContext
        assert isinstance(context, yanex.ExperimentContext)
        assert len(context.experiment_id) == 8

    @patch("yanex.api._get_experiment_manager")
    def test_create_context_existing_experiment(self, mock_get_manager):
        """Test create_context with existing experiment."""
        mock_get_manager.return_value = self.manager

        # Create experiment using utilities
        experiment_id = "test1234"
        exp_dir = self.manager.storage.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Use factory for metadata
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id, status="created"
        )
        self.manager.storage.save_metadata(experiment_id, metadata)

        # Create context
        context = yanex.create_context(experiment_id)
        assert isinstance(context, yanex.ExperimentContext)
        assert context.experiment_id == experiment_id

    @patch("yanex.api._get_experiment_manager")
    def test_create_context_nonexistent_experiment(self, mock_get_manager):
        """Test create_context with non-existent experiment."""
        mock_get_manager.return_value = self.manager

        from yanex.utils.exceptions import ExperimentNotFoundError

        with pytest.raises(ExperimentNotFoundError):
            yanex.create_context("nonexistent")


class TestAPIParameterizedScenarios:
    """Additional tests using utilities for comprehensive scenarios."""

    @pytest.mark.parametrize(
        "status,expected_fields",
        [
            ("running", ["started_at"]),
            ("completed", ["completed_at", "duration"]),
            ("failed", ["failed_at", "error"]),
        ],
    )
    def test_get_metadata_different_statuses(
        self, isolated_manager, status, expected_fields
    ):
        """Test getting metadata for experiments in different statuses."""
        experiment_id = f"{status[:4].ljust(4, 'x')}test"

        # Create experiment directory
        exp_dir = isolated_manager.storage.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # NEW: Use factory for different status types
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id, status=status
        )
        isolated_manager.storage.save_metadata(experiment_id, metadata)

        yanex._set_current_experiment_id(experiment_id)

        try:
            with patch("yanex.api._get_experiment_manager") as mock_get_manager:
                mock_get_manager.return_value = isolated_manager

                retrieved_metadata = yanex.get_metadata()
                assert retrieved_metadata["status"] == status

                # Verify status-specific fields are present
                for field in expected_fields:
                    assert field in retrieved_metadata

        finally:
            yanex._clear_current_experiment_id()

    @pytest.mark.parametrize(
        "config_type,expected_params",
        [
            ("ml_training", ["learning_rate", "batch_size", "epochs"]),
            ("data_processing", ["n_docs", "chunk_size", "format"]),
            ("simple", ["param1", "param2", "param3"]),
        ],
    )
    def test_get_params_different_config_types(
        self, isolated_manager, config_type, expected_params
    ):
        """Test getting parameters for different config types."""
        experiment_id = f"{config_type[:4].ljust(4, 'x')}cfg1"

        # Create experiment
        exp_dir = isolated_manager.storage.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=experiment_id, status="running"
        )
        isolated_manager.storage.save_metadata(experiment_id, metadata)

        # NEW: Use factory for different config types
        config = TestDataFactory.create_experiment_config(config_type)
        isolated_manager.storage.save_config(experiment_id, config)

        yanex._set_current_experiment_id(experiment_id)

        try:
            with patch("yanex.api._get_experiment_manager") as mock_get_manager:
                mock_get_manager.return_value = isolated_manager

                params = yanex.get_params()

                # Verify expected parameters are present
                for param in expected_params:
                    assert param in params

        finally:
            yanex._clear_current_experiment_id()


class TestExecuteBashScript:
    """Test bash script execution functionality."""

    def setup_method(self):
        """Set up test experiment context."""
        self.manager = create_isolated_manager()
        self.experiment_id = "script01"

        # Create experiment directory
        exp_dir = self.manager.storage.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "artifacts").mkdir(parents=True, exist_ok=True)

        # Create experiment metadata
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=self.experiment_id, status="running", name="test-script"
        )
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        # Create config with some parameters
        config = TestDataFactory.create_experiment_config(
            "simple", param1="value1", param2=42, nested={"key": "value"}
        )
        self.manager.storage.save_config(self.experiment_id, config)

        # Set current experiment
        yanex._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        yanex._clear_current_experiment_id()

    def test_execute_bash_script_standalone_mode_works(self):
        """Test that execute_bash_script works in standalone mode."""
        yanex._clear_current_experiment_id()  # Ensure standalone mode

        # Should work without raising an error
        result = yanex.execute_bash_script("echo 'test'")

        # Verify basic functionality
        assert result["exit_code"] == 0
        assert "test" in result["stdout"]
        assert result["stderr"] == ""
        assert "working_directory" in result

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_success(self, mock_get_manager):
        """Test successful script execution."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script("echo 'Hello World'", stream_output=False)

        # Check return value structure
        assert isinstance(result, dict)
        assert result["exit_code"] == 0
        assert "Hello World" in result["stdout"]
        assert result["stderr"] == ""
        assert result["execution_time"] > 0
        assert result["command"] == "echo 'Hello World'"
        assert "working_directory" in result

        # Check that execution was logged
        logged_executions = self.manager.storage.load_script_runs(self.experiment_id)
        assert len(logged_executions) == 1
        assert logged_executions[0]["command"] == "echo 'Hello World'"
        assert logged_executions[0]["exit_code"] == 0

        # Check that stdout artifact was created
        artifacts_dir = (
            self.manager.storage.get_experiment_directory(self.experiment_id)
            / "artifacts"
        )
        stdout_file = artifacts_dir / "script_stdout.txt"
        assert stdout_file.exists()
        assert "Hello World" in stdout_file.read_text()

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_failure(self, mock_get_manager):
        """Test script execution with non-zero exit code."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script("exit 1", stream_output=False)

        # Should return failure but not raise exception by default
        assert result["exit_code"] == 1
        assert result["stdout"] == ""
        assert result["stderr"] == ""

        # Check that failure was logged
        logged_executions = self.manager.storage.load_script_runs(self.experiment_id)
        assert len(logged_executions) == 1
        assert logged_executions[0]["exit_code"] == 1

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_failure_with_raise_on_error(self, mock_get_manager):
        """Test script execution with raise_on_error=True."""
        mock_get_manager.return_value = self.manager

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            yanex.execute_bash_script(
                "exit 1", raise_on_error=True, stream_output=False
            )

        assert exc_info.value.returncode == 1
        assert exc_info.value.cmd == "exit 1"

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_with_stderr(self, mock_get_manager):
        """Test script execution that produces stderr output."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script(
            "echo 'error message' >&2", stream_output=False
        )

        assert result["exit_code"] == 0
        assert result["stdout"] == ""
        assert "error message" in result["stderr"]

        # Check that stderr artifact was created
        artifacts_dir = (
            self.manager.storage.get_experiment_directory(self.experiment_id)
            / "artifacts"
        )
        stderr_file = artifacts_dir / "script_stderr.txt"
        assert stderr_file.exists()
        assert "error message" in stderr_file.read_text()

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_timeout(self, mock_get_manager):
        """Test script execution with timeout."""
        mock_get_manager.return_value = self.manager

        with pytest.raises(subprocess.TimeoutExpired):
            yanex.execute_bash_script("sleep 10", timeout=0.1, stream_output=False)

        # Check that timeout was logged
        logged_executions = self.manager.storage.load_script_runs(self.experiment_id)
        assert len(logged_executions) == 1
        assert logged_executions[0]["exit_code"] == -1
        assert "timed out" in logged_executions[0]["error"]

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_working_directory(self, mock_get_manager):
        """Test script execution with custom working directory."""
        mock_get_manager.return_value = self.manager

        # Create a temporary directory for testing
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test.txt"
            test_file.write_text("test content")

            result = yanex.execute_bash_script(
                "ls test.txt", working_dir=temp_path, stream_output=False
            )

            assert result["exit_code"] == 0
            assert "test.txt" in result["stdout"]
            assert str(temp_path) == result["working_directory"]

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_environment_variables(self, mock_get_manager):
        """Test that experiment parameters are passed as environment variables."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script(
            "echo $YANEX_EXPERIMENT_ID $YANEX_PARAM_param1 $YANEX_PARAM_param2",
            stream_output=False,
        )

        assert result["exit_code"] == 0
        output = result["stdout"]
        assert self.experiment_id in output
        assert "value1" in output
        assert "42" in output

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_complex_parameters(self, mock_get_manager):
        """Test that complex parameters are JSON-encoded in environment."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script(
            "echo $YANEX_PARAM_nested", stream_output=False
        )

        assert result["exit_code"] == 0
        # Should contain JSON representation of nested parameter
        assert (
            '{"key": "value"}' in result["stdout"]
            or '{"key":"value"}' in result["stdout"]
        )

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_multiple_calls(self, mock_get_manager):
        """Test multiple script executions in same experiment."""
        mock_get_manager.return_value = self.manager

        # Execute first script
        result1 = yanex.execute_bash_script("echo 'first'", stream_output=False)
        assert result1["exit_code"] == 0

        # Execute second script
        result2 = yanex.execute_bash_script("echo 'second'", stream_output=False)
        assert result2["exit_code"] == 0

        # Check that both executions were logged
        logged_executions = self.manager.storage.load_script_runs(self.experiment_id)
        assert len(logged_executions) == 2

        commands = [r["command"] for r in logged_executions]
        assert "echo 'first'" in commands
        assert "echo 'second'" in commands

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_default_artifact_prefix(self, mock_get_manager):
        """Test that default artifact_prefix creates files with 'script' prefix."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script(
            "echo 'Hello World'; echo 'Error message' >&2", stream_output=False
        )

        assert result["exit_code"] == 0
        assert "Hello World" in result["stdout"]
        assert "Error message" in result["stderr"]

        # Check that artifacts were created with default 'script' prefix
        artifacts_dir = (
            self.manager.storage.get_experiment_directory(self.experiment_id)
            / "artifacts"
        )

        stdout_file = artifacts_dir / "script_stdout.txt"
        stderr_file = artifacts_dir / "script_stderr.txt"

        assert stdout_file.exists()
        assert stderr_file.exists()
        assert "Hello World" in stdout_file.read_text()
        assert "Error message" in stderr_file.read_text()

    @patch("yanex.api._get_experiment_manager")
    def test_execute_bash_script_custom_artifact_prefix(self, mock_get_manager):
        """Test that custom artifact_prefix creates files with specified prefix."""
        mock_get_manager.return_value = self.manager

        result = yanex.execute_bash_script(
            "echo 'Custom output'; echo 'Custom error' >&2",
            stream_output=False,
            artifact_prefix="custom_task",
        )

        assert result["exit_code"] == 0
        assert "Custom output" in result["stdout"]
        assert "Custom error" in result["stderr"]

        # Check that artifacts were created with custom prefix
        artifacts_dir = (
            self.manager.storage.get_experiment_directory(self.experiment_id)
            / "artifacts"
        )

        stdout_file = artifacts_dir / "custom_task_stdout.txt"
        stderr_file = artifacts_dir / "custom_task_stderr.txt"

        assert stdout_file.exists()
        assert stderr_file.exists()
        assert "Custom output" in stdout_file.read_text()
        assert "Custom error" in stderr_file.read_text()

        # Ensure default prefix files were NOT created
        default_stdout = artifacts_dir / "script_stdout.txt"
        default_stderr = artifacts_dir / "script_stderr.txt"
        assert not default_stdout.exists()
        assert not default_stderr.exists()


class TestGetCliArgs:
    """Test get_cli_args() function."""

    def test_get_cli_args_standalone_mode(self):
        """Test get_cli_args() in standalone mode returns empty dict."""
        # Ensure no active context
        yanex._clear_current_experiment_id()

        cli_args = yanex.get_cli_args()
        assert cli_args == {}

    def test_get_cli_args_from_environment(self, tmp_path, monkeypatch):
        """Test get_cli_args() reads from environment variable in CLI mode."""
        import json

        # Create a test experiment
        manager = create_isolated_manager(tmp_path)

        # Create test script
        script = tmp_path / "test.py"
        script.write_text("print('test')")

        experiment_id = manager.create_experiment(
            script_path=script,
            config={},
            cli_args={
                "script": "test.py",
                "parallel": 3,
                "param": ["lr=0.01"],
                "tag": [],
            },
        )

        # Simulate CLI mode by setting environment variables
        monkeypatch.setenv("YANEX_EXPERIMENT_ID", experiment_id)
        monkeypatch.setenv("YANEX_CLI_ACTIVE", "1")
        monkeypatch.setenv(
            "YANEX_CLI_ARGS",
            json.dumps(
                {"script": "test.py", "parallel": 3, "param": ["lr=0.01"], "tag": []}
            ),
        )

        # Clear thread-local storage to force environment read
        yanex._clear_current_experiment_id()

        # Now get_cli_args() should read from environment
        cli_args = yanex.get_cli_args()
        assert cli_args["parallel"] == 3
        assert cli_args["param"] == ["lr=0.01"]
        assert cli_args["script"] == "test.py"
        assert cli_args["tag"] == []

        # Clean up
        monkeypatch.delenv("YANEX_EXPERIMENT_ID", raising=False)
        monkeypatch.delenv("YANEX_CLI_ACTIVE", raising=False)
        monkeypatch.delenv("YANEX_CLI_ARGS", raising=False)

    def test_get_cli_args_from_metadata(self, tmp_path, monkeypatch):
        """Test get_cli_args() reads from metadata in direct API mode."""
        # Point _get_experiment_manager to use our test directory
        monkeypatch.setenv("YANEX_EXPERIMENTS_DIR", str(tmp_path))

        manager = create_isolated_manager(tmp_path)

        # Create test script
        script = tmp_path / "test.py"
        script.write_text("print('test')")

        # Create experiment with CLI args
        experiment_id = manager.create_experiment(
            script_path=script,
            config={},
            cli_args={"script": "test.py", "parallel": 5, "tag": ["ml"]},
        )

        # Set thread-local context (direct API mode)
        yanex._set_current_experiment_id(experiment_id)

        try:
            # Should read from metadata
            cli_args = yanex.get_cli_args()
            assert cli_args["parallel"] == 5
            assert cli_args["script"] == "test.py"
            assert cli_args["tag"] == ["ml"]
        finally:
            yanex._clear_current_experiment_id()
            monkeypatch.delenv("YANEX_EXPERIMENTS_DIR", raising=False)

    def test_get_cli_args_empty_when_not_set(self, tmp_path):
        """Test get_cli_args() returns empty dict when CLI args not set."""
        manager = create_isolated_manager(tmp_path)

        # Create test script
        script = tmp_path / "test.py"
        script.write_text("print('test')")

        # Create experiment without CLI args
        experiment_id = manager.create_experiment(
            script_path=script,
            config={},
        )

        # Set thread-local context
        yanex._set_current_experiment_id(experiment_id)

        try:
            # Should return empty dict
            cli_args = yanex.get_cli_args()
            assert cli_args == {}
        finally:
            yanex._clear_current_experiment_id()

    def test_get_cli_args_handles_invalid_json(self, tmp_path, monkeypatch):
        """Test get_cli_args() handles invalid JSON gracefully."""

        # Create a test experiment
        manager = create_isolated_manager(tmp_path)

        # Create test script
        script = tmp_path / "test.py"
        script.write_text("print('test')")

        experiment_id = manager.create_experiment(
            script_path=script,
            config={},
        )

        # Simulate CLI mode with invalid JSON
        monkeypatch.setenv("YANEX_EXPERIMENT_ID", experiment_id)
        monkeypatch.setenv("YANEX_CLI_ACTIVE", "1")
        monkeypatch.setenv("YANEX_CLI_ARGS", "{invalid json}")

        # Clear thread-local storage to force environment read
        yanex._clear_current_experiment_id()

        # Should return empty dict on JSON error
        cli_args = yanex.get_cli_args()
        assert cli_args == {}

        # Clean up
        monkeypatch.delenv("YANEX_EXPERIMENT_ID", raising=False)
        monkeypatch.delenv("YANEX_CLI_ACTIVE", raising=False)
        monkeypatch.delenv("YANEX_CLI_ARGS", raising=False)
