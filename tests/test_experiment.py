"""
Tests for experiment public API module.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import yanex.experiment as experiment
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import ExperimentContextError


class TestThreadLocalState:
    """Test thread-local state management."""

    def test_no_active_context_error(self):
        """Test error when no active experiment context."""
        with pytest.raises(
            ExperimentContextError, match="No active experiment context"
        ):
            experiment.get_params()

    def test_get_current_experiment_id_no_context(self):
        """Test getting experiment ID without context raises error."""
        with pytest.raises(ExperimentContextError):
            experiment._get_current_experiment_id()

    def test_set_and_get_experiment_id(self):
        """Test setting and getting experiment ID."""
        test_id = "test12345"
        experiment._set_current_experiment_id(test_id)

        assert experiment._get_current_experiment_id() == test_id

        # Clean up
        experiment._clear_current_experiment_id()

    def test_clear_experiment_id(self):
        """Test clearing experiment ID."""
        experiment._set_current_experiment_id("test12345")
        experiment._clear_current_experiment_id()

        with pytest.raises(ExperimentContextError):
            experiment._get_current_experiment_id()

    def test_thread_isolation(self):
        """Test that thread-local state is isolated between threads."""
        results = {}

        def worker(thread_id):
            exp_id = f"exp{thread_id}"
            experiment._set_current_experiment_id(exp_id)
            time.sleep(0.1)  # Allow other threads to run
            results[thread_id] = experiment._get_current_experiment_id()
            experiment._clear_current_experiment_id()

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
    """Test experiment API functions."""

    def setup_method(self):
        """Set up test experiment context."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)
        self.manager = ExperimentManager(self.experiments_dir)

        # Create test experiment with proper directory structure
        self.experiment_id = "api12345"
        exp_dir = self.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True)
        (exp_dir / "artifacts").mkdir(parents=True)  # Create artifacts directory

        # Set up test metadata and config
        metadata = {
            "id": self.experiment_id,
            "status": "running",
            "name": "test-experiment",
        }
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        config = {"learning_rate": 0.01, "epochs": 10}
        self.manager.storage.save_config(self.experiment_id, config)

        # Set current experiment
        experiment._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        experiment._clear_current_experiment_id()

    @patch("yanex.experiment._get_experiment_manager")
    def test_get_params(self, mock_get_manager):
        """Test getting experiment parameters."""
        mock_get_manager.return_value = self.manager

        params = experiment.get_params()
        assert params["learning_rate"] == 0.01
        assert params["epochs"] == 10

    @patch("yanex.experiment._get_experiment_manager")
    def test_get_param(self, mock_get_manager):
        """Test getting specific parameter."""
        mock_get_manager.return_value = self.manager

        # Get existing parameter
        lr = experiment.get_param("learning_rate")
        assert lr == 0.01

        # Get non-existent parameter with default
        batch_size = experiment.get_param("batch_size", 32)
        assert batch_size == 32

        # Get non-existent parameter without default
        optimizer = experiment.get_param("optimizer")
        assert optimizer is None

    @patch("yanex.experiment._get_experiment_manager")
    def test_get_status(self, mock_get_manager):
        """Test getting experiment status."""
        mock_get_manager.return_value = self.manager

        status = experiment.get_status()
        assert status == "running"

    def test_get_experiment_id(self):
        """Test getting current experiment ID."""
        exp_id = experiment.get_experiment_id()
        assert exp_id == self.experiment_id

    @patch("yanex.experiment._get_experiment_manager")
    def test_get_metadata(self, mock_get_manager):
        """Test getting experiment metadata."""
        mock_get_manager.return_value = self.manager

        metadata = experiment.get_metadata()
        assert metadata["id"] == self.experiment_id
        assert metadata["status"] == "running"
        assert metadata["name"] == "test-experiment"

    @patch("yanex.experiment._get_experiment_manager")
    def test_log_results_basic(self, mock_get_manager):
        """Test basic result logging."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95, "loss": 0.05}
        experiment.log_results(results)

        # Verify results were saved
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        assert saved_results[0]["accuracy"] == 0.95
        assert saved_results[0]["loss"] == 0.05
        assert "step" in saved_results[0]

    @patch("yanex.experiment._get_experiment_manager")
    def test_log_results_with_step(self, mock_get_manager):
        """Test result logging with specific step."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.90}
        experiment.log_results(results, step=5)

        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert saved_results[0]["step"] == 5

    @patch("yanex.experiment._get_experiment_manager")
    @patch("builtins.print")
    def test_log_results_replacement_warning(self, mock_print, mock_get_manager):
        """Test warning when replacing existing step."""
        mock_get_manager.return_value = self.manager

        # Log first result
        experiment.log_results({"accuracy": 0.90}, step=1)

        # Log second result with same step
        experiment.log_results({"accuracy": 0.95}, step=1)

        # Should have printed warning
        mock_print.assert_called_with("Warning: Replacing existing results for step 1")

    @patch("yanex.experiment._get_experiment_manager")
    def test_log_artifact(self, mock_get_manager):
        """Test artifact logging."""
        mock_get_manager.return_value = self.manager

        # Create a test file
        test_file = Path(self.temp_dir) / "test_artifact.txt"
        test_file.write_text("test content")

        experiment.log_artifact("test.txt", test_file)

        # Verify artifact was saved
        artifact_path = (
            self.experiments_dir / self.experiment_id / "artifacts" / "test.txt"
        )
        assert artifact_path.exists()
        assert artifact_path.read_text() == "test content"

    @patch("yanex.experiment._get_experiment_manager")
    def test_log_text(self, mock_get_manager):
        """Test text artifact logging."""
        mock_get_manager.return_value = self.manager

        content = "This is test content"
        experiment.log_text(content, "output.txt")

        # Verify text artifact was saved
        artifact_path = (
            self.experiments_dir / self.experiment_id / "artifacts" / "output.txt"
        )
        assert artifact_path.exists()
        assert artifact_path.read_text() == content

    @patch("yanex.experiment._get_experiment_manager")
    def test_log_matplotlib_figure(self, mock_get_manager):
        """Test matplotlib figure logging."""
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
                experiment.log_matplotlib_figure(None, "plot.png")


class TestManualExperimentControl:
    """Test manual experiment control functions."""

    def setup_method(self):
        """Set up test experiment context."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)
        self.manager = ExperimentManager(self.experiments_dir)

        # Create test experiment
        self.experiment_id = "control123"
        exp_dir = self.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True)

        metadata = {"id": self.experiment_id, "status": "running"}
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        experiment._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        experiment._clear_current_experiment_id()

    @patch("yanex.experiment._get_experiment_manager")
    def test_completed(self, mock_get_manager):
        """Test manual experiment completion."""
        mock_get_manager.return_value = self.manager

        with pytest.raises(experiment._ExperimentCompletedException):
            experiment.completed()

        # Verify experiment was marked as completed
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

    @patch("yanex.experiment._get_experiment_manager")
    def test_fail(self, mock_get_manager):
        """Test manual experiment failure."""
        mock_get_manager.return_value = self.manager

        error_msg = "Something went wrong"
        with pytest.raises(experiment._ExperimentFailedException):
            experiment.fail(error_msg)

        # Verify experiment was marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert metadata["error_message"] == error_msg

    @patch("yanex.experiment._get_experiment_manager")
    def test_cancel(self, mock_get_manager):
        """Test manual experiment cancellation."""
        mock_get_manager.return_value = self.manager

        cancel_msg = "User requested cancellation"
        with pytest.raises(experiment._ExperimentCancelledException):
            experiment.cancel(cancel_msg)

        # Verify experiment was marked as cancelled
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "cancelled"
        assert metadata["cancellation_reason"] == cancel_msg


class TestExperimentContext:
    """Test ExperimentContext context manager."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)
        self.manager = ExperimentManager(self.experiments_dir)

        # Create test experiment in 'created' state
        self.experiment_id = "context123"
        exp_dir = self.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True)

        metadata = {"id": self.experiment_id, "status": "created", "started_at": None}
        self.manager.storage.save_metadata(self.experiment_id, metadata)

    @patch("yanex.experiment._get_experiment_manager")
    def test_context_manager_normal_exit(self, mock_get_manager):
        """Test normal context manager execution."""
        mock_get_manager.return_value = self.manager

        context = experiment.ExperimentContext(self.experiment_id)

        with context:
            # Verify experiment is running and context is set
            assert experiment._get_current_experiment_id() == self.experiment_id
            status = self.manager.get_experiment_status(self.experiment_id)
            assert status == "running"

        # After exit, should be completed and context cleared
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

        with pytest.raises(ExperimentContextError):
            experiment._get_current_experiment_id()

    @patch("yanex.experiment._get_experiment_manager")
    def test_context_manager_with_exception(self, mock_get_manager):
        """Test context manager with unhandled exception."""
        mock_get_manager.return_value = self.manager

        context = experiment.ExperimentContext(self.experiment_id)

        with pytest.raises(ValueError):
            with context:
                raise ValueError("Test error")

        # Should be marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert "ValueError: Test error" in metadata["error_message"]

    @patch("yanex.experiment._get_experiment_manager")
    def test_context_manager_keyboard_interrupt(self, mock_get_manager):
        """Test context manager with KeyboardInterrupt."""
        mock_get_manager.return_value = self.manager

        context = experiment.ExperimentContext(self.experiment_id)

        with pytest.raises(KeyboardInterrupt):
            with context:
                raise KeyboardInterrupt()

        # Should be marked as cancelled
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "cancelled"
        assert "Interrupted by user" in metadata["cancellation_reason"]

    @patch("yanex.experiment._get_experiment_manager")
    def test_context_manager_manual_completion(self, mock_get_manager):
        """Test context manager with manual completion."""
        mock_get_manager.return_value = self.manager

        context = experiment.ExperimentContext(self.experiment_id)

        with context:
            # Manually complete the experiment
            experiment.completed()

        # Should be marked as completed, not completed again
        status = self.manager.get_experiment_status(self.experiment_id)
        assert status == "completed"

    @patch("yanex.experiment._get_experiment_manager")
    def test_context_manager_manual_failure(self, mock_get_manager):
        """Test context manager with manual failure."""
        mock_get_manager.return_value = self.manager

        context = experiment.ExperimentContext(self.experiment_id)

        with context:
            experiment.fail("Manual failure")

        # Should be marked as failed
        metadata = self.manager.get_experiment_metadata(self.experiment_id)
        assert metadata["status"] == "failed"
        assert metadata["error_message"] == "Manual failure"


class TestExperimentCreation:
    """Test experiment creation functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)

    @patch("yanex.experiment._get_experiment_manager")
    @patch("yanex.core.manager.validate_clean_working_directory")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_create_experiment(
        self, mock_capture_env, mock_git_info, mock_validate_git, mock_get_manager
    ):
        """Test create_experiment function."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        script_path = Path(__file__)
        context = experiment.create_experiment(
            script_path=script_path,
            name="test-experiment",
            config={"learning_rate": 0.01},
            tags=["ml", "test"],
            description="Test experiment",
        )

        # Should return ExperimentContext
        assert isinstance(context, experiment.ExperimentContext)
        assert len(context.experiment_id) == 8

    @patch("yanex.experiment._get_experiment_manager")
    def test_create_context_existing_experiment(self, mock_get_manager):
        """Test create_context with existing experiment."""
        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        # Create experiment
        experiment_id = "test1234"
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True)
        metadata = {"id": experiment_id, "status": "created"}
        manager.storage.save_metadata(experiment_id, metadata)

        # Create context
        context = experiment.create_context(experiment_id)
        assert isinstance(context, experiment.ExperimentContext)
        assert context.experiment_id == experiment_id

    @patch("yanex.experiment._get_experiment_manager")
    def test_create_context_nonexistent_experiment(self, mock_get_manager):
        """Test create_context with non-existent experiment."""
        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        from yanex.utils.exceptions import ExperimentNotFoundError

        with pytest.raises(ExperimentNotFoundError):
            experiment.create_context("nonexistent")


class TestExperimentRun:
    """Test experiment.run() function."""

    def test_run_not_implemented(self):
        """Test that run() raises appropriate error."""
        with pytest.raises(
            ExperimentContextError,
            match="Direct experiment.run\\(\\) is not yet implemented",
        ):
            experiment.run()
