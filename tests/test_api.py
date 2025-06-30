"""
Tests for experiment public API module.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import yanex
from yanex.core.manager import ExperimentManager
# from yanex.utils.exceptions import ExperimentContextError  # Unused import


class TestThreadLocalState:
    """Test thread-local state management."""

    def test_no_active_context_standalone_mode(self):
        """Test standalone mode when no active experiment context."""
        # In standalone mode, get_params() should return empty dict
        params = yanex.get_params()
        assert params == {}

        # And is_standalone() should return True
        assert yanex.is_standalone() is True
        assert yanex.has_context() is False

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

        config = {
            "learning_rate": 0.01,
            "epochs": 10,
            "model": {
                "architecture": "resnet",
                "layers": 18,
                "optimizer": {
                    "type": "adam",
                    "lr": 0.001,
                    "weight_decay": 1e-4
                }
            },
            "data": {
                "batch_size": 32,
                "augmentation": True
            }
        }
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

        # Test empty key parts (though this shouldn't happen in practice)
        # This tests robustness

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
    def test_get_metadata(self, mock_get_manager):
        """Test getting experiment metadata."""
        mock_get_manager.return_value = self.manager

        metadata = yanex.get_metadata()
        assert metadata["id"] == self.experiment_id
        assert metadata["status"] == "running"
        assert metadata["name"] == "test-experiment"

    @patch("yanex.api._get_experiment_manager")
    def test_log_results_basic(self, mock_get_manager):
        """Test basic result logging."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.95, "loss": 0.05}
        yanex.log_results(results)

        # Verify results were saved
        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert len(saved_results) == 1
        assert saved_results[0]["accuracy"] == 0.95
        assert saved_results[0]["loss"] == 0.05
        assert "step" in saved_results[0]

    @patch("yanex.api._get_experiment_manager")
    def test_log_results_with_step(self, mock_get_manager):
        """Test result logging with specific step."""
        mock_get_manager.return_value = self.manager

        results = {"accuracy": 0.90}
        yanex.log_results(results, step=5)

        saved_results = self.manager.storage.load_results(self.experiment_id)
        assert saved_results[0]["step"] == 5

    @patch("yanex.api._get_experiment_manager")
    @patch("builtins.print")
    def test_log_results_replacement_warning(self, mock_print, mock_get_manager):
        """Test warning when replacing existing step."""
        mock_get_manager.return_value = self.manager

        # Log first result
        yanex.log_results({"accuracy": 0.90}, step=1)

        # Log second result with same step
        yanex.log_results({"accuracy": 0.95}, step=1)

        # Should have printed warning
        mock_print.assert_called_with("Warning: Replacing existing results for step 1")

    @patch("yanex.api._get_experiment_manager")
    def test_log_artifact(self, mock_get_manager):
        """Test artifact logging."""
        mock_get_manager.return_value = self.manager

        # Create a test file
        test_file = Path(self.temp_dir) / "test_artifact.txt"
        test_file.write_text("test content")

        yanex.log_artifact("test.txt", test_file)

        # Verify artifact was saved
        artifact_path = self.experiments_dir / self.experiment_id / "artifacts" / "test.txt"
        assert artifact_path.exists()
        assert artifact_path.read_text() == "test content"

    @patch("yanex.api._get_experiment_manager")
    def test_log_text(self, mock_get_manager):
        """Test text artifact logging."""
        mock_get_manager.return_value = self.manager

        content = "This is test content"
        yanex.log_text(content, "output.txt")

        # Verify text artifact was saved
        artifact_path = self.experiments_dir / self.experiment_id / "artifacts" / "output.txt"
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
    """Test experiment creation functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)

    @patch("yanex.api._get_experiment_manager")
    @patch("yanex.core.manager.validate_clean_working_directory")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_create_experiment(self, mock_capture_env, mock_git_info, mock_validate_git, mock_get_manager):
        """Test create_experiment function."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        script_path = Path(__file__)
        context = yanex.create_experiment(
            script_path=script_path,
            name="test-experiment",
            config={"learning_rate": 0.01},
            tags=["ml", "test"],
            description="Test experiment",
        )

        # Should return ExperimentContext
        assert isinstance(context, yanex.ExperimentContext)
        assert len(context.experiment_id) == 8

    @patch("yanex.api._get_experiment_manager")
    def test_create_context_existing_experiment(self, mock_get_manager):
        """Test create_context with existing yanex."""
        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        # Create experiment
        experiment_id = "test1234"
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(parents=True)
        metadata = {"id": experiment_id, "status": "created"}
        manager.storage.save_metadata(experiment_id, metadata)

        # Create context
        context = yanex.create_context(experiment_id)
        assert isinstance(context, yanex.ExperimentContext)
        assert context.experiment_id == experiment_id

    @patch("yanex.api._get_experiment_manager")
    def test_create_context_nonexistent_experiment(self, mock_get_manager):
        """Test create_context with non-existent yanex."""
        manager = ExperimentManager(self.experiments_dir)
        mock_get_manager.return_value = manager

        from yanex.utils.exceptions import ExperimentNotFoundError

        with pytest.raises(ExperimentNotFoundError):
            yanex.create_context("nonexistent")
