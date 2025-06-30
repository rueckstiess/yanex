"""
Tests for experiment API standalone mode functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import yanex
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import ExperimentContextError


class TestStandaloneMode:
    """Test experiment API in standalone mode (no experiment context)."""

    def test_mode_detection(self):
        """Test mode detection functions."""
        assert yanex.is_standalone() is True
        assert yanex.has_context() is False

    def test_get_params_standalone(self):
        """Test get_params returns empty dict in standalone mode."""
        params = yanex.get_params()
        assert params == {}

    def test_get_param_standalone(self):
        """Test get_param returns defaults in standalone mode."""
        # Should return defaults
        assert yanex.get_param("learning_rate", 0.01) == 0.01
        assert yanex.get_param("epochs", 10) == 10
        assert yanex.get_param("model_type", "linear") == "linear"

        # Should return None for non-existent param without default
        assert yanex.get_param("nonexistent") is None

    def test_get_param_dot_notation_standalone(self):
        """Test get_param with dot notation returns defaults in standalone mode."""
        # Should return defaults for nested keys
        assert yanex.get_param("model.learning_rate", 0.001) == 0.001
        assert yanex.get_param("model.architecture", "resnet") == "resnet"
        assert yanex.get_param("data.batch_size", 32) == 32
        
        # Test multi-level nesting
        assert yanex.get_param("model.optimizer.type", "adam") == "adam"
        assert yanex.get_param("model.optimizer.lr", 1e-3) == 1e-3
        
        # Should return None for non-existent nested param without default
        assert yanex.get_param("model.nonexistent") is None
        assert yanex.get_param("nonexistent.nested.path") is None
        
        # Should not fail with deep nesting
        assert yanex.get_param("a.b.c.d.e.f", "deep_default") == "deep_default"

    def test_get_experiment_id_standalone(self):
        """Test get_experiment_id returns None in standalone mode."""
        assert yanex.get_experiment_id() is None

    def test_get_status_standalone(self):
        """Test get_status returns None in standalone mode."""
        assert yanex.get_status() is None

    def test_get_metadata_standalone(self):
        """Test get_metadata returns empty dict in standalone mode."""
        metadata = yanex.get_metadata()
        assert metadata == {}

    def test_log_results_standalone(self):
        """Test log_results is no-op in standalone mode."""
        # Should not raise any exceptions
        yanex.log_results({"accuracy": 0.95, "loss": 0.05})
        yanex.log_results({"accuracy": 0.97}, step=1)

    def test_log_artifact_standalone(self):
        """Test log_artifact is no-op in standalone mode."""
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = Path(temp_file.name)
            # Should not raise any exceptions
            yanex.log_artifact("test.txt", temp_path)

    def test_log_text_standalone(self):
        """Test log_text is no-op in standalone mode."""
        # Should not raise any exceptions
        yanex.log_text("Hello standalone", "test.txt")

    def test_log_matplotlib_figure_standalone(self):
        """Test log_matplotlib_figure is no-op in standalone mode."""
        # Should not raise any exceptions (no matplotlib import needed)
        yanex.log_matplotlib_figure(None, "plot.png")

    def test_manual_control_functions_error(self):
        """Test manual control functions raise errors in standalone mode."""
        with pytest.raises(
            ExperimentContextError, match="No active experiment context"
        ):
            yanex.completed()

        with pytest.raises(
            ExperimentContextError, match="No active experiment context"
        ):
            yanex.fail("test error")

        with pytest.raises(
            ExperimentContextError, match="No active experiment context"
        ):
            yanex.cancel("test cancellation")


class TestContextMode:
    """Test experiment API in context mode (with active experiment)."""

    def setup_method(self):
        """Set up test experiment context."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiments_dir = Path(self.temp_dir)
        self.manager = ExperimentManager(self.experiments_dir)

        # Create test experiment
        self.experiment_id = "test12345"
        exp_dir = self.experiments_dir / self.experiment_id
        exp_dir.mkdir(parents=True)
        (exp_dir / "artifacts").mkdir(parents=True)

        metadata = {
            "id": self.experiment_id,
            "status": "running",
            "name": "test-experiment",
        }
        self.manager.storage.save_metadata(self.experiment_id, metadata)

        config = {"learning_rate": 0.01, "epochs": 10}
        self.manager.storage.save_config(self.experiment_id, config)

        # Set current experiment
        yanex._set_current_experiment_id(self.experiment_id)

    def teardown_method(self):
        """Clean up after test."""
        yanex._clear_current_experiment_id()

    def test_mode_detection_with_context(self):
        """Test mode detection functions with active context."""
        assert yanex.is_standalone() is False
        assert yanex.has_context() is True

    @patch("yanex.api._get_experiment_manager")
    def test_get_params_with_context(self, mock_get_manager):
        """Test get_params returns actual config in context mode."""
        mock_get_manager.return_value = self.manager

        params = yanex.get_params()
        assert params["learning_rate"] == 0.01
        assert params["epochs"] == 10

    @patch("yanex.api._get_experiment_manager")
    def test_get_param_with_context(self, mock_get_manager):
        """Test get_param returns actual values in context mode."""
        mock_get_manager.return_value = self.manager

        assert yanex.get_param("learning_rate") == 0.01
        assert yanex.get_param("epochs") == 10

        # Should still use default for non-existent param
        assert yanex.get_param("batch_size", 32) == 32

    def test_get_experiment_id_with_context(self):
        """Test get_experiment_id returns actual ID in context mode."""
        assert yanex.get_experiment_id() == self.experiment_id

    @patch("yanex.api._get_experiment_manager")
    def test_get_status_with_context(self, mock_get_manager):
        """Test get_status returns actual status in context mode."""
        mock_get_manager.return_value = self.manager
        assert yanex.get_status() == "running"

    @patch("yanex.api._get_experiment_manager")
    def test_get_metadata_with_context(self, mock_get_manager):
        """Test get_metadata returns actual metadata in context mode."""
        mock_get_manager.return_value = self.manager

        metadata = yanex.get_metadata()
        assert metadata["id"] == self.experiment_id
        assert metadata["status"] == "running"
        assert metadata["name"] == "test-experiment"



class TestModeTransition:
    """Test transitioning between standalone and context modes."""

    @patch("yanex.api._get_experiment_manager")
    def test_standalone_to_context_transition(self, mock_get_manager):
        """Test transition from standalone to context mode."""
        # Start in standalone mode
        assert yanex.is_standalone() is True
        assert yanex.get_experiment_id() is None
        assert yanex.get_param("test", "default") == "default"

        # Set up experiment context
        temp_dir = tempfile.mkdtemp()
        experiments_dir = Path(temp_dir)
        manager = ExperimentManager(experiments_dir)
        mock_get_manager.return_value = manager

        experiment_id = "transition123"
        exp_dir = experiments_dir / experiment_id
        exp_dir.mkdir(parents=True)

        config = {"test": "context_value"}
        manager.storage.save_config(experiment_id, config)

        try:
            # Transition to context mode
            yanex._set_current_experiment_id(experiment_id)

            # Should now be in context mode
            assert yanex.is_standalone() is False
            assert yanex.get_experiment_id() == experiment_id
            assert yanex.get_param("test", "default") == "context_value"

            # Clear context - should return to standalone
            yanex._clear_current_experiment_id()

            # Should be back in standalone mode
            assert yanex.is_standalone() is True
            assert yanex.get_experiment_id() is None
            assert yanex.get_param("test", "default") == "default"

        finally:
            # Ensure cleanup
            yanex._clear_current_experiment_id()
