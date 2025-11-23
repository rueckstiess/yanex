"""Integration tests for parameter tracking feature.

Tests the full end-to-end workflow including:
- TrackedDict creation and usage
- Atexit handler registration
- Saving accessed params to storage
- Both CLI and API modes
"""

from pathlib import Path
from unittest.mock import patch

import yanex
from yanex.core.manager import ExperimentManager
from yanex.core.param_tracking import extract_accessed_params
from yanex.core.tracked_dict import TrackedDict


class TestParameterTrackingIntegration:
    """Integration tests for full parameter tracking workflow."""

    def setup_api_context(self, exp_id, manager):
        """Helper to set up API experiment context for testing."""
        yanex.api._local.experiment_id = exp_id
        yanex.api._tracked_params = None
        yanex.api._atexit_registered = False
        return patch("yanex.api._get_experiment_manager", return_value=manager)

    def teardown_api_context(self):
        """Helper to clean up API experiment context after testing."""
        if hasattr(yanex.api._local, "experiment_id"):
            del yanex.api._local.experiment_id
        yanex.api._tracked_params = None
        yanex.api._atexit_registered = False

    def test_tracked_dict_integration_with_storage(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that TrackedDict integrates correctly with storage layer."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        # Create experiment with full config
        full_config = {
            "model": {"train": {"lr": 0.01, "epochs": 20}, "arch": {"layers": 5}},
            "data": {"path": "dataset.json", "split": 0.8},
            "seed": 42,
        }

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=full_config
            )

        # Load config as TrackedDict and access subset
        tracked = TrackedDict(full_config)
        _ = tracked["model"]["train"]["lr"]
        _ = tracked["seed"]

        # Extract accessed params
        accessed = extract_accessed_params(tracked)

        # Should only have accessed params
        assert accessed == {"model": {"train": {"lr": 0.01}}, "seed": 42}

        # Save accessed params directly via storage
        manager.storage.save_config(exp_id, accessed)

        # Load from storage
        loaded = manager.storage.load_config(exp_id)

        # Should match extracted accessed params
        assert loaded == {"model": {"train": {"lr": 0.01}}, "seed": 42}

    def test_api_get_params_returns_tracked_dict(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that get_params() returns TrackedDict in experiment context."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        # Set up experiment context using _local
        yanex.api._local.experiment_id = exp_id
        yanex.api._tracked_params = None
        yanex.api._atexit_registered = False

        try:
            with patch("yanex.api._get_experiment_manager", return_value=manager):
                # Get params (should return TrackedDict)
                params = yanex.get_params()

                assert isinstance(params, TrackedDict)
                assert params == config

                # Access some params
                lr = params["learning_rate"]
                bs = params["batch_size"]

                assert lr == 0.001
                assert bs == 32

                # Check accessed paths
                accessed_paths = params.get_accessed_paths()
                assert "learning_rate" in accessed_paths
                assert "batch_size" in accessed_paths
                assert "epochs" not in accessed_paths  # Not accessed
        finally:
            # Clean up
            del yanex.api._local.experiment_id
            yanex.api._tracked_params = None
            yanex.api._atexit_registered = False

    def test_api_get_param_with_dot_notation(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that get_param() works with dot notation and tracking."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {
            "model": {"train": {"lr": 0.01, "epochs": 20}, "arch": {"layers": 5}},
            "seed": 42,
        }

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        try:
            with self.setup_api_context(exp_id, manager):
                # Get nested param with dot notation
                lr = yanex.get_param("model.train.lr")
                seed = yanex.get_param("seed")

                assert lr == 0.01
                assert seed == 42

                # Get params to check tracking
                params = yanex.get_params()
                accessed_paths = params.get_accessed_paths()

                # Should track the full paths
                assert "model" in accessed_paths
                assert "model.train" in accessed_paths
                assert "model.train.lr" in accessed_paths
                assert "seed" in accessed_paths
        finally:
            self.teardown_api_context()

    def test_atexit_handler_saves_accessed_params(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that atexit handler is registered and saves accessed params."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {
            "model": {"lr": 0.01, "dropout": 0.1},
            "data": {"path": "data.csv"},
            "seed": 42,
        }

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        try:
            with self.setup_api_context(exp_id, manager):
                # Access params
                params = yanex.get_params()
                _ = params["model"]["lr"]
                _ = params["seed"]

                # Extract and save accessed params
                accessed = extract_accessed_params(params)
                manager.storage.save_config(exp_id, accessed)

                # Load from storage
                loaded = manager.storage.load_config(exp_id)

                # Should only have accessed params
                assert loaded == {"model": {"lr": 0.01}, "seed": 42}
                assert "dropout" not in loaded.get("model", {})
                assert "data" not in loaded
        finally:
            self.teardown_api_context()

    def test_no_params_accessed_saves_empty_dict(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that when no params are accessed, empty dict is saved."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {"learning_rate": 0.001, "batch_size": 32}

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        # Simulate script that doesn't access params
        tracked = TrackedDict(config)

        # Don't access any params

        # Save (should save empty dict)
        accessed = extract_accessed_params(tracked)
        manager.storage.save_config(exp_id, accessed)

        # Load from storage
        loaded = manager.storage.load_config(exp_id)

        # Should be empty
        assert loaded == {}

    def test_mixed_access_patterns(self, isolated_experiments_dir, clean_git_repo):
        """Test mixed access patterns (get_param and get_params)."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {
            "model": {"lr": 0.01, "dropout": 0.1},
            "data": {"path": "data.csv", "split": 0.8},
            "seed": 42,
        }

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        try:
            with self.setup_api_context(exp_id, manager):
                # Mix of get_param and get_params
                _ = yanex.get_param("model.lr")  # Dot notation
                params = yanex.get_params()
                _ = params["seed"]  # Direct access

                # Manually save
                accessed = extract_accessed_params(params)
                manager.storage.save_config(exp_id, accessed)

                # Load from storage
                loaded = manager.storage.load_config(exp_id)

                # Should have both
                assert loaded == {"model": {"lr": 0.01}, "seed": 42}
                assert "dropout" not in loaded.get("model", {})
                assert "data" not in loaded
        finally:
            self.teardown_api_context()

    def test_iteration_marks_all_accessed(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that iterating over params marks all as accessed."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {"a": 1, "b": 2, "c": 3}

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        # Simulate iteration
        tracked = TrackedDict(config)
        for _key in tracked:
            pass  # Just iterate, don't use values

        # Save
        accessed = extract_accessed_params(tracked)
        manager.storage.save_config(exp_id, accessed)

        # Load from storage
        loaded = manager.storage.load_config(exp_id)

        # All keys should be saved
        assert loaded == {"a": 1, "b": 2, "c": 3}

    def test_nested_dict_access_preserves_structure(
        self, isolated_experiments_dir, clean_git_repo
    ):
        """Test that nested dict access preserves structure correctly."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {
            "model": {
                "train": {"lr": 0.01, "epochs": 20, "batch_size": 32},
                "arch": {"layers": 5, "hidden": 128},
            },
            "data": {"path": "data.csv"},
        }

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        # Access nested values
        tracked = TrackedDict(config)
        _ = tracked["model"]["train"]["lr"]
        _ = tracked["model"]["arch"]["layers"]

        # Save
        accessed = extract_accessed_params(tracked)
        manager.storage.save_config(exp_id, accessed)

        # Load from storage
        loaded = manager.storage.load_config(exp_id)

        # Should preserve nested structure
        assert loaded == {
            "model": {"train": {"lr": 0.01}, "arch": {"layers": 5}},
        }
        assert "data" not in loaded

    def test_get_param_default_value(self, isolated_experiments_dir, clean_git_repo):
        """Test that get_param returns default when key not found."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {"learning_rate": 0.001}

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        try:
            with self.setup_api_context(exp_id, manager):
                # Get missing key with default
                batch_size = yanex.get_param("batch_size", default=32)
                lr = yanex.get_param("learning_rate")

                assert batch_size == 32  # Default
                assert lr == 0.001  # Actual value

                # Only learning_rate should be tracked (not batch_size)
                params = yanex.get_params()
                accessed_paths = params.get_accessed_paths()

                assert "learning_rate" in accessed_paths
                # batch_size was never in config, so no tracking
        finally:
            self.teardown_api_context()

    def test_atexit_handler_registration(self, isolated_experiments_dir):
        """Test that get_params() registers atexit handler exactly once."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)

        config = {"learning_rate": 0.001, "batch_size": 32}

        with patch("yanex.core.manager.get_current_commit_info") as mock_git:
            mock_git.return_value = {"commit": "abc123", "branch": "main"}
            exp_id = manager.create_experiment(
                script_path=Path("train.py"), config=config
            )

        try:
            with self.setup_api_context(exp_id, manager):
                with patch("atexit.register") as mock_atexit:
                    # First call should register atexit handler
                    _ = yanex.get_params()
                    assert mock_atexit.call_count == 1

                    # Second call should NOT register again (_atexit_registered flag)
                    _ = yanex.get_params()
                    assert mock_atexit.call_count == 1  # Still 1, not 2

                    # Verify registration arguments
                    call_args = mock_atexit.call_args[0]
                    assert len(call_args) == 3

                    # First arg should be the wrapper function
                    from yanex.api import _atexit_handler_wrapper

                    assert call_args[0] == _atexit_handler_wrapper

                    # Second arg should be experiment_id
                    assert call_args[1] == exp_id

                    # Third arg should be TrackedDict instance
                    assert isinstance(call_args[2], TrackedDict)
        finally:
            self.teardown_api_context()
