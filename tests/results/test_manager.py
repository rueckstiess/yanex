"""
Tests for the ResultsManager class.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from yanex.core.manager import ExperimentManager
from yanex.results.experiment import Experiment
from yanex.results.manager import ResultsManager
from yanex.utils.exceptions import ExperimentNotFoundError


class TestResultsManager:
    """Test the ResultsManager class."""

    def teardown_method(self, method):
        """Clean up experiments after each test method."""
        try:
            from yanex.core.filtering import ExperimentFilter
            from yanex.core.manager import ExperimentManager

            manager = ExperimentManager()
            filter_obj = ExperimentFilter(manager=manager)
            test_experiments = filter_obj.filter_experiments(
                tags=["unit-tests"], limit=100
            )

            for exp in test_experiments:
                try:
                    if exp.get("status") == "running":
                        manager.cancel_experiment(exp["id"], "Test cleanup")
                    manager.delete_experiment(exp["id"])
                except Exception:
                    pass
        except Exception:
            pass

    @pytest.fixture
    def manager(self, isolated_experiments_dir, clean_git_repo):
        """Create a results manager for testing."""
        return ResultsManager(storage_path=isolated_experiments_dir)

    @pytest.fixture
    def experiment_manager(self, isolated_experiments_dir, clean_git_repo):
        """Create an experiment manager for setup."""
        return ExperimentManager(experiments_dir=isolated_experiments_dir)

    @pytest.fixture
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def sample_experiments(self, mock_capture_env, mock_git_info, experiment_manager):
        """Create multiple sample experiments for testing."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        experiments = []

        # Experiment 1: completed training
        exp1_id = experiment_manager.create_experiment(
            script_path=Path("train1.py"),
            name="training_run_1",
            config={"learning_rate": 0.001, "epochs": 100},
            tags=["training", "cnn", "unit-tests"],
            description="First training run",
        )
        experiment_manager.start_experiment(exp1_id)
        experiment_manager.storage.add_result_step(
            exp1_id, {"accuracy": 0.95, "loss": 0.05}
        )
        experiment_manager.complete_experiment(exp1_id)
        experiments.append(exp1_id)

        # Experiment 2: failed training
        exp2_id = experiment_manager.create_experiment(
            script_path=Path("train2.py"),
            name="training_run_2",
            config={"learning_rate": 0.01, "epochs": 50},
            tags=["training", "rnn"],
            description="Second training run",
        )
        experiment_manager.start_experiment(exp2_id)
        experiment_manager.storage.add_result_step(
            exp2_id, {"accuracy": 0.75, "loss": 0.25}
        )
        experiment_manager.fail_experiment(exp2_id, "Out of memory")
        experiments.append(exp2_id)

        # Experiment 3: running evaluation
        exp3_id = experiment_manager.create_experiment(
            script_path=Path("eval.py"),
            name="evaluation_1",
            config={"model_path": "/models/best.pt"},
            tags=["evaluation"],
            description="Model evaluation",
        )
        experiment_manager.start_experiment(exp3_id)
        experiment_manager.storage.add_result_step(
            exp3_id, {"precision": 0.92, "recall": 0.88}
        )
        # Set status to cancelled to avoid having multiple completed experiments
        experiment_manager.cancel_experiment(exp3_id, "Test cancellation")
        experiments.append(exp3_id)

        return experiments

    def test_manager_creation(self, isolated_experiments_dir):
        """Test creating ResultsManager instances."""
        # With custom path
        manager = ResultsManager(storage_path=isolated_experiments_dir)
        assert manager.storage_path == isolated_experiments_dir

        # With default path
        default_manager = ResultsManager()
        assert default_manager.storage_path is not None

    def test_find_experiments(self, manager, sample_experiments):
        """Test finding experiments with various filters."""
        exp1, exp2, exp3 = sample_experiments

        # Find all experiments
        all_experiments = manager.find()
        assert len(all_experiments) == 3

        # Find by status
        completed = manager.find(status="completed")
        assert len(completed) == 1
        assert completed[0]["id"] == exp1

        # Find by tags
        training = manager.find(tags=["training"])
        assert len(training) == 2
        training_ids = {exp["id"] for exp in training}
        assert training_ids == {exp1, exp2}

        # Find by multiple criteria
        completed_training = manager.find(status="completed", tags=["training"])
        assert len(completed_training) == 1
        assert completed_training[0]["id"] == exp1

    def test_get_experiment(self, manager, sample_experiments):
        """Test getting individual experiments."""
        exp1, exp2, exp3 = sample_experiments

        # Get existing experiment
        exp = manager.get_experiment(exp1)
        assert isinstance(exp, Experiment)
        assert exp.id == exp1
        assert exp.name == "training_run_1"

        # Test with non-existent experiment
        with pytest.raises(ExperimentNotFoundError):
            manager.get_experiment("nonexistent")

    def test_get_experiments(self, manager, sample_experiments):
        """Test getting multiple experiments as Experiment objects."""
        exp1, exp2, exp3 = sample_experiments

        # Get training experiments
        training_experiments = manager.get_experiments(tags=["training"])
        assert len(training_experiments) == 2
        assert all(isinstance(exp, Experiment) for exp in training_experiments)

        experiment_ids = {exp.id for exp in training_experiments}
        assert experiment_ids == {exp1, exp2}

    def test_get_latest(self, manager, sample_experiments):
        """Test getting latest experiment."""
        exp1, exp2, exp3 = sample_experiments

        # Get latest overall
        latest = manager.get_latest()
        assert isinstance(latest, Experiment)
        assert latest.id == exp3  # Should be the last created

        # Get latest training experiment
        latest_training = manager.get_latest(tags=["training"])
        assert latest_training.id == exp2  # Latest among training experiments

    def test_get_best(self, manager, sample_experiments):
        """Test getting best experiment by metric."""
        exp1, exp2, exp3 = sample_experiments

        # Get best accuracy (maximize)
        best_accuracy = manager.get_best("accuracy", maximize=True)
        assert best_accuracy.id == exp1  # Has accuracy 0.95

        # Get best loss (minimize)
        best_loss = manager.get_best("loss", maximize=False)
        assert best_loss.id == exp1  # Has loss 0.05

        # Get best among training experiments only
        best_training_accuracy = manager.get_best(
            "accuracy", maximize=True, tags=["training"]
        )
        assert best_training_accuracy.id == exp1

    def test_experiment_count(self, manager, sample_experiments):
        """Test counting experiments."""
        # Total count
        total = manager.get_experiment_count()
        assert total == 3

        # Count by status
        completed_count = manager.get_experiment_count(status="completed")
        assert completed_count == 1

        # Count by tags
        training_count = manager.get_experiment_count(tags=["training"])
        assert training_count == 2

    def test_experiment_exists(self, manager, sample_experiments):
        """Test checking experiment existence."""
        exp1, exp2, exp3 = sample_experiments

        # Existing experiments
        assert manager.experiment_exists(exp1) is True
        assert manager.experiment_exists(exp2) is True
        assert manager.experiment_exists(exp3) is True

        # Non-existent experiment
        assert manager.experiment_exists("nonexistent") is False

    def test_list_experiments(self, manager, sample_experiments):
        """Test listing experiments with limit."""
        # Default limit
        experiments = manager.list_experiments()
        assert len(experiments) <= 10  # Default limit
        assert len(experiments) == 3  # We only have 3 experiments

        # Custom limit
        limited = manager.list_experiments(limit=2)
        assert len(limited) == 2

        # With filters
        training_limited = manager.list_experiments(limit=1, tags=["training"])
        assert len(training_limited) == 1

    def test_archive_experiments(self, manager, sample_experiments):
        """Test archiving experiments."""
        exp1, exp2, exp3 = sample_experiments

        # Archive failed experiments
        archived_count = manager.archive_experiments(status="failed")
        assert archived_count == 1

        # Verify experiment is archived
        assert manager.experiment_exists(exp2, include_archived=True) is True
        assert manager.experiment_exists(exp2, include_archived=False) is False

    def test_delete_experiments(self, manager, sample_experiments):
        """Test deleting experiments."""
        exp1, exp2, exp3 = sample_experiments

        # Delete failed experiments
        deleted_count = manager.delete_experiments(status="failed")
        assert deleted_count == 1

        # Verify experiment is completely gone
        assert manager.experiment_exists(exp2, include_archived=True) is False
        assert manager.experiment_exists(exp2, include_archived=False) is False

        # Remaining experiments should still exist
        assert manager.experiment_exists(exp1, include_archived=False) is True
        assert manager.experiment_exists(exp3, include_archived=False) is True

    def test_export_experiments_json(self, manager, sample_experiments, tmp_path):
        """Test exporting experiments to JSON."""
        output_path = tmp_path / "export.json"

        # Export training experiments
        manager.export_experiments(str(output_path), format="json", tags=["training"])

        # Verify file was created
        assert output_path.exists()

        # Verify content
        import json

        with output_path.open() as f:
            data = json.load(f)

        assert len(data) == 2  # Two training experiments
        assert all("id" in exp for exp in data)
        assert all("name" in exp for exp in data)

    def test_string_representation(self, manager):
        """Test string representation of manager."""
        repr_str = repr(manager)
        assert "ResultsManager" in repr_str
        assert str(manager.storage_path) in repr_str

    @pytest.mark.skipif(
        condition=True,  # Skip by default since pandas might not be available
        reason="Requires pandas for DataFrame comparison",
    )
    def test_compare_experiments_with_pandas(self, manager, sample_experiments):
        """Test experiment comparison with pandas (if available)."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        # Compare training experiments
        df = manager.compare_experiments(
            tags=["training"],
            params=["learning_rate", "epochs"],
            metrics=["accuracy", "loss"],
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two training experiments

        # Check hierarchical columns
        assert isinstance(df.columns, pd.MultiIndex)

        # Check parameter columns exist
        assert ("param", "learning_rate") in df.columns
        assert ("param", "epochs") in df.columns

        # Check metric columns exist
        assert ("metric", "accuracy") in df.columns
        assert ("metric", "loss") in df.columns

    def test_compare_experiments_without_pandas(
        self, manager, sample_experiments, monkeypatch
    ):
        """Test experiment comparison gracefully handles missing pandas."""

        # Mock pandas import to fail
        def mock_import(name, *args, **kwargs):
            if name == "pandas":
                raise ImportError("No module named 'pandas'")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        # Should raise ImportError with helpful message
        with pytest.raises(ImportError, match="pandas is required"):
            manager.compare_experiments(tags=["training"])

    def test_edge_cases(self, manager):
        """Test edge cases and error conditions."""
        # Invalid status should raise ValueError
        with pytest.raises(ValueError, match="Invalid status"):
            manager.find(status="nonexistent_status")

        with pytest.raises(ValueError, match="Invalid status"):
            manager.get_latest(status="nonexistent_status")

        # get_best with no matches (using valid filter)
        best_none = manager.get_best("nonexistent_metric", status="completed")
        assert best_none is None
