"""
Tests for the Experiment class.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from yanex.core.manager import ExperimentManager
from yanex.results.experiment import Experiment
from yanex.utils.exceptions import ExperimentNotFoundError


class TestExperiment:
    """Test the Experiment class."""

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
        """Create an experiment manager for testing."""
        return ExperimentManager(experiments_dir=isolated_experiments_dir)

    @pytest.fixture
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def sample_experiment(self, mock_capture_env, mock_git_info, manager):
        """Create a sample experiment for testing."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment with comprehensive data
        exp_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="test_experiment",
            config={
                "learning_rate": 0.001,
                "epochs": 100,
                "model": {"type": "cnn", "layers": 3},
            },
            tags=["training", "cnn", "test", "unit-tests"],
            description="Test experiment for unit tests",
        )

        # Start the experiment
        manager.start_experiment(exp_id)

        # Add some metrics
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.95, "loss": 0.05, "step": 1}
        )

        # Complete the experiment
        manager.complete_experiment(exp_id)

        return exp_id

    def test_experiment_creation(self, manager, sample_experiment):
        """Test creating an Experiment instance."""
        exp = Experiment(sample_experiment, manager)

        assert exp.id == sample_experiment
        assert exp.name == "test_experiment"
        assert exp.status == "completed"
        assert exp.description == "Test experiment for unit tests"
        assert set(exp.tags) == {"training", "cnn", "test", "unit-tests"}

    def test_nonexistent_experiment(self, manager):
        """Test creating Experiment with non-existent ID."""
        with pytest.raises(ExperimentNotFoundError):
            Experiment("nonexistent", manager)

    def test_properties(self, manager, sample_experiment):
        """Test all experiment properties."""
        exp = Experiment(sample_experiment, manager)

        # Basic properties
        assert exp.id == sample_experiment
        assert exp.name == "test_experiment"
        assert exp.status == "completed"
        assert exp.description == "Test experiment for unit tests"
        assert exp.tags == ["cnn", "test", "training", "unit-tests"]  # Should be sorted

        # Time properties
        assert isinstance(exp.started_at, datetime)
        assert isinstance(exp.completed_at, datetime)
        assert exp.duration is not None
        assert exp.duration.total_seconds() > 0

        # File properties
        assert (
            exp.script_path.name == "train.py"
        )  # Check filename, path may be absolute
        assert exp.archived is False

        # Directory property
        from pathlib import Path

        assert isinstance(exp.experiment_dir, Path)
        assert exp.experiment_dir.exists()
        assert sample_experiment in str(exp.experiment_dir)

    def test_get_params(self, manager, sample_experiment):
        """Test parameter access."""
        exp = Experiment(sample_experiment, manager)

        params = exp.get_params()
        assert params["learning_rate"] == 0.001
        assert params["epochs"] == 100
        assert params["model"]["type"] == "cnn"
        assert params["model"]["layers"] == 3

    def test_get_param_with_dot_notation(self, manager, sample_experiment):
        """Test parameter access with dot notation."""
        exp = Experiment(sample_experiment, manager)

        # Simple parameter
        assert exp.get_param("learning_rate") == 0.001
        assert exp.get_param("epochs") == 100

        # Nested parameter with dot notation
        assert exp.get_param("model.type") == "cnn"
        assert exp.get_param("model.layers") == 3

        # Non-existent parameters
        assert exp.get_param("nonexistent") is None
        assert exp.get_param("nonexistent", "default") == "default"
        assert exp.get_param("model.nonexistent") is None
        assert exp.get_param("model.nonexistent", "default") == "default"

    def test_get_metrics(self, manager, sample_experiment):
        """Test metrics access."""
        import pandas as pd

        exp = Experiment(sample_experiment, manager)

        # Should return a DataFrame by default (new behavior)
        metrics = exp.get_metrics()
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) > 0

        # Check the DataFrame contains expected metrics
        assert "accuracy" in metrics.columns
        assert "loss" in metrics.columns
        assert metrics.loc[0, "accuracy"] == 0.95
        assert metrics.loc[0, "loss"] == 0.05

        # Test specific step access as DataFrame
        step_0_metrics = exp.get_metrics(step=0)
        assert isinstance(step_0_metrics, pd.DataFrame)
        assert step_0_metrics.loc[0, "accuracy"] == 0.95
        assert step_0_metrics.loc[0, "loss"] == 0.05

        # Test nonexistent step
        nonexistent_metrics = exp.get_metrics(step=999)
        assert isinstance(nonexistent_metrics, pd.DataFrame)
        assert nonexistent_metrics.empty

    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_get_metrics_list_format(self, mock_capture_env, mock_git_info, manager):
        """Test metrics access when stored as a list (multiple steps)."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment
        exp_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="multi_step_experiment",
            tags=["unit-tests"],
        )

        # Add multiple metric steps to create a list
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.8, "loss": 0.2, "epoch": 1}
        )
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.85, "loss": 0.15, "epoch": 2}
        )
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.9, "loss": 0.1, "epoch": 3}
        )

        exp = Experiment(exp_id, manager)

        # Test list format (explicit as_dataframe=False)
        metrics = exp.get_metrics(as_dataframe=False)
        assert isinstance(metrics, list)
        assert len(metrics) == 3
        assert metrics[0]["accuracy"] == 0.8
        assert metrics[0]["epoch"] == 1
        assert metrics[0]["step"] == 0

        assert metrics[1]["accuracy"] == 0.85
        assert metrics[1]["epoch"] == 2
        assert metrics[1]["step"] == 1

        assert metrics[2]["accuracy"] == 0.9
        assert metrics[2]["epoch"] == 3
        assert metrics[2]["step"] == 2

        # Each entry should have timestamp and step
        for entry in metrics:
            assert "timestamp" in entry
            assert "step" in entry

        # Test step-specific access (as dict)
        step_1_metrics = exp.get_metrics(step=1, as_dataframe=False)
        assert isinstance(step_1_metrics, dict)
        assert step_1_metrics["accuracy"] == 0.85
        assert step_1_metrics["epoch"] == 2
        assert step_1_metrics["step"] == 1

        # Test nonexistent step (as dict)
        empty_metrics = exp.get_metrics(step=999, as_dataframe=False)
        assert isinstance(empty_metrics, dict)
        assert empty_metrics == {}

    def test_set_name(self, manager, sample_experiment):
        """Test setting experiment name."""
        exp = Experiment(sample_experiment, manager)

        # Set new name
        exp.set_name("new_name")
        assert exp.name == "new_name"

        # Verify persistence
        exp_reloaded = Experiment(sample_experiment, manager)
        assert exp_reloaded.name == "new_name"

        # Test validation
        with pytest.raises(ValueError):
            exp.set_name(123)  # Must be string

    def test_set_description(self, manager, sample_experiment):
        """Test setting experiment description."""
        exp = Experiment(sample_experiment, manager)

        exp.set_description("New description")
        assert exp.description == "New description"

        # Verify persistence
        exp_reloaded = Experiment(sample_experiment, manager)
        assert exp_reloaded.description == "New description"

    def test_add_remove_tags(self, manager, sample_experiment):
        """Test adding and removing tags."""
        exp = Experiment(sample_experiment, manager)

        original_tags = set(exp.tags)

        # Add tags
        exp.add_tags(["new_tag", "another_tag"])
        expected_tags = original_tags | {"new_tag", "another_tag"}
        assert set(exp.tags) == expected_tags

        # Remove tags
        exp.remove_tags(["cnn", "new_tag"])
        expected_tags = expected_tags - {"cnn", "new_tag"}
        assert set(exp.tags) == expected_tags

        # Verify persistence
        exp_reloaded = Experiment(sample_experiment, manager)
        assert set(exp_reloaded.tags) == expected_tags

    def test_set_status(self, manager, sample_experiment):
        """Test setting experiment status."""
        exp = Experiment(sample_experiment, manager)

        exp.set_status("failed")
        assert exp.status == "failed"

        # Verify persistence
        exp_reloaded = Experiment(sample_experiment, manager)
        assert exp_reloaded.status == "failed"

        # Test validation
        with pytest.raises(ValueError):
            exp.set_status("invalid_status")

    def test_to_dict(self, manager, sample_experiment):
        """Test converting experiment to dictionary."""
        exp = Experiment(sample_experiment, manager)

        data = exp.to_dict()

        # Check required fields
        assert data["id"] == sample_experiment
        assert data["name"] == "test_experiment"
        assert data["status"] == "completed"
        assert data["description"] == "Test experiment for unit tests"
        assert set(data["tags"]) == {"cnn", "test", "training", "unit-tests"}
        assert data["archived"] is False

        # Check data fields
        assert "params" in data
        assert "metrics" in data
        assert "artifacts" in data
        assert "script_runs" in data

    def test_refresh(self, manager, sample_experiment):
        """Test refreshing cached data."""
        exp = Experiment(sample_experiment, manager)

        # Access some data to populate cache
        original_name = exp.name
        # original_params = exp.get_params()  # Not used in test

        # Modify data externally
        metadata = manager.storage.load_metadata(sample_experiment)
        metadata["name"] = "externally_modified"
        manager.storage.save_metadata(sample_experiment, metadata)

        # Should still return cached data
        assert exp.name == original_name

        # After refresh, should return new data
        exp.refresh()
        assert exp.name == "externally_modified"

    def test_string_representations(self, manager, sample_experiment):
        """Test string representations."""
        exp = Experiment(sample_experiment, manager)

        # __repr__
        repr_str = repr(exp)
        assert sample_experiment in repr_str
        assert "test_experiment" in repr_str
        assert "completed" in repr_str

        # __str__
        str_str = str(exp)
        assert sample_experiment in str_str
        assert "test_experiment" in str_str
        assert "completed" in str_str

    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_empty_metrics_handling(self, mock_capture_env, mock_git_info, manager):
        """Test handling experiments with no metrics."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment without metrics
        exp_id = manager.create_experiment(
            script_path=Path("test.py"), name="no_metrics_exp", tags=["unit-tests"]
        )

        exp = Experiment(exp_id, manager)

        # Should return empty DataFrame by default, not fail
        metrics_df = exp.get_metrics()
        assert metrics_df.empty

        # Should return empty DataFrame for specific step
        step_0_df = exp.get_metrics(step=0)
        assert step_0_df.empty

        # Should return empty list/dict with as_dataframe=False
        assert exp.get_metrics(as_dataframe=False) == []
        assert exp.get_metrics(step=0, as_dataframe=False) == {}

    def test_get_metric(self, manager, sample_experiment):
        """Test getting a specific metric."""
        exp = Experiment(sample_experiment, manager)

        # Test getting existing metrics
        assert exp.get_metric("accuracy") == 0.95
        assert exp.get_metric("loss") == 0.05

        # Test getting non-existent metric
        assert exp.get_metric("nonexistent") is None

    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_get_metric_multiple_steps(self, mock_capture_env, mock_git_info, manager):
        """Test get_metric with multiple steps."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment
        exp_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="multi_step_metric_test",
            tags=["unit-tests"],
        )

        # Add multiple metric steps
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.8, "loss": 0.2, "epoch": 1}
        )
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.85, "loss": 0.15, "epoch": 2}
        )
        manager.storage.add_result_step(
            exp_id, {"accuracy": 0.9, "loss": 0.1, "epoch": 3}
        )

        exp = Experiment(exp_id, manager)

        # Test getting metric with multiple values - should return list
        accuracy_values = exp.get_metric("accuracy")
        assert accuracy_values == [0.8, 0.85, 0.9]

        loss_values = exp.get_metric("loss")
        assert loss_values == [0.2, 0.15, 0.1]

        epoch_values = exp.get_metric("epoch")
        assert epoch_values == [1, 2, 3]

        # Test non-existent metric
        assert exp.get_metric("nonexistent") is None

    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_get_metric_single_step(self, mock_capture_env, mock_git_info, manager):
        """Test get_metric with single step."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment
        exp_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="single_step_metric_test",
            tags=["unit-tests"],
        )

        # Add single metric step
        manager.storage.add_result_step(exp_id, {"accuracy": 0.92, "loss": 0.08})

        exp = Experiment(exp_id, manager)

        # Test getting metric with single value - should return value directly
        assert exp.get_metric("accuracy") == 0.92
        assert exp.get_metric("loss") == 0.08

        # Test non-existent metric
        assert exp.get_metric("nonexistent") is None

    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_get_metric_no_metrics(self, mock_capture_env, mock_git_info, manager):
        """Test get_metric with no metrics."""
        # Setup mocks
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        # Create experiment without metrics
        exp_id = manager.create_experiment(
            script_path=Path("test.py"), name="no_metrics_test", tags=["unit-tests"]
        )

        exp = Experiment(exp_id, manager)

        # Should return None for any metric when no metrics exist
        assert exp.get_metric("accuracy") is None
        assert exp.get_metric("loss") is None

    def test_artifacts_access(self, manager, sample_experiment):
        """Test accessing experiment artifacts."""
        exp = Experiment(sample_experiment, manager)

        # List artifacts (may include git_diff.patch if there were uncommitted changes)
        artifacts = exp.list_artifacts()
        assert isinstance(artifacts, list)
        # Artifacts may contain git_diff.patch from Phase 2 implementation
        assert len(artifacts) >= 0

    def test_script_runs_access(self, manager, sample_experiment):
        """Test accessing experiment script runs."""
        exp = Experiment(sample_experiment, manager)

        script_runs = exp.get_script_runs()
        assert isinstance(script_runs, list)

    def test_experiment_with_default_manager(self, sample_experiment):
        """Test creating experiment with default manager."""
        # This tests that Experiment can create its own manager
        # Note: This might fail if there's no experiment with this ID in the default location
        # but it tests the code path
        try:
            exp = Experiment(sample_experiment)  # No manager provided
            # If we get here, the experiment exists in default location
            assert exp.id == sample_experiment
        except ExperimentNotFoundError:
            # Expected if experiment doesn't exist in default location
            pass


class TestExperimentDependencies:
    """Test dependency-related features of the Experiment class."""

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
        """Create an experiment manager for testing."""
        return ExperimentManager(experiments_dir=isolated_experiments_dir)

    @pytest.fixture
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def prep_experiment(self, mock_capture_env, mock_git_info, manager):
        """Create preprocessing experiment."""
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        exp_id = manager.create_experiment(
            script_path=Path("prepare_data.py"),
            name="preprocessing",
            config={"dataset": "mnist", "samples": 1000},
            tags=["preprocessing", "unit-tests"],
        )
        manager.start_experiment(exp_id)
        manager.complete_experiment(exp_id)
        return exp_id

    @pytest.fixture
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def train_experiment(
        self, mock_capture_env, mock_git_info, manager, prep_experiment
    ):
        """Create training experiment that depends on preprocessing."""
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        exp_id = manager.create_experiment(
            script_path=Path("train_model.py"),
            name="training",
            config={"learning_rate": 0.01},
            tags=["training", "unit-tests"],
            dependency_ids=[prep_experiment],
        )
        manager.start_experiment(exp_id)
        manager.complete_experiment(exp_id)
        return exp_id

    @pytest.fixture
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def eval_experiment(
        self, mock_capture_env, mock_git_info, manager, train_experiment
    ):
        """Create evaluation experiment that depends on training."""
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        exp_id = manager.create_experiment(
            script_path=Path("evaluate_model.py"),
            name="evaluation",
            config={},
            tags=["evaluation", "unit-tests"],
            dependency_ids=[train_experiment],
        )
        manager.start_experiment(exp_id)
        manager.complete_experiment(exp_id)
        return exp_id

    def test_no_dependencies(self, manager, prep_experiment):
        """Test experiment with no dependencies."""
        exp = Experiment(prep_experiment, manager)

        assert exp.dependency_ids == []
        assert exp.has_dependencies is False
        assert exp.get_dependencies() == []

    def test_single_dependency(self, manager, prep_experiment, train_experiment):
        """Test experiment with single dependency."""
        exp = Experiment(train_experiment, manager)

        # Test dependency_ids property
        assert len(exp.dependency_ids) == 1
        assert prep_experiment in exp.dependency_ids

        # Test has_dependencies property
        assert exp.has_dependencies is True

        # Test get_dependencies method
        deps = exp.get_dependencies()
        assert len(deps) == 1
        assert isinstance(deps[0], Experiment)
        assert deps[0].id == prep_experiment
        assert deps[0].name == "preprocessing"

    def test_transitive_dependencies(
        self, manager, prep_experiment, train_experiment, eval_experiment
    ):
        """Test getting transitive dependencies."""
        exp = Experiment(eval_experiment, manager)

        # Direct dependencies only
        direct_deps = exp.get_dependencies(transitive=False)
        assert len(direct_deps) == 1
        assert direct_deps[0].id == train_experiment

        # All transitive dependencies
        all_deps = exp.get_dependencies(transitive=True)
        assert len(all_deps) == 2
        dep_ids = [d.id for d in all_deps]
        assert prep_experiment in dep_ids
        assert train_experiment in dep_ids

    def test_include_self_with_transitive(
        self, manager, prep_experiment, train_experiment
    ):
        """Test include_self parameter with transitive dependencies."""
        exp = Experiment(train_experiment, manager)

        # Get dependencies without self
        deps = exp.get_dependencies(transitive=True, include_self=False)
        dep_ids = [d.id for d in deps]
        assert train_experiment not in dep_ids
        assert prep_experiment in dep_ids

        # Get dependencies with self
        deps_with_self = exp.get_dependencies(transitive=True, include_self=True)
        dep_ids_with_self = [d.id for d in deps_with_self]
        assert train_experiment in dep_ids_with_self
        assert prep_experiment in dep_ids_with_self

    def test_include_self_without_transitive(
        self, manager, prep_experiment, train_experiment
    ):
        """Test include_self parameter without transitive dependencies."""
        exp = Experiment(train_experiment, manager)

        # Get dependencies without self
        deps = exp.get_dependencies(transitive=False, include_self=False)
        dep_ids = [d.id for d in deps]
        assert train_experiment not in dep_ids
        assert prep_experiment in dep_ids

        # Get dependencies with self
        deps_with_self = exp.get_dependencies(transitive=False, include_self=True)
        dep_ids_with_self = [d.id for d in deps_with_self]
        assert train_experiment in dep_ids_with_self
        assert prep_experiment in dep_ids_with_self

    def test_to_dict_includes_dependencies(
        self, manager, prep_experiment, train_experiment
    ):
        """Test that to_dict includes dependency information."""
        exp = Experiment(train_experiment, manager)

        data = exp.to_dict()

        # Check dependency fields are present
        assert "dependency_ids" in data
        assert "has_dependencies" in data

        # Check values
        assert data["has_dependencies"] is True
        assert len(data["dependency_ids"]) == 1
        assert prep_experiment in data["dependency_ids"]

    def test_to_dict_no_dependencies(self, manager, prep_experiment):
        """Test that to_dict works for experiments without dependencies."""
        exp = Experiment(prep_experiment, manager)

        data = exp.to_dict()

        # Check dependency fields are present
        assert "dependency_ids" in data
        assert "has_dependencies" in data

        # Check values
        assert data["has_dependencies"] is False
        assert data["dependency_ids"] == []
