"""Integration tests for parameter conflict detection in yanex API."""

from pathlib import Path

import pytest

import yanex
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import ParameterConflictError


class TestGetParamConflictDetection:
    """Test get_param with conflict detection."""

    @pytest.fixture(autouse=True)
    def setup(self, per_test_experiments_dir, git_repo):
        """Set up test fixtures."""
        self.experiments_dir = per_test_experiments_dir
        self.git_repo = git_repo
        self.manager = ExperimentManager(per_test_experiments_dir)

        # Create a simple script
        self.script_path = Path(git_repo.working_dir) / "train.py"
        self.script_path.write_text("print('training')")

    def test_no_conflict_without_dependencies(self):
        """Test get_param works normally without dependencies."""
        # Create experiment with config
        exp_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01, "batch_size": 32},
        )

        # Simulate being inside the experiment
        yanex._set_current_experiment_id(exp_id)
        try:
            # Should not raise - no dependencies
            params = yanex.get_params()
            assert params["lr"] == 0.01
            assert params["batch_size"] == 32
        finally:
            yanex._clear_current_experiment_id()
            # Reset tracked params
            yanex.api._tracked_params = None

    def test_conflict_detected_between_dep_and_config(self):
        """Test conflict is detected between dependency and current config."""
        # Create first experiment with one lr value
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
        )
        self.manager.complete_experiment(exp1_id)

        # Create second experiment depending on first, but with different lr
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},  # Different from exp1
            dependencies={"data": exp1_id},
        )

        # Simulate being inside exp2
        yanex._set_current_experiment_id(exp2_id)
        try:
            params = yanex.get_params()
            with pytest.raises(ParameterConflictError) as exc_info:
                _ = params["lr"]

            # Error message should be helpful
            error_str = str(exc_info.value)
            assert "lr" in error_str
            assert "config" in error_str
            assert "data" in error_str or exp1_id[:8] in error_str
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_no_conflict_when_values_match(self):
        """Test no conflict when dependency and config have same value."""
        # Create first experiment
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
        )
        self.manager.complete_experiment(exp1_id)

        # Create second experiment with same lr
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},  # Same as exp1
            dependencies={"data": exp1_id},
        )

        yanex._set_current_experiment_id(exp2_id)
        try:
            params = yanex.get_params()
            # Should not raise - values match
            assert params["lr"] == 0.01
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_from_dependency_escape_hatch(self):
        """Test from_dependency bypasses conflict detection."""
        # Create first experiment
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
        )
        self.manager.complete_experiment(exp1_id)

        # Create second experiment with different lr
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
            dependencies={"model": exp1_id},
        )

        yanex._set_current_experiment_id(exp2_id)
        try:
            # Normal access would raise conflict
            params = yanex.get_params()
            with pytest.raises(ParameterConflictError):
                _ = params["lr"]

            # Reset for next test
            yanex.api._tracked_params = None

            # Using from_dependency should work
            lr = yanex.get_param("lr", from_dependency="model")
            assert lr == 0.001  # Gets dependency's value
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_ignore_dependencies_escape_hatch(self):
        """Test ignore_dependencies bypasses conflict detection."""
        # Create first experiment
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
        )
        self.manager.complete_experiment(exp1_id)

        # Create second experiment with different lr
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
            dependencies={"model": exp1_id},
        )

        yanex._set_current_experiment_id(exp2_id)
        try:
            # Using ignore_dependencies should get local config value
            lr = yanex.get_param("lr", ignore_dependencies=True)
            assert lr == 0.01  # Gets local config value
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_both_escape_hatches_raises_error(self):
        """Test specifying both escape hatches raises ValueError."""
        exp_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
        )

        yanex._set_current_experiment_id(exp_id)
        try:
            with pytest.raises(ValueError) as exc_info:
                yanex.get_param("lr", from_dependency="model", ignore_dependencies=True)

            assert "both" in str(exc_info.value).lower()
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_nested_param_conflict(self):
        """Test conflict detection for nested parameters."""
        # Create first experiment with nested config
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"model": {"lr": 0.001, "layers": 3}},
        )
        self.manager.complete_experiment(exp1_id)

        # Create second experiment with different nested value
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"model": {"lr": 0.01, "layers": 3}},
            dependencies={"base": exp1_id},
        )

        yanex._set_current_experiment_id(exp2_id)
        try:
            params = yanex.get_params()

            # layers matches - no conflict
            assert params["model"]["layers"] == 3

            # lr differs - should conflict
            with pytest.raises(ParameterConflictError) as exc_info:
                _ = params["model"]["lr"]

            assert "model.lr" in str(exc_info.value)
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None


class TestTransitiveDependencyConflict:
    """Test conflict detection with transitive dependencies."""

    @pytest.fixture(autouse=True)
    def setup(self, per_test_experiments_dir, git_repo):
        """Set up test fixtures."""
        self.experiments_dir = per_test_experiments_dir
        self.git_repo = git_repo
        self.manager = ExperimentManager(per_test_experiments_dir)

        self.script_path = Path(git_repo.working_dir) / "train.py"
        self.script_path.write_text("print('training')")

    def test_transitive_conflict_detected(self):
        """Test conflict with transitive dependency is detected."""
        # Chain: exp3 -> exp2 -> exp1
        # exp1 has lr=0.001
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
        )
        self.manager.complete_experiment(exp1_id)

        # exp2 has lr=0.001 (same as exp1, no conflict yet)
        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
            dependencies={"data": exp1_id},
        )
        self.manager.complete_experiment(exp2_id)

        # exp3 has lr=0.01 (different!) - should conflict with both exp1 and exp2
        exp3_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
            dependencies={"model": exp2_id},
        )

        yanex._set_current_experiment_id(exp3_id)
        try:
            params = yanex.get_params()
            with pytest.raises(ParameterConflictError):
                _ = params["lr"]
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_transitive_no_conflict_all_agree(self):
        """Test no conflict when all experiments in chain agree."""
        # All have lr=0.01
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
        )
        self.manager.complete_experiment(exp1_id)

        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
            dependencies={"data": exp1_id},
        )
        self.manager.complete_experiment(exp2_id)

        exp3_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
            dependencies={"model": exp2_id},
        )

        yanex._set_current_experiment_id(exp3_id)
        try:
            params = yanex.get_params()
            # Should not raise - all agree
            assert params["lr"] == 0.01
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None


class TestFromDependencySlot:
    """Test from_dependency with various slot names."""

    @pytest.fixture(autouse=True)
    def setup(self, per_test_experiments_dir, git_repo):
        """Set up test fixtures."""
        self.experiments_dir = per_test_experiments_dir
        self.git_repo = git_repo
        self.manager = ExperimentManager(per_test_experiments_dir)

        self.script_path = Path(git_repo.working_dir) / "train.py"
        self.script_path.write_text("print('training')")

    def test_from_dependency_invalid_slot(self):
        """Test from_dependency with non-existent slot returns default."""
        exp_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
        )

        yanex._set_current_experiment_id(exp_id)
        try:
            # No dependencies, so any slot is invalid
            result = yanex.get_param("lr", default=0.999, from_dependency="nonexistent")
            assert result == 0.999
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None

    def test_from_dependency_correct_slot(self):
        """Test from_dependency gets correct dependency by slot."""
        # Create two dependencies with different values
        exp1_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.001},
        )
        self.manager.complete_experiment(exp1_id)

        exp2_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.01},
        )
        self.manager.complete_experiment(exp2_id)

        # Create experiment depending on both
        exp3_id = self.manager.create_experiment(
            script_path=self.script_path,
            config={"lr": 0.1},
            dependencies={"model_a": exp1_id, "model_b": exp2_id},
        )

        yanex._set_current_experiment_id(exp3_id)
        try:
            # Get from specific slots
            lr_a = yanex.get_param("lr", from_dependency="model_a")
            lr_b = yanex.get_param("lr", from_dependency="model_b")

            assert lr_a == 0.001
            assert lr_b == 0.01
        finally:
            yanex._clear_current_experiment_id()
            yanex.api._tracked_params = None
