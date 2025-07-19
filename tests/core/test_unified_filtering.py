"""
Tests for the unified experiment filtering system.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from yanex.core.filtering import UnifiedExperimentFilter
from yanex.core.manager import ExperimentManager


class TestUnifiedExperimentFilter:
    """Test the UnifiedExperimentFilter class."""

    @pytest.fixture
    def filter_obj(self, isolated_experiments_dir, clean_git_repo):
        """Create a filter object with test manager."""
        manager = ExperimentManager(experiments_dir=isolated_experiments_dir)
        return UnifiedExperimentFilter(manager=manager)

    def teardown_method(self, method):
        """Clean up experiments after each test method."""
        try:
            # This will use the default manager location, but that's ok for cleanup
            from yanex.core.manager import ExperimentManager

            manager = ExperimentManager()

            # Find all test experiments
            from yanex.core.filtering import UnifiedExperimentFilter

            filter_obj = UnifiedExperimentFilter(manager=manager)
            test_experiments = filter_obj.filter_experiments(
                tags=["unit-tests"], limit=100
            )

            # Stop any running experiments and delete all test experiments
            for exp in test_experiments:
                try:
                    if exp.get("status") == "running":
                        manager.cancel_experiment(exp["id"], "Test cleanup")
                    manager.delete_experiment(exp["id"])
                except Exception:
                    pass  # Ignore errors during cleanup
        except Exception:
            pass  # Ignore all cleanup errors

    @pytest.fixture
    @patch("yanex.core.manager.validate_clean_working_directory")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def sample_experiments(
        self, mock_capture_env, mock_git_info, mock_validate_git, filter_obj
    ):
        """Create sample experiments for testing."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        manager = filter_obj.manager

        experiments = []

        # Create experiment 1: completed training
        exp1_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="training_run_1",
            config={"learning_rate": 0.001, "epochs": 100},
            tags=["training", "cnn", "unit-tests"],
            description="First training run",
        )
        manager.start_experiment(exp1_id)
        manager.complete_experiment(exp1_id)
        experiments.append(exp1_id)

        # Create experiment 2: failed training
        exp2_id = manager.create_experiment(
            script_path=Path("train.py"),
            name="training_run_2",
            config={"learning_rate": 0.01, "epochs": 50},
            tags=["training", "rnn", "unit-tests"],
            description="Second training run",
        )
        manager.start_experiment(exp2_id)
        manager.fail_experiment(exp2_id, "Out of memory")
        experiments.append(exp2_id)

        # Create experiment 3: running hyperparameter search
        exp3_id = manager.create_experiment(
            script_path=Path("hyperparam.py"),
            name="hyperparam_search_1",
            config={"search_space": "large"},
            tags=["hyperparameter", "training", "unit-tests"],
            description="Hyperparameter search",
        )
        manager.start_experiment(exp3_id)
        manager.complete_experiment(exp3_id)
        experiments.append(exp3_id)

        # Create experiment 4: completed evaluation (no training tag)
        exp4_id = manager.create_experiment(
            script_path=Path("eval.py"),
            name="evaluation_1",
            config={"model_path": "/models/best.pt"},
            tags=["evaluation", "unit-tests"],
            description="Model evaluation",
        )
        manager.start_experiment(exp4_id)
        manager.complete_experiment(exp4_id)
        experiments.append(exp4_id)

        # Create experiment 5: unnamed experiment
        exp5_id = manager.create_experiment(
            script_path=Path("test.py"),
            config={"test": True},
            tags=["test", "unit-tests"],
        )
        experiments.append(exp5_id)

        return experiments

    def test_filter_by_ids_or_logic(self, filter_obj, sample_experiments):
        """Test filtering by IDs with OR logic."""
        exp1, exp2, exp3, exp4, exp5 = sample_experiments

        # Test single ID
        result = filter_obj.filter_experiments(ids=[exp1])
        assert len(result) == 1
        assert result[0]["id"] == exp1

        # Test multiple IDs (OR logic)
        result = filter_obj.filter_experiments(ids=[exp1, exp3])
        assert len(result) == 2
        result_ids = {exp["id"] for exp in result}
        assert result_ids == {exp1, exp3}

        # Test non-existent ID
        result = filter_obj.filter_experiments(ids=["nonexistent"])
        assert len(result) == 0

    def test_filter_by_status_or_logic(self, filter_obj, sample_experiments):
        """Test filtering by status with OR logic."""

        # Test single status
        result = filter_obj.filter_experiments(status="completed")
        assert len(result) == 3  # exp1, exp3, and exp4
        assert all(exp["status"] == "completed" for exp in result)

        # Test multiple statuses (OR logic)
        result = filter_obj.filter_experiments(status=["completed", "failed"])
        assert len(result) == 4  # exp1, exp2, exp3, exp4
        statuses = {exp["status"] for exp in result}
        assert statuses == {"completed", "failed"}

        # Test string vs list input
        result1 = filter_obj.filter_experiments(status="created")
        result2 = filter_obj.filter_experiments(status=["created"])
        assert len(result1) == len(result2) == 1
        assert result1[0]["id"] == result2[0]["id"]

    def test_filter_by_tags_and_logic(self, filter_obj, sample_experiments):
        """Test filtering by tags with AND logic."""

        # Test single tag
        result = filter_obj.filter_experiments(tags=["training"])
        assert len(result) == 3  # exp1, exp2, exp3

        # Test multiple tags (AND logic - must have ALL)
        result = filter_obj.filter_experiments(tags=["training", "cnn"])
        assert len(result) == 1  # only exp1
        assert result[0]["id"] == sample_experiments[0]

        # Test tags that no experiment has both of
        result = filter_obj.filter_experiments(tags=["training", "evaluation"])
        assert len(result) == 0

    def test_filter_by_name_pattern(self, filter_obj, sample_experiments):
        """Test filtering by name pattern."""

        # Test exact match
        result = filter_obj.filter_experiments(name_pattern="training_run_1")
        assert len(result) == 1
        assert result[0]["name"] == "training_run_1"

        # Test glob pattern
        result = filter_obj.filter_experiments(name_pattern="training_*")
        assert len(result) == 2  # training_run_1 and training_run_2

        # Test wildcard in middle
        result = filter_obj.filter_experiments(name_pattern="*_1")
        assert len(result) == 3  # training_run_1, hyperparam_search_1, evaluation_1

        # Test empty pattern (should match unnamed experiments)
        result = filter_obj.filter_experiments(name_pattern="")
        assert len(result) == 1  # exp5 (unnamed)

    def test_combined_filters_and_logic(self, filter_obj, sample_experiments):
        """Test combining multiple filter types with AND logic."""
        exp1, exp2, exp3, exp4, exp5 = sample_experiments

        # Combine IDs and status (AND logic between filter types)
        result = filter_obj.filter_experiments(
            ids=[exp1, exp2, exp3], status="completed"
        )
        assert len(result) == 2  # exp1 and exp3 (in IDs AND completed)
        result_ids = {exp["id"] for exp in result}
        assert result_ids == {exp1, exp3}

        # Combine status and tags
        result = filter_obj.filter_experiments(
            status=["completed", "failed"], tags=["training"]
        )
        assert (
            len(result) == 3
        )  # exp1, exp2, and exp3 (training AND (completed OR failed))

        # Combine multiple filters
        result = filter_obj.filter_experiments(
            status="completed", tags=["training"], name_pattern="training_*"
        )
        assert len(result) == 1  # only exp1
        assert result[0]["id"] == exp1

    def test_archived_filtering(self, filter_obj, sample_experiments):
        """Test archived flag filtering."""
        exp1 = sample_experiments[0]
        manager = filter_obj.manager

        # Archive one experiment
        manager.storage.archive_experiment(exp1)

        # Test archived=False (default behavior)
        result = filter_obj.filter_experiments(archived=False)
        archived_ids = {exp["id"] for exp in result}
        assert exp1 not in archived_ids

        # Test archived=True
        result = filter_obj.filter_experiments(archived=True)
        assert len(result) == 1
        assert result[0]["id"] == exp1
        assert result[0]["archived"] is True

        # Test archived=None (both)
        result = filter_obj.filter_experiments(archived=None)
        all_ids = {exp["id"] for exp in result}
        assert exp1 in all_ids

    def test_limit_and_sorting(self, filter_obj, sample_experiments):
        """Test limit and sorting functionality."""

        # Test default limit (10)
        result = filter_obj.filter_experiments()
        assert len(result) <= 10

        # Test custom limit
        result = filter_obj.filter_experiments(limit=2)
        assert len(result) == 2

        # Test include_all overrides limit
        result = filter_obj.filter_experiments(limit=2, include_all=True)
        assert len(result) == 5  # All sample experiments

        # Test sorting by name (handle None names properly)
        result = filter_obj.filter_experiments(
            sort_by="name", sort_desc=False, include_all=True
        )
        names = [exp.get("name") or "" for exp in result]
        assert names == sorted(names)

    def test_time_filtering(self, filter_obj, sample_experiments):
        """Test time-based filtering."""

        # Get current time for relative comparisons
        now = datetime.now(timezone.utc).replace(
            microsecond=0
        )  # Use UTC timezone-aware time
        past = now - timedelta(hours=1)
        # future = now + timedelta(hours=1)  # Not used in tests

        # Test started_after
        result = filter_obj.filter_experiments(started_after=past)
        # Should include all started experiments (exp1, exp2, exp3, exp4)
        assert len(result) >= 4

        # Test started_before (should exclude recent experiments)
        result = filter_obj.filter_experiments(started_before=past)
        assert len(result) == 0  # No experiments started before an hour ago

        # Test string time parsing
        result = filter_obj.filter_experiments(started_after="1970-01-01")
        assert len(result) >= 4  # All started experiments

    def test_validation_errors(self, filter_obj):
        """Test input validation and error handling."""

        # Test invalid status
        with pytest.raises(ValueError, match="Invalid status"):
            filter_obj.filter_experiments(status="invalid_status")

        # Test invalid IDs type
        with pytest.raises(ValueError, match="ids must be a list"):
            filter_obj.filter_experiments(ids="not_a_list")

        # Test invalid tags type
        with pytest.raises(ValueError, match="tags must be a list"):
            filter_obj.filter_experiments(tags="not_a_list")

        # Test invalid archived type
        with pytest.raises(ValueError, match="archived must be a boolean"):
            filter_obj.filter_experiments(archived="not_a_bool")

    def test_count_and_existence_methods(self, filter_obj, sample_experiments):
        """Test utility methods for counting and checking existence."""
        exp1 = sample_experiments[0]

        # Test get_experiment_count
        count = filter_obj.get_experiment_count()
        assert count == 5

        count = filter_obj.get_experiment_count(status="completed")
        assert count == 3

        # Test experiment_exists
        assert filter_obj.experiment_exists(exp1) is True
        assert filter_obj.experiment_exists("nonexistent") is False

    def test_empty_filters(self, filter_obj, sample_experiments):
        """Test behavior with no filters (should return all experiments)."""
        result = filter_obj.filter_experiments(include_all=True)
        assert len(result) == 5

        # Test with default limit
        result = filter_obj.filter_experiments()
        assert len(result) == 5  # Still within default limit

    def test_edge_cases(self, filter_obj, sample_experiments):
        """Test edge cases and boundary conditions."""

        # Test empty lists
        result = filter_obj.filter_experiments(ids=[], include_all=True)
        assert len(result) == 5  # Empty list should be ignored

        result = filter_obj.filter_experiments(tags=[], include_all=True)
        assert len(result) == 5  # Empty list should be ignored

        # Test None values (should be ignored)
        result = filter_obj.filter_experiments(
            ids=None, status=None, tags=None, include_all=True
        )
        assert len(result) == 5

    def test_mixed_case_sensitivity(self, filter_obj, sample_experiments):
        """Test case sensitivity in various filters."""

        # Name pattern should be case insensitive
        result1 = filter_obj.filter_experiments(name_pattern="TRAINING_*")
        result2 = filter_obj.filter_experiments(name_pattern="training_*")
        assert len(result1) == len(result2) == 2

        # Status validation is case-sensitive and should raise error for invalid case
        with pytest.raises(ValueError, match="Invalid status"):
            filter_obj.filter_experiments(status="COMPLETED")
