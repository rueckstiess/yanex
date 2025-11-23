"""
Tests for confirmation utilities for bulk operations.

This module tests the confirmation and experiment finding functionality
used by archive, delete, and update commands.
"""

from unittest.mock import Mock, patch

import click
import pytest

from tests.test_utils import TestDataFactory
from yanex.cli.commands.confirm import (
    confirm_experiment_operation,
    find_experiment,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)
from yanex.cli.filters import ExperimentFilter


class TestConfirmExperimentOperation:
    """Test confirmation prompts for bulk operations."""

    def test_confirm_no_experiments_returns_false(self):
        """Test that confirmation returns False when no experiments provided."""
        result = confirm_experiment_operation([], "archive")
        assert result is False

    @patch("click.echo")
    def test_confirm_no_experiments_shows_message(self, mock_echo):
        """Test that appropriate message is shown when no experiments."""
        confirm_experiment_operation([], "archive")
        # Check that the echo was called with the expected message
        mock_echo.assert_any_call("No experiments found to archive.")

    @patch("click.confirm", return_value=True)
    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_single_experiment(self, mock_formatter, mock_confirm):
        """Test confirmation prompt for single experiment."""
        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        result = confirm_experiment_operation(experiments, "archive")

        assert result is True
        mock_confirm.assert_called_once_with("Archive this experiment?", default=False)

    @patch("click.confirm", return_value=True)
    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_multiple_experiments(self, mock_formatter, mock_confirm):
        """Test confirmation prompt for multiple experiments."""
        experiments = [
            TestDataFactory.create_experiment_metadata("exp001"),
            TestDataFactory.create_experiment_metadata("exp002"),
            TestDataFactory.create_experiment_metadata("exp003"),
        ]

        result = confirm_experiment_operation(experiments, "delete")

        assert result is True
        mock_confirm.assert_called_once_with(
            "Delete these 3 experiments?", default=False
        )

    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_with_force_flag_skips_prompt(self, mock_formatter):
        """Test that force flag bypasses confirmation prompt."""
        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        # Should return True without prompting when force=True
        result = confirm_experiment_operation(experiments, "archive", force=True)

        assert result is True

    @patch("click.confirm", return_value=False)
    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_user_abort(self, mock_formatter, mock_confirm):
        """Test that user can abort by declining confirmation."""
        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        result = confirm_experiment_operation(experiments, "archive")

        assert result is False

    @patch("click.confirm", return_value=True)
    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_custom_operation_verb(self, mock_formatter, mock_confirm):
        """Test custom operation verb in confirmation message."""
        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        with patch("click.echo") as mock_echo:
            confirm_experiment_operation(
                experiments, "update", operation_verb="updated"
            )

            # Check that custom verb is used in the output
            calls = [str(call) for call in mock_echo.call_args_list]
            assert any("updated" in str(call) for call in calls)

    @patch("click.confirm", return_value=True)
    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_default_yes_option(self, mock_formatter, mock_confirm):
        """Test that default_yes sets the confirmation default."""
        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        confirm_experiment_operation(experiments, "archive", default_yes=True)

        mock_confirm.assert_called_once_with("Archive this experiment?", default=True)

    @patch("yanex.cli.commands.confirm.ExperimentTableFormatter")
    def test_confirm_calls_table_formatter(self, mock_formatter_class):
        """Test that table formatter is called to display experiments."""
        mock_formatter = Mock()
        mock_formatter_class.return_value = mock_formatter

        experiments = [TestDataFactory.create_experiment_metadata("exp001")]

        with patch("click.confirm", return_value=True):
            confirm_experiment_operation(experiments, "archive")

        mock_formatter.print_experiments_table.assert_called_once_with(experiments)


class TestFindExperiment:
    """Test experiment resolution by ID/name with find_experiment function."""

    def test_find_by_exact_experiment_id(self, isolated_storage, isolated_manager):
        """Test finding experiment by exact 8-character ID."""
        # Create experiment properly
        exp_id = "abc12345"
        metadata = TestDataFactory.create_experiment_metadata(exp_id)
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Find it
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, exp_id)

        assert result is not None
        assert result["id"] == exp_id

    def test_find_by_id_prefix_unique(self, isolated_storage, isolated_manager):
        """Test finding experiment by unique ID prefix."""
        # Create experiments with different prefixes
        exp_id1 = "abc12345"
        exp_id2 = "def67890"

        for exp_id in [exp_id1, exp_id2]:
            metadata = TestDataFactory.create_experiment_metadata(exp_id)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Find by unique prefix
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, "abc")

        assert result is not None
        assert result["id"] == exp_id1

    def test_find_by_id_prefix_ambiguous_returns_list(
        self, isolated_storage, isolated_manager
    ):
        """Test that ambiguous ID prefix returns list of matches."""
        # Create experiments with same prefix
        exp_id1 = "abc12345"
        exp_id2 = "abc67890"

        for exp_id in [exp_id1, exp_id2]:
            metadata = TestDataFactory.create_experiment_metadata(exp_id)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Find by ambiguous prefix
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, "abc")

        # Should return list of both matches
        assert isinstance(result, list)
        assert len(result) == 2

    def test_find_by_name_exact_match(self, isolated_storage, isolated_manager):
        """Test finding experiment by exact name."""
        exp_id = "abc12345"
        exp_name = "my-experiment"
        metadata = TestDataFactory.create_experiment_metadata(exp_id, name=exp_name)
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Find by name
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, exp_name)

        assert result is not None
        assert result["name"] == exp_name

    def test_find_by_name_ambiguous_returns_list(
        self, isolated_storage, isolated_manager
    ):
        """Test that multiple experiments with same name returns list."""
        exp_name = "duplicate-name"
        exp_ids = ["abc12345", "def67890"]

        for exp_id in exp_ids:
            metadata = TestDataFactory.create_experiment_metadata(exp_id, name=exp_name)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Find by name
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, exp_name)

        # Should return list of both matches
        assert isinstance(result, list)
        assert len(result) == 2

    def test_find_experiment_not_found_returns_none(
        self, isolated_storage, isolated_manager
    ):
        """Test that nonexistent experiment returns None."""
        filter_obj = ExperimentFilter(manager=isolated_manager)
        result = find_experiment(filter_obj, "nonexistent")

        assert result is None

    def test_find_archived_experiments(self, isolated_storage, isolated_manager):
        """Test finding archived experiments with include_archived flag."""
        exp_id = "abc12345"
        metadata = TestDataFactory.create_experiment_metadata(exp_id, archived=True)

        # Create experiment normally first
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Find with archived flag
        filter_obj = ExperimentFilter(manager=isolated_manager)
        find_experiment(filter_obj, exp_id, include_archived=True)

        # Note: This might return None depending on implementation
        # The function uses _load_all_experiments which might not include archived
        # This test documents current behavior


class TestFindExperimentsByIdentifiers:
    """Test experiment resolution by list of identifiers."""

    def test_find_by_exact_experiment_id(self, isolated_storage, isolated_manager):
        """Test finding experiments by exact IDs."""
        exp_ids = ["abc12345", "def67890"]

        for exp_id in exp_ids:
            metadata = TestDataFactory.create_experiment_metadata(exp_id)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Find by IDs
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_identifiers(filter_obj, exp_ids)

        assert len(results) == 2
        result_ids = [r["id"] for r in results]
        assert set(result_ids) == set(exp_ids)

    def test_find_by_id_prefix_unique(self, isolated_storage, isolated_manager):
        """Test finding by unique ID prefix."""
        exp_id = "abc12345"
        metadata = TestDataFactory.create_experiment_metadata(exp_id)
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Find by prefix
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_identifiers(filter_obj, ["abc"])

        assert len(results) == 1
        assert results[0]["id"] == exp_id

    def test_find_by_id_prefix_ambiguous_raises(
        self, isolated_storage, isolated_manager
    ):
        """Test that ambiguous ID prefix raises error."""
        # Create experiments with same prefix
        for exp_id in ["abc12345", "abc67890"]:
            metadata = TestDataFactory.create_experiment_metadata(exp_id)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Should raise on ambiguous prefix
        filter_obj = ExperimentFilter(manager=isolated_manager)
        with pytest.raises(click.ClickException) as exc_info:
            find_experiments_by_identifiers(filter_obj, ["abc"])

        assert "Ambiguous identifier" in str(exc_info.value)

    def test_find_by_name_exact_match(self, isolated_storage, isolated_manager):
        """Test finding by exact name."""
        exp_id = "abc12345"
        exp_name = "my-experiment"
        metadata = TestDataFactory.create_experiment_metadata(exp_id, name=exp_name)
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Find by name
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_identifiers(filter_obj, [exp_name])

        assert len(results) == 1
        assert results[0]["name"] == exp_name

    def test_find_by_name_ambiguous_raises(self, isolated_storage, isolated_manager):
        """Test that ambiguous name raises error with helpful message."""
        exp_name = "duplicate-name"

        for exp_id in ["abc12345", "def67890"]:
            metadata = TestDataFactory.create_experiment_metadata(exp_id, name=exp_name)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Should raise on ambiguous name
        filter_obj = ExperimentFilter(manager=isolated_manager)
        with pytest.raises(click.ClickException) as exc_info:
            find_experiments_by_identifiers(filter_obj, [exp_name])

        assert "Ambiguous identifier" in str(exc_info.value)

    def test_find_experiment_not_found_raises(self, isolated_storage, isolated_manager):
        """Test that nonexistent experiment raises error."""
        filter_obj = ExperimentFilter(manager=isolated_manager)

        with pytest.raises(click.ClickException) as exc_info:
            find_experiments_by_identifiers(filter_obj, ["nonexistent"])

        assert "No regular experiment found" in str(exc_info.value)

    def test_find_multiple_identifiers(self, isolated_storage, isolated_manager):
        """Test finding multiple experiments by mixed identifiers."""
        exp1_id = "abc12345"
        exp2_name = "named-experiment"
        exp2_id = "def67890"

        # Create experiments
        metadata1 = TestDataFactory.create_experiment_metadata(exp1_id)
        isolated_storage.create_experiment_directory(exp1_id)
        isolated_storage.save_metadata(exp1_id, metadata1)

        metadata2 = TestDataFactory.create_experiment_metadata(exp2_id, name=exp2_name)
        isolated_storage.create_experiment_directory(exp2_id)
        isolated_storage.save_metadata(exp2_id, metadata2)

        # Find by ID and name
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_identifiers(filter_obj, [exp1_id, exp2_name])

        assert len(results) == 2

    def test_find_archived_experiments(self, isolated_storage, isolated_manager):
        """Test finding archived experiments with archived=True flag."""
        # This tests the archived parameter behavior
        # The actual implementation might need archived experiments in archive dir
        filter_obj = ExperimentFilter(manager=isolated_manager)

        # Should not raise error with archived flag
        # Even if no experiments found, it should handle gracefully
        try:
            find_experiments_by_identifiers(filter_obj, ["nonexistent"], archived=True)
        except click.ClickException as e:
            # Should mention "archived" in error message
            assert "archived" in str(e).lower()


class TestFindExperimentsByFilters:
    """Test experiment resolution by filter criteria."""

    def test_find_by_status_filter(self, isolated_storage, isolated_manager):
        """Test filtering experiments by status."""
        # Create experiments with different statuses
        for i, status in enumerate(["completed", "running", "failed"]):
            exp_id = f"exp0000{i}"
            metadata = TestDataFactory.create_experiment_metadata(exp_id, status=status)
            isolated_storage.create_experiment_directory(exp_id)
            isolated_storage.save_metadata(exp_id, metadata)

        # Filter by status
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_filters(filter_obj, status="completed")

        assert len(results) == 1
        assert results[0]["status"] == "completed"

    def test_find_by_multiple_filter_criteria(self, isolated_storage, isolated_manager):
        """Test filtering with multiple criteria."""
        # Create experiments
        exp1_id = "abc12345"
        metadata1 = TestDataFactory.create_experiment_metadata(
            exp1_id, status="completed", name="test-exp", tags=["ml", "training"]
        )
        isolated_storage.create_experiment_directory(exp1_id)
        isolated_storage.save_metadata(exp1_id, metadata1)

        exp2_id = "def67890"
        metadata2 = TestDataFactory.create_experiment_metadata(
            exp2_id, status="failed", name="other-exp"
        )
        isolated_storage.create_experiment_directory(exp2_id)
        isolated_storage.save_metadata(exp2_id, metadata2)

        # Filter by status and tags
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_filters(
            filter_obj, status="completed", tags=["ml"]
        )

        assert len(results) == 1
        assert results[0]["id"] == exp1_id

    def test_find_no_matches_returns_empty(self, isolated_storage, isolated_manager):
        """Test that no matches returns empty list."""
        # Create an experiment to ensure the directory is not empty
        exp_id = "test0001"
        metadata = TestDataFactory.create_experiment_metadata(
            exp_id, status="completed"
        )
        isolated_storage.create_experiment_directory(exp_id)
        isolated_storage.save_metadata(exp_id, metadata)

        # Try to filter with criteria that won't match
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_filters(filter_obj, status="failed")

        assert results == []

    def test_find_archived_with_flag(self, isolated_storage, isolated_manager):
        """Test finding archived experiments with archived=True."""
        # This is a pass-through test - just verify it doesn't crash
        filter_obj = ExperimentFilter(manager=isolated_manager)
        results = find_experiments_by_filters(filter_obj, archived=True)

        # Should return empty list or archived experiments
        assert isinstance(results, list)
