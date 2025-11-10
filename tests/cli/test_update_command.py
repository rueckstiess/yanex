"""
Tests for yanex CLI update command.
"""

import pytest

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestUpdateCommand:
    """Test update command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_update_help_contains_required_sections(self):
        """Test update command help output contains all required sections."""
        result = self.runner.invoke(cli, ["update", "--help"])

        assert result.exit_code == 0

        # Check main description
        assert (
            "Update experiment metadata including name, description, status, and tags"
            in result.output
        )

        # Check all expected options are present
        expected_options = [
            "--set-name",
            "--set-description",
            "--set-status",
            "--add-tag",
            "--remove-tag",
            "--dry-run",
            "--archived",
        ]
        for option in expected_options:
            assert option in result.output

    @pytest.mark.parametrize(
        "command_args,expected_error",
        [
            # Mutual exclusivity violations
            (
                ["update", "exp1", "--status", "completed", "--set-name", "test"],
                "Cannot use both experiment identifiers and filter options",
            ),
            (
                ["update", "exp1", "--name", "test*", "--set-name", "new-name"],
                "Cannot use both experiment identifiers and filter options",
            ),
            # Missing arguments
            (["update"], "Must specify at least one update option"),
            (["update", "exp1"], "Must specify at least one update option"),
            (["update", "--name", "test*"], "Must specify at least one update option"),
        ],
    )
    def test_update_command_validation_errors(self, command_args, expected_error):
        """Test various command validation error scenarios."""
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 1
        assert expected_error in result.output

    @pytest.mark.parametrize(
        "invalid_status",
        [
            "invalid-status",
            "unknown",
            "COMPLETED",  # Wrong case
            "not-a-status",
        ],
    )
    def test_update_invalid_status_values(self, invalid_status):
        """Test update command rejects invalid status values."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--set-status", invalid_status]
        )

        assert result.exit_code == 2  # Click validation error
        assert "Invalid value for '--set-status'" in result.output

    @pytest.mark.parametrize(
        "experiment_id",
        [
            "nonexistent123",
            "missing-exp",
            "not-found",
            "fake12345",
        ],
    )
    def test_update_nonexistent_experiments(self, experiment_id):
        """Test updating non-existent experiments."""
        result = self.runner.invoke(
            cli, ["update", experiment_id, "--set-name", "test"]
        )

        assert result.exit_code == 1
        assert (
            f"No regular experiment found with ID or name '{experiment_id}'"
            in result.output
        )

    @pytest.mark.parametrize(
        "filter_args",
        [
            ["--name", "definitely-nonexistent-*"],
        ],
    )
    def test_update_with_filters_no_matches(self, filter_args):
        """Test update command with filters that match no experiments."""
        command_args = ["update"] + filter_args + ["--set-name", "test", "--force"]
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 0
        assert "No regular experiments found to update" in result.output

    @pytest.mark.parametrize(
        "identifier_args,update_args",
        [
            # Basic identifier with various update operations
            (["exp1"], ["--set-name", "test"]),
            (["exp123"], ["--set-description", "new desc"]),
            (["sample-exp"], ["--set-status", "completed"]),
            (["test-exp"], ["--add-tag", "new-tag"]),
            (["my-exp"], ["--remove-tag", "old-tag"]),
            # Multiple identifiers
            (["exp1", "exp2"], ["--set-name", "batch-update"]),
            (["exp1", "exp2", "exp3"], ["--add-tag", "multi-tag"]),
        ],
    )
    def test_update_with_valid_identifiers_only(self, identifier_args, update_args):
        """Test update command accepts experiment identifiers without filters."""
        command_args = ["update"] + identifier_args + update_args
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiment found" in result.output

    @pytest.mark.parametrize(
        "filter_args,update_args",
        [
            # Name pattern filter that shouldn't match anything
            (["--name", "definitely-nonexistent-test-*"], ["--set-name", "updated"]),
            (
                ["--name", "nonexistent-*experiment*"],
                ["--set-description", "batch updated"],
            ),
        ],
    )
    def test_update_with_valid_filters_only(self, filter_args, update_args):
        """Test update command accepts filters without identifiers."""
        command_args = ["update"] + filter_args + update_args + ["--force"]
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 0
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiments found to update" in result.output

    @pytest.mark.parametrize(
        "dry_run_args,expected_behavior",
        [
            # Dry run with filters
            (
                [
                    "update",
                    "--name",
                    "definitely-nonexistent-unique-pattern-*",
                    "--set-name",
                    "test",
                    "--dry-run",
                ],
                "should show no experiments found",
            ),
        ],
    )
    def test_update_dry_run_behavior(self, dry_run_args, expected_behavior):
        """Test update command dry-run mode behavior."""
        result = self.runner.invoke(cli, dry_run_args)

        assert result.exit_code == 0
        assert "No regular experiments found to update" in result.output

    def test_update_archived_experiments_search(self):
        """Test update command with archived flag searches archived experiments."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--set-name", "test", "--archived"]
        )

        assert result.exit_code == 1
        # Should look for archived experiments instead of regular ones
        assert "No archived experiment found with ID or name 'exp1'" in result.output

    @pytest.mark.parametrize(
        "tag_operations",
        [
            # Single operations
            ["--add-tag", "tag1"],
            ["--remove-tag", "old-tag"],
            # Multiple add operations
            ["--add-tag", "tag1", "--add-tag", "tag2"],
            ["--add-tag", "tag1", "--add-tag", "tag2", "--add-tag", "tag3"],
            # Multiple remove operations
            ["--remove-tag", "old1", "--remove-tag", "old2"],
            # Mixed operations
            ["--add-tag", "new1", "--remove-tag", "old1"],
            ["--add-tag", "tag1", "--add-tag", "tag2", "--remove-tag", "old-tag"],
        ],
    )
    def test_update_multiple_tag_operations(self, tag_operations):
        """Test update command with various tag add/remove operations."""
        command_args = ["update", "exp1"] + tag_operations
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiment found" in result.output

    @pytest.mark.parametrize(
        "field_clearing_args",
        [
            # Clear individual fields
            ["--set-name", ""],
            ["--set-description", ""],
            # Clear multiple fields
            ["--set-name", "", "--set-description", ""],
            ["--set-name", "", "--set-description", "", "--add-tag", "cleared"],
            # Mix clearing with setting
            ["--set-name", "", "--set-description", "new description"],
            ["--set-name", "new-name", "--set-description", ""],
        ],
    )
    def test_update_field_clearing_operations(self, field_clearing_args):
        """Test update command field clearing with empty strings."""
        command_args = ["update", "exp1"] + field_clearing_args
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiment found" in result.output

    def test_update_comprehensive_operation_combination(self):
        """Test update command with comprehensive combination of operations."""
        command_args = [
            "update",
            "test-exp",
            "--set-name",
            "comprehensive-test",
            "--set-description",
            "Updated with comprehensive test",
            "--set-status",
            "completed",
            "--add-tag",
            "tested",
            "--add-tag",
            "verified",
            "--remove-tag",
            "draft",
            "--dry-run",
        ]

        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 1
        # Should try to find the experiment first before dry-run processing
        assert "No regular experiment found with ID or name 'test-exp'" in result.output

    @pytest.mark.parametrize(
        "complex_filter_combination",
        [
            # Multiple filter types
            ["--status", "running", "--name", "test-*"],
            ["--status", "completed", "--name", "prod-*"],
        ],
    )
    def test_update_with_complex_filter_combinations(self, complex_filter_combination):
        """Test update command with complex filter combinations."""
        command_args = (
            ["update"]
            + complex_filter_combination
            + ["--set-name", "filtered-update", "--force"]
        )
        result = self.runner.invoke(cli, command_args)

        assert result.exit_code == 0
        assert "No regular experiments found to update" in result.output
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_update_command_force_flag_behavior(self):
        """Test that force flag prevents confirmation prompts in batch operations."""
        # Test force flag with filters (should proceed without confirmation)
        result = self.runner.invoke(
            cli, ["update", "--name", "nonexistent-*", "--set-name", "test", "--force"]
        )

        assert result.exit_code == 0
        assert "No regular experiments found to update" in result.output
        # Should not prompt for confirmation due to --force flag


class TestUpdateCommandIntegration:
    """Integration tests for update command with real experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_update_experiment_tags_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test updating experiment tags with real experiment."""
        # Create an experiment
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "test-exp",
                "--tag",
                "initial",
            ],
        )
        assert result.exit_code == 0

        # Extract experiment ID from output (format: "✓ Experiment completed successfully: exp_id")
        exp_id = None
        for line in result.output.split("\n"):
            if (
                "Experiment completed successfully:" in line
                or "Experiment staged:" in line
            ):
                exp_id = line.split(":")[-1].strip()
                break

        assert exp_id is not None, (
            f"Could not find experiment ID in output:\n{result.output}"
        )

        # Update the experiment to add a tag
        result = self.runner.invoke(
            cli, ["update", exp_id, "--add-tag", "updated", "--force"]
        )

        assert result.exit_code == 0
        assert "Successfully updated 1 experiment" in result.output

        # Verify the tag was added
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        assert "updated" in result.output
        assert "initial" in result.output

    def test_update_experiment_name_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test updating experiment name with real experiment."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "old-name"]
        )
        assert result.exit_code == 0

        # Extract experiment ID (format: "✓ Experiment completed successfully: exp_id")
        exp_id = None
        for line in result.output.split("\n"):
            if (
                "Experiment completed successfully:" in line
                or "Experiment staged:" in line
            ):
                exp_id = line.split(":")[-1].strip()
                break

        assert exp_id is not None, (
            f"Could not find experiment ID in output:\n{result.output}"
        )

        # Update the name
        result = self.runner.invoke(
            cli, ["update", exp_id, "--set-name", "new-name", "--force"]
        )

        assert result.exit_code == 0
        assert "Successfully updated 1 experiment" in result.output

        # Verify the name was changed
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0
        assert "new-name" in result.output
