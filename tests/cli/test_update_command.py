"""
Tests for yanex CLI update command.
"""

from click.testing import CliRunner

from yanex.cli.main import cli


class TestUpdateCommand:
    """Test update command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_update_help(self):
        """Test update command help output."""
        result = self.runner.invoke(cli, ["update", "--help"])
        assert result.exit_code == 0
        assert (
            "Update experiment metadata including name, description, status, and tags"
            in result.output
        )
        assert "--set-name" in result.output
        assert "--set-description" in result.output
        assert "--set-status" in result.output
        assert "--add-tag" in result.output
        assert "--remove-tag" in result.output
        assert "--dry-run" in result.output

    def test_update_mutual_exclusivity_error(self):
        """Test that update command enforces mutual exclusivity between identifiers and filters."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--status", "completed", "--set-name", "test"]
        )
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    def test_update_no_arguments_error(self):
        """Test that update command requires either identifiers or filters."""
        result = self.runner.invoke(cli, ["update"])
        assert result.exit_code == 1
        assert "Must specify at least one update option" in result.output

    def test_update_no_update_options_error(self):
        """Test that update command requires at least one update option."""
        result = self.runner.invoke(cli, ["update", "exp1"])
        assert result.exit_code == 1
        assert "Must specify at least one update option" in result.output

    def test_update_nonexistent_experiment(self):
        """Test updating a non-existent experiment."""
        result = self.runner.invoke(
            cli, ["update", "nonexistent123", "--set-name", "test"]
        )
        assert result.exit_code == 1
        assert (
            "No regular experiment found with ID or name 'nonexistent123'"
            in result.output
        )

    def test_update_with_filters_no_matches(self):
        """Test update command with filters that match no experiments."""
        result = self.runner.invoke(
            cli,
            [
                "update",
                "--name",
                "definitely-nonexistent-*",
                "--set-name",
                "test",
                "--force",
            ],
        )
        assert result.exit_code == 0
        assert "No regular experiments found to update" in result.output

    def test_update_with_valid_identifiers_only(self):
        """Test update command accepts experiment identifiers without filters."""
        # This will fail since the experiment doesn't exist, but validates the CLI parsing
        result = self.runner.invoke(cli, ["update", "exp1", "--set-name", "test"])
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_update_with_valid_filters_only(self):
        """Test update command accepts filters without identifiers."""
        result = self.runner.invoke(
            cli,
            [
                "update",
                "--name",
                "definitely-nonexistent-*",
                "--set-name",
                "test",
                "--force",
            ],
        )
        assert result.exit_code == 0
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_update_invalid_status_value(self):
        """Test update command rejects invalid status values."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--set-status", "invalid-status"]
        )
        assert result.exit_code == 2  # Click validation error
        assert "Invalid value for '--set-status'" in result.output

    def test_update_dry_run_with_identifiers(self):
        """Test update command dry-run mode with identifiers."""
        result = self.runner.invoke(
            cli, ["update", "nonexistent123", "--set-name", "test", "--dry-run"]
        )
        assert result.exit_code == 1
        # Should try to find the experiment first before dry-run
        assert (
            "No regular experiment found with ID or name 'nonexistent123'"
            in result.output
        )

    def test_update_dry_run_with_filters(self):
        """Test update command dry-run mode with filters."""
        result = self.runner.invoke(
            cli,
            [
                "update",
                "--name",
                "definitely-nonexistent-*",
                "--set-name",
                "test",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        # Should show no experiments found since we don't have any in test environment
        assert "No regular experiments found to update" in result.output

    def test_update_archived_flag(self):
        """Test update command with archived flag."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--set-name", "test", "--archived"]
        )
        assert result.exit_code == 1
        # Should look for archived experiments
        assert "No archived experiment found with ID or name 'exp1'" in result.output

    def test_update_multiple_tag_operations(self):
        """Test update command with multiple tag add/remove operations."""
        result = self.runner.invoke(
            cli,
            [
                "update",
                "exp1",
                "--add-tag",
                "tag1",
                "--add-tag",
                "tag2",
                "--remove-tag",
                "old-tag",
            ],
        )
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiment found" in result.output

    def test_update_clear_fields(self):
        """Test update command field clearing with empty strings."""
        result = self.runner.invoke(
            cli, ["update", "exp1", "--set-name", "", "--set-description", ""]
        )
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
        assert "No regular experiment found" in result.output
