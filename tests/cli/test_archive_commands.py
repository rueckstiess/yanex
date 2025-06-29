"""
Tests for yanex CLI archive, delete, and unarchive commands.
"""

import tempfile
from pathlib import Path
from click.testing import CliRunner

import pytest

from yanex.cli.main import cli
from yanex.core.manager import ExperimentManager


class TestArchiveCommands:
    """Test archive, delete, and unarchive command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_archive_help(self):
        """Test archive command help output."""
        result = self.runner.invoke(cli, ["archive", "--help"])
        assert result.exit_code == 0
        assert (
            "Archive experiments by moving them to archived directory" in result.output
        )
        assert "--status" in result.output
        assert "--force" in result.output

    def test_delete_help(self):
        """Test delete command help output."""
        result = self.runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "Permanently delete experiments" in result.output
        assert "WARNING: This operation cannot be undone" in result.output
        assert "--archived" in result.output

    def test_unarchive_help(self):
        """Test unarchive command help output."""
        result = self.runner.invoke(cli, ["unarchive", "--help"])
        assert result.exit_code == 0
        assert (
            "Unarchive experiments by moving them back to experiments directory"
            in result.output
        )
        assert "--status" in result.output

    def test_archive_mutual_exclusivity_error(self):
        """Test that archive command enforces mutual exclusivity between identifiers and filters."""
        result = self.runner.invoke(cli, ["archive", "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    def test_delete_mutual_exclusivity_error(self):
        """Test that delete command enforces mutual exclusivity between identifiers and filters."""
        result = self.runner.invoke(cli, ["delete", "exp1", "--status", "failed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    def test_unarchive_mutual_exclusivity_error(self):
        """Test that unarchive command enforces mutual exclusivity between identifiers and filters."""
        result = self.runner.invoke(cli, ["unarchive", "exp1", "--status", "completed"])
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    def test_archive_no_arguments_error(self):
        """Test that archive command requires either identifiers or filters."""
        result = self.runner.invoke(cli, ["archive"])
        assert result.exit_code == 1
        assert (
            "Must specify either experiment identifiers or filter options"
            in result.output
        )

    def test_delete_no_arguments_error(self):
        """Test that delete command requires either identifiers or filters."""
        result = self.runner.invoke(cli, ["delete"])
        assert result.exit_code == 1
        assert (
            "Must specify either experiment identifiers or filter options"
            in result.output
        )

    def test_unarchive_no_arguments_error(self):
        """Test that unarchive command requires either identifiers or filters."""
        result = self.runner.invoke(cli, ["unarchive"])
        assert result.exit_code == 1
        assert (
            "Must specify either experiment identifiers or filter options"
            in result.output
        )

    def test_archive_nonexistent_experiment(self):
        """Test archiving a non-existent experiment."""
        result = self.runner.invoke(cli, ["archive", "nonexistent123"])
        assert result.exit_code == 1
        assert (
            "No regular experiment found with ID or name 'nonexistent123'"
            in result.output
        )

    def test_delete_nonexistent_experiment(self):
        """Test deleting a non-existent experiment."""
        result = self.runner.invoke(cli, ["delete", "nonexistent123"])
        assert result.exit_code == 1
        assert (
            "No regular experiment found with ID or name 'nonexistent123'"
            in result.output
        )

    def test_unarchive_nonexistent_experiment(self):
        """Test unarchiving a non-existent experiment."""
        result = self.runner.invoke(cli, ["unarchive", "nonexistent123"])
        assert result.exit_code == 1
        assert (
            "No archived experiment found with ID or name 'nonexistent123'"
            in result.output
        )

    def test_archive_with_filters_no_matches(self):
        """Test archive command with filters that match no experiments."""
        # Use a name pattern that won't match any experiments
        result = self.runner.invoke(
            cli, ["archive", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        assert "No experiments found to archive" in result.output

    def test_delete_with_filters_no_matches(self):
        """Test delete command with filters that match no experiments."""
        # Use a name pattern that won't match any experiments
        result = self.runner.invoke(
            cli, ["delete", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        assert "No regular experiments found to delete" in result.output

    def test_unarchive_with_filters_no_matches(self):
        """Test unarchive command with filters that match no experiments."""
        # Use a name pattern that won't match any experiments
        result = self.runner.invoke(
            cli, ["unarchive", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        assert "No archived experiments found to unarchive" in result.output

    def test_archive_with_valid_identifiers_only(self):
        """Test archive command accepts experiment identifiers without filters."""
        # This will fail since the experiment doesn't exist, but validates the CLI parsing
        result = self.runner.invoke(cli, ["archive", "exp1", "exp2"])
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_archive_with_valid_filters_only(self):
        """Test archive command accepts filters without identifiers."""
        # Use --force to skip confirmation and a non-matching pattern
        result = self.runner.invoke(
            cli, ["archive", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_delete_with_valid_identifiers_only(self):
        """Test delete command accepts experiment identifiers without filters."""
        result = self.runner.invoke(cli, ["delete", "exp1", "exp2"])
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_delete_with_valid_filters_only(self):
        """Test delete command accepts filters without identifiers."""
        result = self.runner.invoke(
            cli, ["delete", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_unarchive_with_valid_identifiers_only(self):
        """Test unarchive command accepts experiment identifiers without filters."""
        result = self.runner.invoke(cli, ["unarchive", "exp1", "exp2"])
        assert result.exit_code == 1
        # Should not show mutual exclusivity error, should show experiment not found
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )

    def test_unarchive_with_valid_filters_only(self):
        """Test unarchive command accepts filters without identifiers."""
        result = self.runner.invoke(
            cli, ["unarchive", "--name", "definitely-nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        # Should not show mutual exclusivity error
        assert (
            "Cannot use both experiment identifiers and filter options"
            not in result.output
        )
