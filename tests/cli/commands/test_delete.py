"""
Tests for delete command functionality.

This module tests the delete command for permanently removing experiments.
"""

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestDeleteCommandHelp:
    """Test delete command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_delete_help_output(self):
        """Test that delete command shows help information."""
        result = self.runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "Permanently delete experiments" in result.output
        assert "WARNING: This operation cannot be undone!" in result.output
        assert "--force" in result.output
        assert "--archived" in result.output

    def test_delete_help_shows_examples(self):
        """Test that help includes usage examples."""
        result = self.runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.output or "yanex delete" in result.output


class TestDeleteCommandValidation:
    """Test delete command validation and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_delete_requires_identifiers_or_filters(self):
        """Test that delete requires either identifiers or filters."""
        result = self.runner.invoke(cli, ["delete", "--force"])
        # Should fail with error about missing identifiers/filters
        assert result.exit_code == 1
        assert (
            "Must specify either experiment identifiers or filter options"
            in result.output
        )

    def test_delete_mutual_exclusivity_identifiers_and_filters(self):
        """Test that identifiers and filters are mutually exclusive."""
        result = self.runner.invoke(
            cli, ["delete", "exp1", "--status", "failed", "--force"]
        )
        assert result.exit_code == 1
        assert (
            "Cannot use both experiment identifiers and filter options" in result.output
        )

    def test_delete_nonexistent_experiment(self):
        """Test deleting non-existent experiment."""
        result = self.runner.invoke(cli, ["delete", "nonexistent123", "--force"])
        assert result.exit_code == 1
        assert (
            "No regular experiment found with ID or name 'nonexistent123'"
            in result.output
        )


class TestDeleteCommandFiltering:
    """Test delete command filtering behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_delete_with_status_filter_no_matches(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test delete with status filter that doesn't match created experiment."""
        # Create a completed experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "completed-exp"]
        )
        assert result.exit_code == 0

        # Extract experiment ID to verify it's not deleted
        lines = result.output.split("\n")
        exp_id = None
        for line in lines:
            if "Experiment completed successfully:" in line:
                exp_id = line.split("Experiment completed successfully:")[1].strip()
                break

        # Try to delete running experiments (should not match our completed one)
        result = self.runner.invoke(cli, ["delete", "--status", "running", "--force"])
        assert result.exit_code == 0

        # Verify our completed experiment still exists by trying to show it
        if exp_id:
            result = self.runner.invoke(cli, ["show", exp_id])
            assert result.exit_code == 0
            assert "completed-exp" in result.output

    def test_delete_with_name_pattern_no_matches(self, clean_git_repo):
        """Test delete with name pattern that matches nothing."""
        # In a clean environment, no experiments with this pattern should exist
        result = self.runner.invoke(
            cli, ["delete", "--name", "nonexistent-*", "--force"]
        )
        assert result.exit_code == 0
        assert "No regular experiments found to delete" in result.output

    def test_delete_archived_flag_searches_archived(self):
        """Test that --archived flag searches archived experiments."""
        result = self.runner.invoke(cli, ["delete", "exp1", "--archived", "--force"])
        assert result.exit_code == 1
        assert "No archived experiment found with ID or name 'exp1'" in result.output


class TestDeleteCommandIntegration:
    """Integration tests for delete command with real experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_delete_experiment_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test deleting a real experiment end-to-end."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "to-delete"]
        )
        assert result.exit_code == 0

        # Extract experiment ID
        exp_id = None
        for line in result.output.split("\n"):
            if (
                "Experiment completed successfully:" in line
                or "Experiment staged:" in line
            ):
                exp_id = line.split(":")[-1].strip()
                break

        assert exp_id is not None

        # Delete the experiment
        result = self.runner.invoke(cli, ["delete", exp_id, "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted 1 experiment" in result.output

        # Verify it's gone by trying to show it
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 1
        assert "No experiment found" in result.output

    def test_delete_multiple_experiments_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test deleting multiple experiments end-to-end."""
        # Create multiple experiments
        exp_ids = []
        for i in range(3):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"to-delete-{i}"]
            )
            assert result.exit_code == 0

            # Extract experiment ID
            for line in result.output.split("\n"):
                if (
                    "Experiment completed successfully:" in line
                    or "Experiment staged:" in line
                ):
                    exp_id = line.split(":")[-1].strip()
                    exp_ids.append(exp_id)
                    break

        assert len(exp_ids) == 3

        # Delete all experiments
        result = self.runner.invoke(cli, ["delete"] + exp_ids + ["--force"])

        assert result.exit_code == 0
        assert "Successfully deleted 3 experiments" in result.output

        # Verify they're all gone
        for exp_id in exp_ids:
            result = self.runner.invoke(cli, ["show", exp_id])
            assert result.exit_code == 1

    def test_delete_with_confirmation_prompts(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test delete with confirmation prompts."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "to-confirm-delete"]
        )
        assert result.exit_code == 0

        # Extract experiment ID
        exp_id = None
        for line in result.output.split("\n"):
            if (
                "Experiment completed successfully:" in line
                or "Experiment staged:" in line
            ):
                exp_id = line.split(":")[-1].strip()
                break

        assert exp_id is not None

        # Try to delete with "no" confirmation
        result = self.runner.invoke(cli, ["delete", exp_id], input="n\n")

        assert result.exit_code == 0
        assert "Delete this experiment?" in result.output
        assert "Delete operation cancelled" in result.output

        # Verify experiment still exists
        result = self.runner.invoke(cli, ["show", exp_id])
        assert result.exit_code == 0

        # Now delete with "yes" confirmation
        result = self.runner.invoke(cli, ["delete", exp_id], input="y\n")

        assert result.exit_code == 0
        assert "Successfully deleted 1 experiment" in result.output

    def test_delete_multiple_with_double_confirmation(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that bulk delete shows double confirmation."""
        # Create multiple experiments
        exp_ids = []
        for i in range(2):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"bulk-delete-{i}"],
            )
            assert result.exit_code == 0

            # Extract experiment ID
            for line in result.output.split("\n"):
                if (
                    "Experiment completed successfully:" in line
                    or "Experiment staged:" in line
                ):
                    exp_id = line.split(":")[-1].strip()
                    exp_ids.append(exp_id)
                    break

        assert len(exp_ids) == 2

        # Try to delete but decline second confirmation
        result = self.runner.invoke(cli, ["delete"] + exp_ids, input="y\nn\n")

        assert result.exit_code == 0
        assert "Delete these 2 experiments?" in result.output
        assert (
            "You are about to permanently delete multiple experiments" in result.output
        )
        assert "Are you absolutely sure?" in result.output
        assert "Delete operation cancelled" in result.output

        # Verify experiments still exist
        for exp_id in exp_ids:
            result = self.runner.invoke(cli, ["show", exp_id])
            assert result.exit_code == 0

    def test_delete_by_name_pattern_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test deleting experiments by name pattern."""
        # Create experiments with specific name pattern
        for i in range(3):
            result = self.runner.invoke(
                cli,
                ["run", str(sample_experiment_script), "--name", f"test-pattern-{i}"],
            )
            assert result.exit_code == 0

        # Create an experiment that doesn't match the pattern
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "keep-this"]
        )
        assert result.exit_code == 0

        # Delete experiments matching pattern
        result = self.runner.invoke(
            cli, ["delete", "--name", "test-pattern-*", "--force"]
        )

        assert result.exit_code == 0
        assert (
            "Deleting 3 experiment(s)..." in result.output
            or "Successfully deleted 3 experiments" in result.output
        )

        # Verify the one that doesn't match still exists
        result = self.runner.invoke(cli, ["list", "--name", "keep-this"])
        assert result.exit_code == 0
        assert "keep-this" in result.output
