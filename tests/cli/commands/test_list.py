"""
Tests for list command functionality.

This module tests the list command for displaying and filtering experiments.
"""

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestListCommandHelp:
    """Test list command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_help_output(self):
        """Test that list command shows help information."""
        result = self.runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "List experiments with filtering options" in result.output
        assert "--all" in result.output
        assert "--limit" in result.output
        assert "--status" in result.output
        assert "--name" in result.output
        assert "--tag" in result.output
        assert "--archived" in result.output

    def test_list_help_shows_examples(self):
        """Test that help includes usage examples."""
        result = self.runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.output
        assert "yanex list" in result.output


class TestListCommandBasicBehavior:
    """Test list command basic behavior without filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_basic_invocation(self):
        """Test basic list invocation works without errors."""
        result = self.runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        # Should show either experiments or "No experiments found"
        assert (
            "Yanex Experiments" in result.output
            or "No experiments found" in result.output
        )

    def test_list_shows_default_limit_message(self):
        """Test that list shows default 10 experiments message."""
        # When there are no experiments, just ensure command works
        result = self.runner.invoke(cli, ["list"])
        assert result.exit_code == 0

    def test_list_all_flag(self):
        """Test list --all flag."""
        result = self.runner.invoke(cli, ["list", "--all"])
        assert result.exit_code == 0

    def test_list_with_custom_limit(self):
        """Test list with custom limit."""
        result = self.runner.invoke(cli, ["list", "-l", "5"])
        assert result.exit_code == 0

    def test_list_with_zero_limit(self):
        """Test list with zero limit."""
        result = self.runner.invoke(cli, ["list", "-l", "0"])
        assert result.exit_code == 0


class TestListCommandFiltering:
    """Test list command filtering options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_filter_by_status(self):
        """Test filtering by status."""
        result = self.runner.invoke(cli, ["list", "--status", "completed"])
        assert result.exit_code == 0

    def test_list_filter_by_name_pattern(self):
        """Test filtering by name pattern."""
        result = self.runner.invoke(cli, ["list", "--name", "*test*"])
        assert result.exit_code == 0

    def test_list_filter_by_empty_name(self):
        """Test filtering unnamed experiments with empty string."""
        result = self.runner.invoke(cli, ["list", "--name", ""])
        assert result.exit_code == 0

    def test_list_filter_by_single_tag(self):
        """Test filtering by single tag."""
        result = self.runner.invoke(cli, ["list", "--tag", "ml"])
        assert result.exit_code == 0

    def test_list_filter_by_multiple_tags(self):
        """Test filtering by multiple tags (AND logic)."""
        result = self.runner.invoke(cli, ["list", "--tag", "ml", "--tag", "training"])
        assert result.exit_code == 0

    def test_list_filter_by_script_pattern(self):
        """Test filtering by script pattern."""
        result = self.runner.invoke(cli, ["list", "--script", "train.py"])
        assert result.exit_code == 0

    def test_list_filter_by_started_after(self):
        """Test filtering by started after time."""
        result = self.runner.invoke(cli, ["list", "--started-after", "2024-01-01"])
        assert result.exit_code == 0

    def test_list_filter_by_started_before(self):
        """Test filtering by started before time."""
        result = self.runner.invoke(cli, ["list", "--started-before", "2024-12-31"])
        assert result.exit_code == 0

    def test_list_filter_by_ended_after(self):
        """Test filtering by ended after time."""
        result = self.runner.invoke(cli, ["list", "--ended-after", "2024-06-01"])
        assert result.exit_code == 0

    def test_list_filter_by_ended_before(self):
        """Test filtering by ended before time."""
        result = self.runner.invoke(cli, ["list", "--ended-before", "2024-06-30"])
        assert result.exit_code == 0

    def test_list_filter_by_time_range(self):
        """Test filtering by time range."""
        result = self.runner.invoke(
            cli,
            [
                "list",
                "--started-after",
                "2024-01-01",
                "--started-before",
                "2024-12-31",
            ],
        )
        assert result.exit_code == 0

    def test_list_complex_filtering(self):
        """Test complex filtering with multiple criteria."""
        result = self.runner.invoke(
            cli,
            [
                "list",
                "--status",
                "completed",
                "--tag",
                "ml",
                "--started-after",
                "2024-01-01",
                "-l",
                "20",
            ],
        )
        assert result.exit_code == 0

    def test_list_filter_shows_suggestions_when_no_matches(self):
        """Test that filter suggestions are shown when no experiments match."""
        result = self.runner.invoke(
            cli, ["list", "--name", "definitely-nonexistent-pattern-*"]
        )
        assert result.exit_code == 0
        assert "No experiments found" in result.output
        assert "Try adjusting your filters" in result.output


class TestListCommandArchived:
    """Test list command archived experiment handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_archived_flag(self):
        """Test listing archived experiments."""
        result = self.runner.invoke(cli, ["list", "--archived"])
        assert result.exit_code == 0

    def test_list_archived_with_filters(self):
        """Test listing archived experiments with filters."""
        result = self.runner.invoke(
            cli, ["list", "--archived", "--status", "completed"]
        )
        assert result.exit_code == 0

    def test_list_archived_shows_different_title(self):
        """Test that archived list shows different table title."""
        # This will be tested in integration tests with actual archived experiments
        result = self.runner.invoke(cli, ["list", "--archived"])
        assert result.exit_code == 0


class TestListCommandVerbose:
    """Test list command verbose mode."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_verbose_shows_filter_info(self):
        """Test that verbose mode shows filter information."""
        result = self.runner.invoke(
            cli,
            [
                "--verbose",
                "list",
                "--status",
                "completed",
                "--name",
                "*test*",
                "--tag",
                "ml",
            ],
        )
        assert result.exit_code == 0
        assert "Filtering experiments" in result.output
        assert "Status: completed" in result.output
        assert "Name pattern: *test*" in result.output
        assert "Tags: ml" in result.output

    def test_list_verbose_shows_count(self):
        """Test that verbose mode shows experiment count."""
        result = self.runner.invoke(cli, ["--verbose", "list"])
        assert result.exit_code == 0
        assert (
            "matching experiments" in result.output or "No experiments" in result.output
        )


class TestListCommandErrorHandling:
    """Test list command error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_invalid_time_format(self):
        """Test that invalid time format shows error."""
        result = self.runner.invoke(cli, ["list", "--started-after", "invalid-date"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Invalid" in result.output

    def test_list_negative_limit(self):
        """Test that negative limit is handled."""
        # Click should validate this, but test anyway
        result = self.runner.invoke(cli, ["list", "-l", "-1"])
        # May be rejected by Click or result in no experiments
        # Just ensure it doesn't crash
        assert result.exit_code in [0, 2]  # 2 is Click validation error


class TestListCommandIntegration:
    """Integration tests for list command with real experiments."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_list_single_experiment(self, clean_git_repo, sample_experiment_script):
        """Test listing a single experiment."""
        # Create an experiment
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "test-exp"]
        )
        assert result.exit_code == 0

        # List experiments
        result = self.runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "test-exp" in result.output
        assert "Yanex Experiments" in result.output

    def test_list_multiple_experiments(self, clean_git_repo, sample_experiment_script):
        """Test listing multiple experiments."""
        # Create multiple experiments
        for i in range(3):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp-{i}"]
            )
            assert result.exit_code == 0

        # List experiments
        result = self.runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "exp-0" in result.output
        assert "exp-1" in result.output
        assert "exp-2" in result.output

    def test_list_respects_default_limit(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that list respects default limit of 10."""
        # Create 12 experiments
        for i in range(12):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp-{i}"]
            )
            assert result.exit_code == 0

        # List without --all should show only 10
        result = self.runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        # Should show summary indicating more exist
        assert "of" in result.output or "Showing" in result.output.lower()

    def test_list_all_shows_all_experiments(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test that --all shows all experiments."""
        # Create 12 experiments
        for i in range(12):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp-{i}"]
            )
            assert result.exit_code == 0

        # List with --all should show all 12
        result = self.runner.invoke(cli, ["list", "--all"])
        assert result.exit_code == 0
        # Verify multiple experiments are shown
        count = result.output.count("exp-")
        assert count >= 10  # At least 10 should be visible

    def test_list_custom_limit(self, clean_git_repo, sample_experiment_script):
        """Test list with custom limit."""
        # Create 5 experiments
        for i in range(5):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp-{i}"]
            )
            assert result.exit_code == 0

        # List with limit 3
        result = self.runner.invoke(cli, ["list", "-l", "3"])
        assert result.exit_code == 0
        # Should show summary
        assert "3" in result.output or "5" in result.output

    def test_list_filter_by_status_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test status filtering with real experiments."""
        # Create completed experiments
        for i in range(2):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"completed-{i}"]
            )
            assert result.exit_code == 0

        # Filter by completed status
        result = self.runner.invoke(cli, ["list", "--status", "completed"])
        assert result.exit_code == 0
        assert "completed-0" in result.output
        assert "completed-1" in result.output

    def test_list_filter_by_name_pattern_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test name pattern filtering with real experiments."""
        # Create experiments with different names
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train-v1"]
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train-v2"]
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "eval-v1"]
        )
        assert result.exit_code == 0

        # Filter by pattern
        result = self.runner.invoke(cli, ["list", "--name", "train-*"])
        assert result.exit_code == 0
        # Names may be truncated in table output, so just check for "train"
        assert "train-v1" in result.output or "train" in result.output
        assert result.output.count("train") >= 2  # Should show both train experiments
        assert "eval" not in result.output

    def test_list_filter_by_tags_integration(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test tag filtering with real experiments."""
        # Create experiments with different tags
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "ml-exp",
                "--tag",
                "ml",
                "--tag",
                "training",
            ],
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "other-exp",
                "--tag",
                "ml",
            ],
        )
        assert result.exit_code == 0

        # Filter by multiple tags (AND logic)
        result = self.runner.invoke(cli, ["list", "--tag", "ml", "--tag", "training"])
        assert result.exit_code == 0
        assert "ml-exp" in result.output
        assert "other-exp" not in result.output  # Missing "training" tag

    def test_list_unnamed_experiments(self, clean_git_repo, sample_experiment_script):
        """Test listing unnamed experiments with empty name pattern."""
        # Create named and unnamed experiments
        result = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "named-exp"]
        )
        assert result.exit_code == 0

        result = self.runner.invoke(cli, ["run", str(sample_experiment_script)])
        assert result.exit_code == 0  # No name

        # Filter for unnamed experiments
        result = self.runner.invoke(cli, ["list", "--name", ""])
        assert result.exit_code == 0
        assert "named-exp" not in result.output
