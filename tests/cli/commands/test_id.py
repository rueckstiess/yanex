"""Tests for the id command."""

import json

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestIdCommandHelp:
    """Test id command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_help_output(self):
        """Test that id command shows help information."""
        result = self.runner.invoke(cli, ["id", "--help"])
        assert result.exit_code == 0
        assert "Output experiment IDs" in result.output
        assert "--limit" in result.output
        assert "--format" in result.output
        assert "csv" in result.output
        assert "json" in result.output
        assert "line" in result.output

    def test_id_help_shows_examples(self):
        """Test that help includes usage examples."""
        result = self.runner.invoke(cli, ["id", "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.output
        assert "yanex id" in result.output


class TestIdCommandBasicBehavior:
    """Test id command basic behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_basic_invocation(self):
        """Test basic id invocation works without errors."""
        result = self.runner.invoke(cli, ["id"])
        assert result.exit_code == 0

    def test_id_no_experiments_outputs_nothing(self):
        """Test that id outputs nothing when no experiments found."""
        result = self.runner.invoke(cli, ["id"])
        assert result.exit_code == 0
        # Should output nothing (allows safe shell composition)
        assert result.output.strip() == ""


class TestIdCommandOutputFormats:
    """Test id command output format options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_csv_format_default(self, clean_git_repo, sample_experiment_script):
        """Test CSV format is the default."""
        # Create a couple of experiments
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp1"]
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp2"]
        )

        result = self.runner.invoke(cli, ["id"])
        assert result.exit_code == 0

        # Should be comma-separated
        output = result.output.strip()
        assert "," in output
        # Should have two IDs
        ids = output.split(",")
        assert len(ids) == 2
        # Each ID should be 8 characters
        for exp_id in ids:
            assert len(exp_id) == 8

    def test_id_csv_format_explicit(self, clean_git_repo, sample_experiment_script):
        """Test CSV format when explicitly specified."""
        # Create an experiment
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])

        result = self.runner.invoke(cli, ["id", "--format", "csv"])
        assert result.exit_code == 0

        # Should have single ID (no comma)
        output = result.output.strip()
        assert len(output) == 8

    def test_id_line_format(self, clean_git_repo, sample_experiment_script):
        """Test line format (one ID per line)."""
        # Create a couple of experiments
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp1"]
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp2"]
        )

        result = self.runner.invoke(cli, ["id", "--format", "line"])
        assert result.exit_code == 0

        # Should have two lines
        lines = result.output.strip().split("\n")
        assert len(lines) == 2
        # Each line should be an 8-char ID
        for line in lines:
            assert len(line) == 8

    def test_id_json_format(self, clean_git_repo, sample_experiment_script):
        """Test JSON format."""
        # Create a couple of experiments
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp1"]
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "exp2"]
        )

        result = self.runner.invoke(cli, ["id", "--format", "json"])
        assert result.exit_code == 0

        # Should be valid JSON array
        ids = json.loads(result.output.strip())
        assert isinstance(ids, list)
        assert len(ids) == 2
        # Each ID should be 8 characters
        for exp_id in ids:
            assert len(exp_id) == 8

    def test_id_json_format_empty(self):
        """Test JSON format with no experiments."""
        result = self.runner.invoke(cli, ["id", "--format", "json"])
        assert result.exit_code == 0

        # Should output nothing (for shell composition safety)
        output = result.output.strip()
        assert output == ""


class TestIdCommandLimit:
    """Test id command limit option."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_with_limit(self, clean_git_repo, sample_experiment_script):
        """Test limiting number of results."""
        # Create 5 experiments
        for i in range(5):
            self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp{i}"]
            )

        result = self.runner.invoke(cli, ["id", "--limit", "3"])
        assert result.exit_code == 0

        # Should have exactly 3 IDs
        ids = result.output.strip().split(",")
        assert len(ids) == 3

    def test_id_with_limit_json(self, clean_git_repo, sample_experiment_script):
        """Test limit with JSON format."""
        # Create 5 experiments
        for i in range(5):
            self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp{i}"]
            )

        result = self.runner.invoke(cli, ["id", "--limit", "2", "--format", "json"])
        assert result.exit_code == 0

        # Should have exactly 2 IDs in JSON array
        ids = json.loads(result.output.strip())
        assert len(ids) == 2

    def test_id_with_limit_line(self, clean_git_repo, sample_experiment_script):
        """Test limit with line format."""
        # Create 5 experiments
        for i in range(5):
            self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"exp{i}"]
            )

        result = self.runner.invoke(cli, ["id", "--limit", "3", "--format", "line"])
        assert result.exit_code == 0

        # Should have exactly 3 lines
        lines = result.output.strip().split("\n")
        assert len(lines) == 3


class TestIdCommandFiltering:
    """Test id command with filter options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_with_status_filter(self, clean_git_repo, sample_experiment_script):
        """Test filtering by status."""
        # Create experiments
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "completed-exp"]
        )

        result = self.runner.invoke(cli, ["id", "--status", "completed"])
        assert result.exit_code == 0

        # Should have at least one ID
        assert len(result.output.strip()) >= 8

    def test_id_with_name_filter(self, clean_git_repo, sample_experiment_script):
        """Test filtering by name pattern."""
        # Create experiments with different names
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train-v1"]
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train-v2"]
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "eval-v1"]
        )

        result = self.runner.invoke(cli, ["id", "--name", "train-*"])
        assert result.exit_code == 0

        # Should have 2 IDs (train-v1 and train-v2)
        ids = result.output.strip().split(",")
        assert len(ids) == 2

    def test_id_with_tag_filter(self, clean_git_repo, sample_experiment_script):
        """Test filtering by tag."""
        # Create experiments with tags
        self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "tagged", "--tag", "ml"],
        )
        self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "untagged"]
        )

        result = self.runner.invoke(cli, ["id", "--tag", "ml"])
        assert result.exit_code == 0

        # Should have 1 ID (the tagged one)
        ids = result.output.strip().split(",")
        assert len(ids) == 1

    def test_id_with_script_filter(self, clean_git_repo, temp_dir):
        """Test filtering by script name."""
        # Create two different scripts
        script1 = temp_dir / "train.py"
        script1.write_text("print('training')")

        script2 = temp_dir / "eval.py"
        script2.write_text("print('evaluation')")

        # Run with each script
        self.runner.invoke(cli, ["run", str(script1)])
        self.runner.invoke(cli, ["run", str(script2)])

        result = self.runner.invoke(cli, ["id", "--script", "train.py"])
        assert result.exit_code == 0

        # Should have 1 ID
        ids = result.output.strip().split(",")
        assert len(ids) == 1

    def test_id_with_multiple_filters(self, clean_git_repo, sample_experiment_script):
        """Test combining multiple filters."""
        # Create various experiments
        self.runner.invoke(
            cli,
            [
                "run",
                str(sample_experiment_script),
                "--name",
                "ml-exp",
                "--tag",
                "ml",
                "--tag",
                "experiment",
            ],
        )
        self.runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--name", "other", "--tag", "ml"],
        )

        result = self.runner.invoke(
            cli, ["id", "--tag", "experiment", "--name", "ml-*"]
        )
        assert result.exit_code == 0

        # Should have 1 ID (ml-exp with both tags and matching name)
        output = result.output.strip()
        assert len(output) == 8  # Single ID, no comma


class TestIdCommandEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_single_experiment_csv(self, clean_git_repo, sample_experiment_script):
        """Test CSV format with single experiment (no trailing comma)."""
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])

        result = self.runner.invoke(cli, ["id", "--format", "csv"])
        assert result.exit_code == 0

        # Should be single ID with no comma
        output = result.output.strip()
        assert "," not in output
        assert len(output) == 8

    def test_id_single_experiment_line(self, clean_git_repo, sample_experiment_script):
        """Test line format with single experiment."""
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])

        result = self.runner.invoke(cli, ["id", "--format", "line"])
        assert result.exit_code == 0

        # Should be single line
        lines = result.output.strip().split("\n")
        assert len(lines) == 1
        assert len(lines[0]) == 8

    def test_id_with_limit_exceeding_results(
        self, clean_git_repo, sample_experiment_script
    ):
        """Test limit larger than available results."""
        # Create 2 experiments
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])

        result = self.runner.invoke(cli, ["id", "--limit", "10"])
        assert result.exit_code == 0

        # Should have only 2 IDs (all available)
        ids = result.output.strip().split(",")
        assert len(ids) == 2

    def test_id_with_zero_limit(self, clean_git_repo, sample_experiment_script):
        """Test with limit of zero."""
        self.runner.invoke(cli, ["run", str(sample_experiment_script)])

        result = self.runner.invoke(cli, ["id", "--limit", "0"])
        assert result.exit_code == 0

        # Should output nothing
        assert result.output.strip() == ""
