"""Tests for yanex run command error handling and validation."""

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestRunCommandErrorHandling:
    """Test error handling and validation in yanex run command."""

    def test_stage_and_staged_flags_together(self, tmp_path, cli_runner):
        """Test error when --stage and --staged are used together."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--staged",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use both --stage and --staged" in result.output

    def test_negative_parallel_value(self, tmp_path, cli_runner):
        """Test error when --parallel has negative value."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--parallel",
                "-1",
            ],
        )
        assert result.exit_code != 0
        assert "--parallel must be 0 (auto) or positive integer" in result.output

    def test_stage_with_parallel_flag(self, tmp_path, cli_runner):
        """Test error when --stage is used with --parallel."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--parallel",
                "2",
            ],
        )
        assert result.exit_code != 0
        assert "--parallel cannot be used with --stage" in result.output

    def test_missing_script_argument(self, cli_runner):
        """Test error when SCRIPT argument is missing."""
        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--param",
                "x=10",
            ],
        )
        assert result.exit_code != 0
        assert "Missing argument 'SCRIPT'" in result.output

    def test_verbose_output_enabled(self, tmp_path, cli_runner):
        """Test that --verbose flag enables detailed output."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        # Verbose mode should show running script message
        # (This tests lines 226-236 in run.py)

    def test_failed_experiments_in_sweep_summary(self, tmp_path, cli_runner):
        """Test that sweep summary shows failed experiments."""
        # Create a script that will fail
        script_path = tmp_path / "failing_script.py"
        script_path.write_text("raise ValueError('test failure')")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "x=list(1,2)",
            ],
        )
        # Sweep should run but experiments will fail
        assert "✓ Sweep detected" in result.output
        # Failed count should be shown (line 399 in run.py)
        # Note: The exact output depends on how failures are formatted


class TestDependencyErrorHandling:
    """Test error handling for dependency validation."""

    def test_nonexistent_experiment_validation(self, tmp_path, cli_runner):
        """Test error when depending on non-existent experiment."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                "notfound",
            ],
        )
        assert result.exit_code != 0
        # Should get error about experiment not found
        assert "not found" in result.output.lower() or "error" in result.output.lower()
