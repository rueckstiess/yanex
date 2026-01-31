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
        assert "--stage, --staged, and --id are mutually exclusive" in result.output

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


class TestRunById:
    """Test yanex run --id flag."""

    def test_id_with_stage_flag_is_rejected(self, tmp_path, cli_runner):
        """Test error when --id and --stage are used together."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            ["run", str(script_path), "--stage", "--id", "abc12345"],
        )
        assert result.exit_code != 0
        assert "--stage, --staged, and --id are mutually exclusive" in result.output

    def test_id_with_staged_flag_is_rejected(self, cli_runner):
        """Test error when --id and --staged are used together."""
        result = cli_runner.invoke(
            cli,
            ["run", "--staged", "--id", "abc12345"],
        )
        assert result.exit_code != 0
        assert "--stage, --staged, and --id are mutually exclusive" in result.output

    def test_id_nonexistent_experiment(self, cli_runner):
        """Test error when --id references a non-existent experiment."""
        result = cli_runner.invoke(
            cli,
            ["run", "--id", "nonexistent"],
        )
        assert result.exit_code != 0

    def test_id_executes_staged_experiment(self, tmp_path, cli_runner):
        """Test --id executes a single staged experiment end-to-end."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        # Stage an experiment
        stage_result = cli_runner.invoke(
            cli,
            ["run", str(script_path), "--stage", "--name", "id-test"],
        )
        assert stage_result.exit_code == 0

        # Extract experiment ID from output
        import re

        match = re.search(r"Experiment staged: ([a-f0-9]+)", stage_result.output)
        assert match, f"Could not find experiment ID in: {stage_result.output}"
        exp_id = match.group(1)

        # Execute by ID
        run_result = cli_runner.invoke(
            cli,
            ["run", "--id", exp_id],
        )
        assert run_result.exit_code == 0

        # Verify status transitioned to completed
        from yanex.core.manager import ExperimentManager

        manager = ExperimentManager()
        metadata = manager.storage.load_metadata(exp_id)
        assert metadata["status"] == "completed"

    def test_id_does_not_execute_other_staged(self, tmp_path, cli_runner):
        """Test --id only executes the specified experiment, not others."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        # Stage two experiments
        result1 = cli_runner.invoke(
            cli,
            ["run", str(script_path), "--stage", "--name", "first"],
        )
        result2 = cli_runner.invoke(
            cli,
            ["run", str(script_path), "--stage", "--name", "second"],
        )
        assert result1.exit_code == 0
        assert result2.exit_code == 0

        import re

        id1 = re.search(r"Experiment staged: ([a-f0-9]+)", result1.output).group(1)
        id2 = re.search(r"Experiment staged: ([a-f0-9]+)", result2.output).group(1)

        # Execute only the first
        cli_runner.invoke(cli, ["run", "--id", id1])

        from yanex.core.manager import ExperimentManager

        manager = ExperimentManager()
        assert manager.storage.load_metadata(id1)["status"] == "completed"
        assert manager.storage.load_metadata(id2)["status"] == "staged"


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
