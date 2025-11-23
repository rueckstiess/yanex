"""Tests for parallel experiment execution."""

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestParallelExecution:
    """Test parallel experiment execution functionality."""

    def test_parallel_flag_allowed_with_single_experiment(self, tmp_path, cli_runner):
        """Test that --parallel is allowed with single experiments (orchestrator pattern)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Single experiment with --parallel should succeed (flag captured in cli_args)
        result = cli_runner.invoke(
            cli, ["run", str(script_path), "--parallel", "2", "--ignore-dirty"]
        )
        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.output

    def test_parallel_negative_value_rejected(self, tmp_path, cli_runner):
        """Test that negative --parallel values are rejected."""
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "-1"])
        assert result.exit_code != 0
        assert "--parallel must be 0 (auto) or positive" in result.output

    def test_parallel_zero_accepted(
        self, tmp_path, cli_runner, per_test_experiments_dir
    ):
        """Test that --parallel 0 (auto) is accepted."""
        # Should not error on validation (even if no staged experiments exist)
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "0"])
        assert result.exit_code == 0
        assert "No staged experiments found" in result.output

    def test_parallel_positive_value_accepted(
        self, tmp_path, cli_runner, per_test_experiments_dir
    ):
        """Test that positive --parallel values are accepted."""
        # Should not error on validation (even if no staged experiments exist)
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "4"])
        assert result.exit_code == 0
        assert "No staged experiments found" in result.output

    def test_parallel_short_flag(self, tmp_path, cli_runner, per_test_experiments_dir):
        """Test that -j short flag works for --parallel."""
        # Should not error on validation
        result = cli_runner.invoke(cli, ["run", "--staged", "-j", "2"])
        assert result.exit_code == 0
        assert "No staged experiments found" in result.output

    def test_sequential_execution_still_works(
        self, tmp_path, cli_runner, per_test_experiments_dir
    ):
        """Test that sequential execution (no --parallel) still works."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Stage a single experiment
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--name",
                "sequential-test",
            ],
        )
        assert result.exit_code == 0
        assert "Experiment staged" in result.output

        # Execute without --parallel (sequential mode)
        result = cli_runner.invoke(cli, ["run", "--staged"])
        assert result.exit_code == 0
        # Should complete successfully

    def test_parallel_execution_with_staged_experiments(
        self, tmp_path, cli_runner, per_test_experiments_dir
    ):
        """Test parallel execution with multiple staged experiments."""
        # Create a simple test script
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_test.py", "simple", test_message="parallel test"
        )

        # Stage multiple experiments using parameter sweep
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "lr=list(0.01, 0.02)",
                "--stage",
                "--name",
                "parallel-sweep",
            ],
        )
        assert result.exit_code == 0
        assert "Staged 2 sweep experiments" in result.output

        # Execute with parallel mode
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "2"])
        assert result.exit_code == 0
        assert "Executing with 2 parallel workers" in result.output
        assert "Execution Summary" in result.output
