"""Tests for parallel experiment execution."""

import os

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestParallelExecution:
    """Test parallel experiment execution functionality."""

    def test_parallel_flag_requires_staged_or_sweep(self, tmp_path, cli_runner):
        """Test that --parallel requires --staged or parameter sweep (v0.6.0)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Single experiment without sweep should error
        result = cli_runner.invoke(
            cli, ["run", str(script_path), "--parallel", "2", "--ignore-dirty"]
        )
        assert result.exit_code != 0
        assert (
            "--parallel can only be used with parameter sweeps or --staged"
            in result.output
        )

    def test_parallel_negative_value_rejected(self, tmp_path, cli_runner):
        """Test that negative --parallel values are rejected."""
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "-1"])
        assert result.exit_code != 0
        assert "--parallel must be 0 (auto) or positive" in result.output

    def test_parallel_zero_accepted(self, tmp_path, cli_runner):
        """Test that --parallel 0 (auto) is accepted."""
        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Should not error on validation (even if no staged experiments exist)
            result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "0"])
            assert result.exit_code == 0
            assert "No staged experiments found" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_parallel_positive_value_accepted(self, tmp_path, cli_runner):
        """Test that positive --parallel values are accepted."""
        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Should not error on validation (even if no staged experiments exist)
            result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "4"])
            assert result.exit_code == 0
            assert "No staged experiments found" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_parallel_short_flag(self, tmp_path, cli_runner):
        """Test that -j short flag works for --parallel."""
        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Should not error on validation
            result = cli_runner.invoke(cli, ["run", "--staged", "-j", "2"])
            assert result.exit_code == 0
            assert "No staged experiments found" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_sequential_execution_still_works(self, tmp_path, cli_runner):
        """Test that sequential execution (no --parallel) still works."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Stage a single experiment
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--stage",
                    "--ignore-dirty",
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
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_parallel_execution_with_staged_experiments(self, tmp_path, cli_runner):
        """Test parallel execution with multiple staged experiments."""
        # Create a simple test script
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_test.py", "simple", test_message="parallel test"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Stage multiple experiments using parameter sweep
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "lr=list(0.01, 0.02)",
                    "--stage",
                    "--ignore-dirty",
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
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]
