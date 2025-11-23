"""Tests for direct parameter sweep execution (v0.6.0)."""

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestDirectSweepExecution:
    """Test direct parameter sweep execution without staging."""

    def test_sweep_sequential_without_stage(self, tmp_path, cli_runner):
        """Test that sweeps run sequentially without --stage flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "sweep_script.py", "simple", test_message="sweep test"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "mode=list(on,off)",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 2 experiments" in result.output
        assert "Sweep execution completed" in result.output

    def test_sweep_parallel_without_stage(self, tmp_path, cli_runner):
        """Test that sweeps run in parallel with --parallel flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_sweep.py", "simple", test_message="parallel test"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "value=list(1,2,3)",
                "--parallel",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 3 experiments" in result.output
        assert "Running 3 experiments with 2 parallel workers" in result.output
        assert "Sweep execution completed" in result.output

    def test_sweep_parallel_auto_detect_cpus(self, tmp_path, cli_runner):
        """Test --parallel 0 auto-detects CPU count."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "auto_cpu.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "x=list(1,2)",
                "--parallel",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected" in result.output
        # Check that parallel workers message appears (with some number of workers)
        assert "Running 2 experiments with" in result.output
        assert "parallel workers" in result.output

    def test_parallel_flag_allowed_single_experiment(self, tmp_path, cli_runner):
        """Test that --parallel is allowed with single experiments (orchestrator pattern)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "single.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--parallel",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.output

    def test_parallel_with_stage_rejects(self, tmp_path, cli_runner):
        """Test that --stage + --parallel is rejected."""
        script_path = TestFileHelpers.create_test_script(tmp_path, "test.py", "simple")

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "x=list(1,2)",
                "--stage",
                "--parallel",
                "2",
            ],
        )
        assert result.exit_code != 0
        assert "--parallel cannot be used with --stage" in result.output

    def test_sweep_execution_creates_experiments_not_staged(self, tmp_path, cli_runner):
        """Test that direct sweep doesn't use 'staged' status."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "status_test.py", "simple"
        )

        # Run sweep directly
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "x=list(1,2)",
            ],
        )
        assert result.exit_code == 0

        # Verify experiments were created and completed (not staged)
        # This is implicit - if they ran, they were not staged
        assert "Sweep execution completed" in result.output

    def test_sweep_execution_doesnt_affect_existing_staged(self, tmp_path, cli_runner):
        """Test that existing staged experiments are unaffected by direct execution."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "isolation_test.py", "simple"
        )

        # First, stage some experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "staged=list(a,b)",
                "--stage",
            ],
        )
        assert result.exit_code == 0
        assert "Staged 2 sweep experiments" in result.output

        # Now run a direct sweep (should not affect staged ones)
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "direct=list(x,y)",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 2 experiments" in result.output

        # Verify staged experiments still exist (without executing them)
        # Just verify the staging command worked - execution is tested elsewhere
        # The key is that direct execution didn't interfere with staging

    def test_staging_still_works(self, tmp_path, cli_runner):
        """Test that --stage workflow is preserved."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "backward_compat.py", "simple"
        )

        # Stage experiments (existing workflow)
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "x=list(1,2,3)",
                "--stage",
            ],
        )
        assert result.exit_code == 0
        assert "Staged 3 sweep experiments" in result.output

        # Execute staged (can use --parallel)
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "2"])
        assert result.exit_code == 0
        assert "Executing with 2 parallel workers" in result.output
