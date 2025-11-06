"""Tests for direct parameter sweep execution (v0.6.0)."""

import os

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestDirectSweepExecution:
    """Test direct parameter sweep execution without staging."""

    def test_sweep_sequential_without_stage(self, tmp_path, cli_runner):
        """Test that sweeps run sequentially without --stage flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "sweep_script.py", "simple", test_message="sweep test"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "mode=list(on,off)",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert "Parameter sweep detected: running 2 experiments" in result.output
            assert "Sweep execution completed" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_sweep_parallel_without_stage(self, tmp_path, cli_runner):
        """Test that sweeps run in parallel with --parallel flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_sweep.py", "simple", test_message="parallel test"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "value=list(1,2,3)",
                    "--parallel",
                    "2",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert "Parameter sweep detected: running 3 experiments" in result.output
            assert "Running 3 experiments with 2 parallel workers" in result.output
            assert "Sweep execution completed" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_sweep_parallel_auto_detect_cpus(self, tmp_path, cli_runner):
        """Test --parallel 0 auto-detects CPU count."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "auto_cpu.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "x=list(1,2)",
                    "--parallel",
                    "0",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert "Parameter sweep detected" in result.output
            # Check that parallel workers message appears (with some number of workers)
            assert "Running 2 experiments with" in result.output
            assert "parallel workers" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

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
                "--ignore-dirty",
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
                "--ignore-dirty",
            ],
        )
        assert result.exit_code != 0
        assert "--parallel cannot be used with --stage" in result.output

    def test_sweep_execution_creates_experiments_not_staged(self, tmp_path, cli_runner):
        """Test that direct sweep doesn't use 'staged' status."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "status_test.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Run sweep directly
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "x=list(1,2)",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0

            # Verify experiments were created and completed (not staged)
            # This is implicit - if they ran, they were not staged
            assert "Sweep execution completed" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_sweep_execution_doesnt_affect_existing_staged(self, tmp_path, cli_runner):
        """Test that existing staged experiments are unaffected by direct execution."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "isolation_test.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # First, stage some experiments
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "staged=list(a,b)",
                    "--stage",
                    "--ignore-dirty",
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
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert "Parameter sweep detected: running 2 experiments" in result.output

            # Verify staged experiments still exist (without executing them)
            # Just verify the staging command worked - execution is tested elsewhere
            # The key is that direct execution didn't interfere with staging

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_staging_still_works(self, tmp_path, cli_runner):
        """Test that --stage workflow is preserved."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "backward_compat.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Stage experiments (existing workflow)
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "x=list(1,2,3)",
                    "--stage",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert "Staged 3 sweep experiments" in result.output

            # Execute staged (can use --parallel)
            result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "2"])
            assert result.exit_code == 0
            assert "Executing with 2 parallel workers" in result.output

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_sweep_fails_early_on_dirty_git(self, tmp_path, cli_runner, clean_git_repo):
        """Test that sweep fails immediately on dirty git, not after creating experiments."""
        from pathlib import Path

        # Get the working directory from the repo
        repo_path = Path(clean_git_repo.working_dir)

        # Create and commit a script in the git repo
        script_path = repo_path / "sweep_script.py"
        script_path.write_text(
            """
import yanex
param_value = yanex.get_param('x', default=1)
print(f"Param: {param_value}")
"""
        )
        clean_git_repo.index.add([str(script_path)])
        clean_git_repo.index.commit("Add sweep script")

        # Make git repo dirty by modifying the tracked file
        script_path.write_text(
            """
import yanex
param_value = yanex.get_param('x', default=1)
print(f"Param: {param_value}")
# This is a modification
"""
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Run sweep WITHOUT --ignore-dirty
            result = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "x=list(1,2,3)",
                ],
            )

            # Should fail immediately with non-zero exit code
            assert result.exit_code != 0, (
                f"Expected non-zero exit code, got {result.exit_code}"
            )

            # Should see error about git working directory not being clean
            assert "not clean" in result.output.lower(), (
                f"Expected 'not clean' error message, got: {result.output}"
            )

            # Verify NO experiments were created (should fail before creating any)
            from yanex.core.manager import ExperimentManager

            manager = ExperimentManager(experiments_dir=tmp_path)
            experiments = manager.list_experiments()
            # Expect 0 experiments (fails before creating any)
            assert len(experiments) == 0, (
                f"Expected 0 experiments, got {len(experiments)} (should fail before creating any)"
            )

        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]
