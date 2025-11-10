"""
Tests for script argument pass-through functionality.

Tests that script-specific arguments are correctly passed through
from yanex run to the executed script.
"""

import os

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestScriptArgumentPassThrough:
    """Test script argument pass-through functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_basic_script_arg_passthrough(self, tmp_path):
        """Test basic argument pass-through to script."""
        # Create script that prints sys.argv
        script_content = """
import sys
import json
print("ARGS:", json.dumps(sys.argv))
"""
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--data-exp",
                    "abc123",
                    "--fold",
                    "0",
                ],
            )

            assert result.exit_code == 0
            # Check that arguments were passed to script
            assert "ARGS:" in result.output
            assert '"--data-exp"' in result.output
            assert '"abc123"' in result.output
            assert '"--fold"' in result.output
            assert '"0"' in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_script_args_with_argparse(self, tmp_path):
        """Test script args work with argparse in script."""
        # Create script that uses argparse
        script_content = """
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--data-exp', required=True, help='Data experiment ID')
parser.add_argument('--fold', type=int, default=0, help='Fold number')
parser.add_argument('--verbose', action='store_true', help='Verbose output')

args = parser.parse_args()

print(f"Data experiment: {args.data_exp}")
print(f"Fold: {args.fold}")
print(f"Verbose: {args.verbose}")
"""
        script_path = tmp_path / "test_argparse.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--data-exp",
                    "xyz789",
                    "--fold",
                    "3",
                    "--verbose",
                ],
            )

            assert result.exit_code == 0
            assert "Data experiment: xyz789" in result.output
            assert "Fold: 3" in result.output
            assert "Verbose: True" in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_script_args_with_yanex_params(self, tmp_path):
        """Test script args work alongside yanex parameters."""
        # Create script that uses both yanex params and script args
        script_content = """
import argparse
import yanex

# Parse script-specific args
parser = argparse.ArgumentParser()
parser.add_argument('--data-exp', required=True)
args = parser.parse_args()

# Get yanex parameters
learning_rate = yanex.get_param('learning_rate', default=0.001)

print(f"Script arg data_exp: {args.data_exp}")
print(f"Yanex param learning_rate: {learning_rate}")
"""
        script_path = tmp_path / "test_mixed.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "-p",
                    "learning_rate=0.01",
                    "--data-exp",
                    "abc123",
                ],
            )

            assert result.exit_code == 0
            assert "Script arg data_exp: abc123" in result.output
            assert "Yanex param learning_rate: 0.01" in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_script_args_stored_in_metadata(self, tmp_path):
        """Test that script args are stored in experiment metadata."""
        from yanex.core.manager import ExperimentManager

        # Create simple script
        script_content = """
import sys
print("Script executed")
"""
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        experiments_dir = tmp_path / "experiments"
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(experiments_dir)

        try:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--name",
                    "test-with-args",
                    "--data-exp",
                    "abc123",
                    "--fold",
                    "0",
                ],
            )

            assert result.exit_code == 0

            # Load metadata and verify script_args are stored
            manager = ExperimentManager(experiments_dir)
            experiments = manager.list_experiments()
            assert len(experiments) == 1

            metadata = manager.get_experiment_metadata(experiments[0])
            assert "script_args" in metadata
            assert metadata["script_args"] == ["--data-exp", "abc123", "--fold", "0"]

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_script_args_with_staged_experiments(self, tmp_path):
        """Test script args work with staged experiments."""

        # Create simple script that prints args
        script_content = """
import sys
import json
print("ARGS:", json.dumps(sys.argv[1:]))  # Skip script name
"""
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        experiments_dir = tmp_path / "experiments"
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(experiments_dir)

        try:
            # Stage experiment with script args
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--stage",
                    "--data-exp",
                    "abc123",
                ],
            )
            assert result.exit_code == 0
            assert "Experiment staged" in result.output

            # Execute staged experiments
            result = self.runner.invoke(cli, ["run", "--staged"])
            assert result.exit_code == 0
            # Verify args were passed through
            assert '"--data-exp"' in result.output
            assert '"abc123"' in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_script_args_with_parameter_sweep(self, tmp_path):
        """Test script args work with parameter sweeps."""
        # Create script that prints both yanex params and script args
        script_content = """
import argparse
import yanex

parser = argparse.ArgumentParser()
parser.add_argument('--data-exp', required=True)
args = parser.parse_args()

lr = yanex.get_param('learning_rate', default=0.001)
print(f"LR={lr} DATA={args.data_exp}")
"""
        script_path = tmp_path / "test_sweep.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        experiments_dir = tmp_path / "experiments"
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(experiments_dir)

        try:
            # Run sweep with script args
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "-p",
                    "learning_rate=list(0.01,0.02)",
                    "--data-exp",
                    "abc123",
                ],
            )

            assert result.exit_code == 0
            # Both sweep values should use the same script args
            assert "LR=0.01 DATA=abc123" in result.output
            assert "LR=0.02 DATA=abc123" in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_empty_script_args(self, tmp_path):
        """Test that experiments work without script args (backward compatibility)."""
        script_content = """
import sys
print("Script executed")
"""
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        experiments_dir = tmp_path / "experiments"
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(experiments_dir)

        try:
            # Run without any script args
            result = self.runner.invoke(
                cli, ["run", str(script_path), "--ignore-dirty"]
            )

            assert result.exit_code == 0
            assert "Script executed" in result.output

            # Verify metadata still has script_args field (empty list)
            from yanex.core.manager import ExperimentManager

            manager = ExperimentManager(experiments_dir)
            experiments = manager.list_experiments()
            metadata = manager.get_experiment_metadata(experiments[0])
            assert "script_args" in metadata
            assert metadata["script_args"] == []

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir

    def test_verbose_mode_shows_script_args(self, tmp_path):
        """Test that verbose mode displays script arguments."""
        script_content = """
print("Script executed")
"""
        script_path = tmp_path / "test_script.py"
        script_path.write_text(script_content)

        # Use isolated temp directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            result = self.runner.invoke(
                cli,
                [
                    "--verbose",
                    "run",
                    str(script_path),
                    "--data-exp",
                    "abc123",
                ],
            )

            assert result.exit_code == 0
            # Verbose mode should show script arguments
            assert "Script arguments:" in result.output
            assert "--data-exp" in result.output
            assert "abc123" in result.output

        finally:
            if old_yanex_dir is None:
                os.environ.pop("YANEX_EXPERIMENTS_DIR", None)
            else:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
