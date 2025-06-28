"""
Tests for yanex CLI main entry point.
"""

import tempfile
from pathlib import Path
from click.testing import CliRunner

import pytest

from yanex.cli.main import cli


class TestCLIMain:
    """Test main CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help output."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Yet Another Experiment Tracker" in result.output
        assert "run" in result.output

    def test_cli_version(self):
        """Test CLI version output."""
        result = self.runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_run_command_help(self):
        """Test run command help."""
        result = self.runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a script as a tracked experiment" in result.output
        assert "--config" in result.output
        assert "--param" in result.output

    def test_run_nonexistent_script(self):
        """Test running non-existent script."""
        result = self.runner.invoke(cli, ["run", "nonexistent.py"])
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_run_dry_run_basic(self):
        """Test basic dry run functionality."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello world')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(cli, ["run", str(script_path), "--dry-run"])
            assert result.exit_code == 0
            assert "Configuration validation passed" in result.output
            assert "Config: {}" in result.output
        finally:
            script_path.unlink()

    def test_run_with_params(self):
        """Test run with parameter overrides."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello world')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(cli, [
                "run", str(script_path), 
                "--param", "learning_rate=0.01",
                "--param", "epochs=100",
                "--name", "test-experiment",
                "--tag", "test",
                "--tag", "cli",
                "--description", "Test experiment",
                "--dry-run"
            ])
            assert result.exit_code == 0
            assert "learning_rate" in result.output
            assert "epochs" in result.output
            assert "test-experiment" in result.output
            assert "['test', 'cli']" in result.output
        finally:
            script_path.unlink()

    def test_run_invalid_param_format(self):
        """Test invalid parameter format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello world')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(cli, [
                "run", str(script_path), 
                "--param", "invalid_format_no_equals",
                "--dry-run"
            ])
            assert result.exit_code != 0
            assert "must be in format 'key=value'" in result.output
        finally:
            script_path.unlink()

    def test_run_with_config_file(self):
        """Test run with configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create script
            script_path = Path(temp_dir) / "script.py"
            script_path.write_text("print('hello world')")
            
            # Create config file
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("learning_rate: 0.01\nepochs: 50")
            
            result = self.runner.invoke(cli, [
                "run", str(script_path),
                "--config", str(config_path),
                "--dry-run"
            ])
            assert result.exit_code == 0
            assert "learning_rate" in result.output
            assert "epochs" in result.output

    def test_verbose_mode(self):
        """Test verbose output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello world')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(cli, [
                "--verbose", "run", str(script_path), "--dry-run"
            ])
            assert result.exit_code == 0
            assert f"Running script: {script_path}" in result.output
            assert "No configuration file found" in result.output
        finally:
            script_path.unlink()