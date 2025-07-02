"""
Tests for yanex CLI main entry point.
"""

import pytest

from tests.test_utils import TestFileHelpers, create_cli_runner
from yanex.cli.main import cli


class TestCLIMain:
    """Test main CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

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

    def test_run_dry_run_basic(self, tmp_path):
        """Test basic dry run functionality."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = self.runner.invoke(cli, ["run", str(script_path), "--dry-run"])
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        assert "Config: {}" in result.output

    @pytest.mark.parametrize(
        "params,expected_in_output",
        [
            (
                ["learning_rate=0.01", "epochs=100"],
                ["learning_rate", "epochs", "0.01", "100"],
            ),
            (
                ["batch_size=32", "model=transformer"],
                ["batch_size", "model", "32", "transformer"],
            ),
        ],
    )
    def test_run_with_params(self, tmp_path, params, expected_in_output):
        """Test run with parameter overrides."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        param_args = []
        for param in params:
            param_args.extend(["--param", param])

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                *param_args,
                "--name",
                "test-experiment",
                "--tag",
                "test",
                "--tag",
                "cli",
                "--description",
                "Test experiment",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        for expected in expected_in_output:
            assert str(expected) in result.output
        assert "test-experiment" in result.output
        assert "['test', 'cli']" in result.output

    @pytest.mark.parametrize(
        "invalid_param,expected_error",
        [
            ("invalid_format_no_equals", "Expected 'key=value'"),
            ("=empty_key", "Empty parameter key"),
        ],
    )
    def test_run_invalid_param_format(self, tmp_path, invalid_param, expected_error):
        """Test invalid parameter format."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                invalid_param,
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert expected_error in result.output

    def test_run_with_empty_value_param(self, tmp_path):
        """Test parameter with empty value (should be allowed)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "empty_param=",
                "--dry-run",
            ],
        )

        # Empty values should be allowed
        assert result.exit_code == 0
        assert "empty_param" in result.output

    def test_run_with_config_file(self, tmp_path):
        """Test run with configuration file."""
        # Create script using utility
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "script.py", "simple"
        )

        # Create config file using utility
        config_data = {"learning_rate": 0.01, "epochs": 50}
        config_path = TestFileHelpers.create_config_file(
            tmp_path, config_data, "config.yaml"
        )

        result = self.runner.invoke(
            cli,
            ["run", str(script_path), "--config", str(config_path), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "learning_rate" in result.output
        assert "epochs" in result.output

    def test_verbose_mode(self, tmp_path):
        """Test verbose output."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = self.runner.invoke(
            cli, ["--verbose", "run", str(script_path), "--dry-run"]
        )
        assert result.exit_code == 0
        # Verbose output now goes to stderr due to colored Rich console
        stderr_text = result.stderr_bytes.decode()
        assert "Running script:" in stderr_text
        assert script_path.name in stderr_text  # Just check for the filename
        assert "No configuration file found" in result.output

    def test_run_stage_flag(self, tmp_path):
        """Test run with --stage flag creates staged experiment."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_script.py", "simple", test_message="staged experiment"
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--param",
                "test_param=value",
                "--name",
                "staged-test",
                "--ignore-dirty",  # Allow dirty working directory in tests
            ],
        )
        assert result.exit_code == 0
        assert "Experiment staged:" in result.output
        assert "Directory:" in result.output
        assert "Use 'yanex run --staged' to execute" in result.output

    def test_run_stage_and_staged_flags_conflict(self, tmp_path):
        """Test run with both --stage and --staged flags shows error."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--staged",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot use both --stage and --staged flags" in result.output

    def test_run_staged_flag_no_script(self):
        """Test run with --staged flag doesn't require script argument."""
        result = self.runner.invoke(cli, ["run", "--staged"])
        # Should succeed - either find no staged experiments or successfully execute any that exist
        assert result.exit_code == 0
        # Check for expected messages (could be no staged experiments or successful execution)
        assert (
            "No staged experiments found" in result.output
            or "Experiment completed successfully" in result.output
            or "Experiment failed" in result.output
        )

    def test_run_staged_flag_with_script_isolation(self, tmp_path):
        """Test run with --staged flag with isolated experiment directory."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        import os

        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # This should work - script argument is optional when using --staged
            result = self.runner.invoke(cli, ["run", str(script_path), "--staged"])
            assert result.exit_code == 0
            assert "No staged experiments found" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_run_missing_script_without_staged(self):
        """Test run without script and without --staged shows error."""
        result = self.runner.invoke(cli, ["run"])
        assert result.exit_code != 0
        assert "Missing argument 'SCRIPT'" in result.output

    def test_stage_experiment_dry_run(self, tmp_path):
        """Test staging experiment with dry-run flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "dry_run_script.py", "simple", test_message="dry run test"
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--stage",
                "--dry-run",
                "--param",
                "learning_rate=0.01",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        assert "Dry run completed" in result.output
        # Should not create staged experiment in dry-run mode
        assert "Experiment staged:" not in result.output

    @pytest.mark.parametrize(
        "script_type,expected_output",
        [
            ("simple", "Hello from test script"),
            ("yanex_basic", "Running with params"),
            ("yanex_ml", "Training completed"),
        ],
    )
    def test_script_templates(self, tmp_path, script_type, expected_output):
        """Test different script templates work correctly in dry-run mode."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, f"{script_type}_script.py", script_type
        )

        result = self.runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--dry-run",
                "--param",
                "learning_rate=0.001",
                "--param",
                "epochs=5",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output

    def test_config_file_formats(self, tmp_path):
        """Test different configuration file formats."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "script.py", "simple"
        )

        # Test JSON config
        json_config = {"learning_rate": 0.01, "epochs": 10}
        json_config_path = TestFileHelpers.create_config_file(
            tmp_path, json_config, "config.json"
        )

        result = self.runner.invoke(
            cli,
            ["run", str(script_path), "--config", str(json_config_path), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "learning_rate" in result.output
        assert "epochs" in result.output

        # Test YAML config
        yaml_config = {"batch_size": 32, "model_type": "transformer"}
        yaml_config_path = TestFileHelpers.create_config_file(
            tmp_path, yaml_config, "config.yaml"
        )

        result = self.runner.invoke(
            cli,
            ["run", str(script_path), "--config", str(yaml_config_path), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "batch_size" in result.output
        assert "model_type" in result.output
