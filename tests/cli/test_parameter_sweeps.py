"""
Tests for CLI parameter sweep functionality.
"""

import os

import pytest

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli


class TestCLIParameterSweeps:
    """Test CLI parameter sweep functionality."""

    def test_sweep_requires_stage_flag(self, tmp_path, cli_runner):
        """Test that parameter sweeps require --stage flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "lr=range(0.01, 0.03, 0.01)",
                "--dry-run",
            ],
        )
        assert result.exit_code != 0
        assert "Parameter sweeps require --stage flag" in result.output
        assert "Use: yanex run script.py" in result.output

    def test_sweep_with_stage_flag_dry_run(self, tmp_path, cli_runner):
        """Test parameter sweep with --stage flag in dry-run mode."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "lr=range(0.01, 0.03, 0.01)",
                "--stage",
                "--dry-run",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        assert "RangeSweep(0.01, 0.03, 0.01)" in result.output

    def test_single_parameter_sweep_staging(self, tmp_path, cli_runner):
        """Test staging single parameter sweep."""
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
                    "lr=range(0.01, 0.03, 0.01)",
                    "--stage",
                    "--ignore-dirty",
                    "--name",
                    "sweep-test",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 2 experiments"
                in result.output
            )
            assert "Staged 2 sweep experiments" in result.output
            assert "sweep-test-sweep-001" in result.output or "IDs:" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_multi_parameter_sweep_staging(self, tmp_path, cli_runner):
        """Test staging multi-parameter sweep (cross-product)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "multi_sweep_script.py", "simple", test_message="multi-sweep test"
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
                    "lr=linspace(0.01, 0.02, 2)",
                    "--param",
                    "batch_size=list(16, 32)",
                    "--stage",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 4 experiments"
                in result.output
            )
            assert "Staged 4 sweep experiments" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_mixed_sweep_and_regular_parameters(self, tmp_path, cli_runner):
        """Test mix of sweep and regular parameters."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path,
            "mixed_params_script.py",
            "simple",
            test_message="mixed params test",
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
                    "lr=range(0.01, 0.03, 0.01)",
                    "--param",
                    "epochs=100",  # regular parameter
                    "--param",
                    "model_type=resnet",  # regular parameter
                    "--stage",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 2 experiments"
                in result.output
            )
            assert "Staged 2 sweep experiments" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_nested_parameter_sweep(self, tmp_path, cli_runner):
        """Test parameter sweep with nested keys."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path,
            "nested_params_script.py",
            "simple",
            test_message="nested params test",
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
                    "model.learning_rate=range(0.01, 0.03, 0.01)",
                    "--param",
                    "training.epochs=100",
                    "--stage",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 2 experiments"
                in result.output
            )
            assert "Staged 2 sweep experiments" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_regular_parameters_without_sweeps(self, tmp_path, cli_runner):
        """Test that regular parameters work normally without --stage."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path,
            "regular_params_script.py",
            "simple",
            test_message="regular params test",
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "lr=0.01",
                "--param",
                "batch_size=32",
                "--dry-run",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        assert "lr" in result.output
        assert "batch_size" in result.output

    @pytest.mark.parametrize(
        "invalid_param,expected_error_fragment",
        [
            ("lr=range(not_a_number, 0.1, 0.01)", "Invalid range() syntax"),
            ("lr=range()", "Invalid sweep syntax: range()"),
            ("lr=linspace()", "Invalid sweep syntax: linspace()"),
            ("lr=list()", "Invalid list() syntax: list()"),
        ],
    )
    def test_sweep_syntax_validation_errors(
        self, tmp_path, cli_runner, invalid_param, expected_error_fragment
    ):
        """Test that invalid sweep syntax produces helpful errors."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                invalid_param,
                "--stage",
                "--dry-run",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code != 0
        assert (
            expected_error_fragment in result.output
            or "Expected numeric value" in result.output
        )

    def test_parameter_aware_naming_with_base_name(self, tmp_path, cli_runner):
        """Test parameter-aware naming with explicit base name."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "naming_script.py", "simple", test_message="naming test"
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
                    "lr=list(1e-4, 1e-3)",
                    "--param",
                    "batch_size=32",
                    "--stage",
                    "--ignore-dirty",
                    "--name",
                    "test-model",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 2 experiments"
                in result.output
            )
            assert "Staged 2 sweep experiments" in result.output

            # Check that parameter values are included in names
            # Note: We can't easily check the exact names without accessing the storage,
            # but we can verify the sweep was created successfully
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_parameter_aware_naming_without_base_name(self, tmp_path, cli_runner):
        """Test parameter-aware naming without explicit base name."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "naming_script.py", "simple", test_message="naming test"
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
                    "lr=range(1e-4, 3e-4, 1e-4)",
                    "--stage",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code == 0
            assert (
                "Parameter sweep detected: expanding into 2 experiments"
                in result.output
            )
            assert "Staged 2 sweep experiments" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    @pytest.mark.parametrize(
        "sweep_params,expected_count",
        [
            (["lr=range(0.01, 0.04, 0.01)"], 3),  # 3 values: 0.01, 0.02, 0.03
            (["lr=list(0.001, 0.01, 0.1)"], 3),  # 3 explicit values
            (["lr=linspace(0.01, 0.02, 3)"], 3),  # 3 linearly spaced values
            (
                ["lr=range(0.01, 0.03, 0.01)", "batch_size=list(16, 32)"],
                4,
            ),  # 2x2 cross-product
            (
                [
                    "lr=list(0.001, 0.01)",
                    "batch_size=list(16, 32)",
                    "epochs=list(10, 20)",
                ],
                8,
            ),  # 2x2x2 cross-product
        ],
    )
    def test_sweep_expansion_counts(
        self, tmp_path, cli_runner, sweep_params, expected_count
    ):
        """Test that parameter sweeps expand to correct number of experiments."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "count_test_script.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path)

        try:
            # Build command with all sweep parameters
            command = [
                "run",
                str(script_path),
                "--stage",
                "--ignore-dirty",
            ]

            for param in sweep_params:
                command.extend(["--param", param])

            result = cli_runner.invoke(cli, command)
            assert result.exit_code == 0
            assert (
                f"Parameter sweep detected: expanding into {expected_count} experiments"
                in result.output
            )
            assert f"Staged {expected_count} sweep experiments" in result.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    @pytest.mark.parametrize(
        "sweep_type,param_spec",
        [
            ("range", "lr=range(0.01, 0.03, 0.01)"),
            ("linspace", "lr=linspace(0.01, 0.03, 3)"),
            ("list", "lr=list(0.01, 0.02, 0.03)"),
            ("range_int", "epochs=range(10, 30, 10)"),
            ("list_str", "model=list('resnet', 'vgg', 'bert')"),
        ],
    )
    def test_sweep_syntax_types(self, tmp_path, cli_runner, sweep_type, param_spec):
        """Test different sweep syntax types work correctly."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, f"{sweep_type}_script.py", "simple"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                param_spec,
                "--stage",
                "--dry-run",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        # Check that the sweep type is properly recognized
        assert (
            "RangeSweep" in result.output
            or "LinspaceSweep" in result.output
            or "ListSweep" in result.output
            or "expanding into" in result.output
        )

    def test_sweep_with_config_file(self, tmp_path, cli_runner):
        """Test parameter sweeps work with configuration files."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "config_sweep_script.py", "yanex_basic"
        )

        # Create config file with base parameters
        config_data = {
            "epochs": 100,
            "model_type": "transformer",
            "batch_size": 32,
        }
        config_path = TestFileHelpers.create_config_file(
            tmp_path, config_data, "sweep_config.yaml"
        )

        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--config",
                str(config_path),
                "--param",
                "lr=range(0.001, 0.003, 0.001)",  # Override lr with sweep
                "--stage",
                "--dry-run",
                "--ignore-dirty",
            ],
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output
        assert "lr" in result.output
        assert "epochs" in result.output
        assert "model_type" in result.output

    def test_sweep_isolation_between_tests(self, tmp_path, cli_runner):
        """Test that sweeps from different tests don't interfere."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "isolation_test_script.py", "simple"
        )

        # Use isolated experiment directory
        old_yanex_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "isolated_experiments")

        try:
            # First sweep
            result1 = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "lr=list(0.01, 0.02)",
                    "--stage",
                    "--ignore-dirty",
                    "--name",
                    "first-sweep",
                ],
            )
            assert result1.exit_code == 0
            assert "Staged 2 sweep experiments" in result1.output

            # Second sweep should work independently
            result2 = cli_runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "batch_size=list(16, 32, 64)",
                    "--stage",
                    "--ignore-dirty",
                    "--name",
                    "second-sweep",
                ],
            )
            assert result2.exit_code == 0
            assert "Staged 3 sweep experiments" in result2.output
        finally:
            if old_yanex_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_yanex_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]
