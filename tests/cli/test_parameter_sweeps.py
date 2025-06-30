"""
Tests for CLI parameter sweep functionality.
"""

import tempfile
from pathlib import Path

from click.testing import CliRunner

from yanex.cli.main import cli


class TestCLIParameterSweeps:
    """Test CLI parameter sweep functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_sweep_requires_stage_flag(self):
        """Test that parameter sweeps require --stage flag."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('test')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(
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
        finally:
            script_path.unlink()

    def test_sweep_with_stage_flag_dry_run(self):
        """Test parameter sweep with --stage flag in dry-run mode."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('test')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(
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
        finally:
            script_path.unlink()

    def test_single_parameter_sweep_staging(self):
        """Test staging single parameter sweep."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('sweep test')")
            script_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use isolated experiment directory
                import os
                old_yanex_dir = os.environ.get('YANEX_EXPERIMENTS_DIR')
                os.environ['YANEX_EXPERIMENTS_DIR'] = temp_dir

                try:
                    result = self.runner.invoke(
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
                    assert "Parameter sweep detected: expanding into 2 experiments" in result.output
                    assert "Staged 2 sweep experiments" in result.output
                    assert "sweep-test-sweep-001" in result.output or "IDs:" in result.output
                finally:
                    if old_yanex_dir:
                        os.environ['YANEX_EXPERIMENTS_DIR'] = old_yanex_dir
                    elif 'YANEX_EXPERIMENTS_DIR' in os.environ:
                        del os.environ['YANEX_EXPERIMENTS_DIR']
        finally:
            script_path.unlink()

    def test_multi_parameter_sweep_staging(self):
        """Test staging multi-parameter sweep (cross-product)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('multi-sweep test')")
            script_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use isolated experiment directory
                import os
                old_yanex_dir = os.environ.get('YANEX_EXPERIMENTS_DIR')
                os.environ['YANEX_EXPERIMENTS_DIR'] = temp_dir

                try:
                    result = self.runner.invoke(
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
                    assert "Parameter sweep detected: expanding into 4 experiments" in result.output
                    assert "Staged 4 sweep experiments" in result.output
                finally:
                    if old_yanex_dir:
                        os.environ['YANEX_EXPERIMENTS_DIR'] = old_yanex_dir
                    elif 'YANEX_EXPERIMENTS_DIR' in os.environ:
                        del os.environ['YANEX_EXPERIMENTS_DIR']
        finally:
            script_path.unlink()

    def test_mixed_sweep_and_regular_parameters(self):
        """Test mix of sweep and regular parameters."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('mixed params test')")
            script_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use isolated experiment directory
                import os
                old_yanex_dir = os.environ.get('YANEX_EXPERIMENTS_DIR')
                os.environ['YANEX_EXPERIMENTS_DIR'] = temp_dir

                try:
                    result = self.runner.invoke(
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
                    assert "Parameter sweep detected: expanding into 2 experiments" in result.output
                    assert "Staged 2 sweep experiments" in result.output
                finally:
                    if old_yanex_dir:
                        os.environ['YANEX_EXPERIMENTS_DIR'] = old_yanex_dir
                    elif 'YANEX_EXPERIMENTS_DIR' in os.environ:
                        del os.environ['YANEX_EXPERIMENTS_DIR']
        finally:
            script_path.unlink()

    def test_nested_parameter_sweep(self):
        """Test parameter sweep with nested keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('nested params test')")
            script_path = Path(f.name)

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use isolated experiment directory
                import os
                old_yanex_dir = os.environ.get('YANEX_EXPERIMENTS_DIR')
                os.environ['YANEX_EXPERIMENTS_DIR'] = temp_dir

                try:
                    result = self.runner.invoke(
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
                    assert "Parameter sweep detected: expanding into 2 experiments" in result.output
                    assert "Staged 2 sweep experiments" in result.output
                finally:
                    if old_yanex_dir:
                        os.environ['YANEX_EXPERIMENTS_DIR'] = old_yanex_dir
                    elif 'YANEX_EXPERIMENTS_DIR' in os.environ:
                        del os.environ['YANEX_EXPERIMENTS_DIR']
        finally:
            script_path.unlink()

    def test_regular_parameters_without_sweeps(self):
        """Test that regular parameters work normally without --stage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('regular params test')")
            script_path = Path(f.name)

        try:
            result = self.runner.invoke(
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
        finally:
            script_path.unlink()

    def test_sweep_syntax_validation_errors(self):
        """Test that invalid sweep syntax produces helpful errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('test')")
            script_path = Path(f.name)

        try:
            # Test invalid range syntax
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    str(script_path),
                    "--param",
                    "lr=range(not_a_number, 0.1, 0.01)",
                    "--stage",
                    "--dry-run",
                    "--ignore-dirty",
                ],
            )
            assert result.exit_code != 0
            assert "Invalid range() syntax" in result.output or "Expected numeric value" in result.output

        finally:
            script_path.unlink()

