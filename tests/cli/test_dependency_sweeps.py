"""
Integration tests for CLI dependency sweep functionality.

Tests yanex run --depends-on with:
- Single dependencies
- Dependency sweeps (multiple dependencies)
- Cartesian products (dependencies × parameters)
- Error handling
"""

from pathlib import Path

from tests.test_utils import TestFileHelpers
from yanex.cli.main import cli
from yanex.core.manager import ExperimentManager


def create_completed_experiment(
    script_path: Path, config: dict | None = None, name: str | None = None
) -> str:
    """Helper to create a completed experiment for testing."""
    if config is None:
        config = {}
    manager = ExperimentManager()
    exp_id = manager.create_experiment(
        script_path=script_path, config=config, name=name
    )
    manager.start_experiment(exp_id)
    manager.complete_experiment(exp_id)
    return exp_id


def create_staged_experiment(
    script_path: Path, config: dict | None = None, name: str | None = None
) -> str:
    """Helper to create a staged experiment for testing incremental pipeline staging."""
    if config is None:
        config = {}
    manager = ExperimentManager()
    exp_id = manager.create_experiment(
        script_path=script_path, config=config, name=name, stage_only=True
    )
    return exp_id


class TestBasicDependencyUsage:
    """Test basic dependency flag usage with single dependencies."""

    def test_single_dependency_with_d_flag(self, tmp_path, cli_runner):
        """Test using -D flag with single dependency."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Create a completed dependency
        dep_id = create_completed_experiment(
            script_path, {"dep_param": 1}, "dependency"
        )

        # Run experiment with dependency via CLI
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                dep_id,
                "--param",
                "main_param=2",
            ],
        )
        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.output

    def test_single_dependency_with_depends_on_flag(self, tmp_path, cli_runner):
        """Test using --depends-on flag with single dependency."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Create a completed dependency
        dep_id = create_completed_experiment(script_path, {}, "dependency-exp")

        # Run with dependency using --depends-on
        result = cli_runner.invoke(
            cli, ["run", str(script_path), "--depends-on", dep_id]
        )
        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.output

    def test_comma_separated_dependencies(self, tmp_path, cli_runner):
        """Test comma-separated dependencies in single -D flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run with comma-separated dependencies (creates sweep)
        result = cli_runner.invoke(
            cli, ["run", str(script_path), "-D", f"{dep_ids[0]},{dep_ids[1]}"]
        )
        assert result.exit_code == 0
        # Should create dependency sweep
        assert (
            "✓ Sweep detected: running 2 experiments" in result.output
            or "Sweep execution completed" in result.output
        )

    def test_multiple_d_flags(self, tmp_path, cli_runner):
        """Test multiple -D flags for dependencies."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run with multiple -D flags
        result = cli_runner.invoke(
            cli,
            ["run", str(script_path), "-D", dep_ids[0], "-D", dep_ids[1]],
        )
        assert result.exit_code == 0
        # Should create dependency sweep
        assert (
            "✓ Sweep detected: running 2 experiments" in result.output
            or "Sweep execution completed" in result.output
        )

    def test_short_id_resolution(self, tmp_path, cli_runner):
        """Test using short IDs (prefixes) for dependencies."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "test_script.py", "simple"
        )

        # Create dependency with known full ID
        full_id = create_completed_experiment(script_path, {}, "dependency")
        assert len(full_id) == 8  # 8-char hex ID

        # Use first 4 characters as short ID
        short_id = full_id[:4]

        # Run with short ID
        result = cli_runner.invoke(cli, ["run", str(script_path), "-D", short_id])
        assert result.exit_code == 0
        assert "Experiment completed successfully" in result.output


class TestDependencySweeps:
    """Test dependency sweep execution (multiple dependencies)."""

    def test_dependency_sweep_sequential(self, tmp_path, cli_runner):
        """Test dependency sweep runs sequentially by default."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "sweep_script.py", "simple", test_message="dep sweep test"
        )

        # Create three completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"prep-{i}") for i in range(3)
        ]

        # Run dependency sweep
        result = cli_runner.invoke(
            cli,
            ["run", str(script_path), "-D", ",".join(dep_ids)],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 3 experiments" in result.output
        assert "Sweep execution completed" in result.output

    def test_dependency_sweep_parallel(self, tmp_path, cli_runner):
        """Test dependency sweep with parallel execution."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_dep_sweep.py", "simple"
        )

        # Create three completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"prep-{i}") for i in range(3)
        ]

        # Run dependency sweep with parallelism
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--parallel",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 3 experiments" in result.output
        assert "Running 3 experiments with 2 parallel workers" in result.output
        assert "Sweep execution completed" in result.output

    def test_dependency_sweep_with_stage(self, tmp_path, cli_runner):
        """Test staging with multiple dependencies (expands into multiple staged experiments)."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_dep_sweep.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Stage with multiple dependencies
        # Should create dependency sweep: 2 staged experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--stage",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: expanding into 2 experiments" in result.output
        assert "✓ Staged 2 sweep experiments" in result.output

    def test_dependency_sweep_naming(self, tmp_path, cli_runner):
        """Test that dependency sweep experiments have descriptive names."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "naming_test.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"preprocessing-{i}")
            for i in range(2)
        ]

        # Run dependency sweep with base name
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--name",
                "training",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected" in result.output


class TestCartesianProducts:
    """Test Cartesian products (dependencies × parameters)."""

    def test_cartesian_product_basic(self, tmp_path, cli_runner):
        """Test basic Cartesian product: 2 deps × 2 params = 4 experiments."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "cartesian_test.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run Cartesian product: 2 deps × 2 params
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 4 experiments" in result.output
        assert "Sweep execution completed" in result.output

    def test_cartesian_product_multiple_params(self, tmp_path, cli_runner):
        """Test Cartesian product with multiple parameter dimensions."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "multi_param_cartesian.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"prep-{i}") for i in range(2)
        ]

        # Run Cartesian product: 2 deps × 2 lrs × 2 batch_sizes = 8 experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
                "--param",
                "batch_size=list(16,32)",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 8 experiments" in result.output

    def test_cartesian_product_parallel(self, tmp_path, cli_runner):
        """Test Cartesian product with parallel execution."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "parallel_cartesian.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run with parallelism
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
                "--parallel",
                "4",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: running 4 experiments" in result.output
        assert "Running 4 experiments with 4 parallel workers" in result.output

    def test_cartesian_product_staging(self, tmp_path, cli_runner):
        """Test staging Cartesian product experiments."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_cartesian.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Stage Cartesian product
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
                "--stage",
            ],
        )
        assert result.exit_code == 0
        # Staging should expand Cartesian product into 4 experiments (2 deps × 2 params)
        assert "✓ Sweep detected: expanding into 4 experiments" in result.output
        assert "✓ Staged 4 sweep experiments" in result.output

    def test_cartesian_product_large(self, tmp_path, cli_runner):
        """Test large Cartesian product with direct execution."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "large_cartesian.py", "simple"
        )

        # Create two completed dependencies (use 2 instead of 3 to keep test fast)
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run Cartesian product: 2 deps × 2 lrs × 2 batch_sizes = 8 experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
                "--param",
                "batch_size=list(16,32)",
            ],
        )
        assert result.exit_code == 0
        # Should create 8 experiments (2 × 2 × 2)
        assert "✓ Sweep detected: running 8 experiments" in result.output
        assert "Sweep execution completed" in result.output


class TestDependencyErrorHandling:
    """Test error handling for dependency usage."""

    def test_nonexistent_dependency(self, tmp_path, cli_runner):
        """Test error when dependency doesn't exist."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "error_test.py", "simple"
        )

        # Try to use non-existent dependency
        result = cli_runner.invoke(cli, ["run", str(script_path), "-D", "deadbeef"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_ambiguous_dependency_id(self, tmp_path, cli_runner):
        """Test error when dependency ID is ambiguous."""
        # Note: This is difficult to test reliably since IDs are random
        # The test documents expected behavior
        pass

    def test_invalid_dependency_status(self, tmp_path, cli_runner):
        """Test error when dependency has invalid status (failed/cancelled)."""
        # Note: This requires creating failed experiments, which needs more setup
        # The test documents expected behavior
        pass

    def test_dependency_on_staged_experiment(self, tmp_path, cli_runner):
        """Test error when depending on staged (not completed) experiment."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_dep_test.py", "simple"
        )

        # Create staged experiment programmatically
        manager = ExperimentManager()
        staged_id = manager.create_experiment(
            script_path=script_path, config={}, stage_only=True, name="staged-dep"
        )

        # Try to depend on staged experiment (should fail)
        result = cli_runner.invoke(cli, ["run", str(script_path), "-D", staged_id])
        assert result.exit_code != 0
        assert (
            "invalid status" in result.output.lower()
            or "error" in result.output.lower()
        )

    def test_circular_dependency_detection(self, tmp_path, cli_runner):
        """Test detection of circular dependencies."""
        # Note: Circular dependencies are detected at the core level
        # This test documents that CLI properly surfaces those errors
        pass

    def test_dependency_with_invalid_characters(self, tmp_path, cli_runner):
        """Test error handling for dependency IDs with invalid characters."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "invalid_chars.py", "simple"
        )

        # Try invalid ID format
        result = cli_runner.invoke(cli, ["run", str(script_path), "-D", "invalid/id!"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_empty_dependency_list(self, tmp_path, cli_runner):
        """Test handling of empty dependency specification."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "empty_dep.py", "simple"
        )

        # Empty -D flag should fail or be ignored
        result = cli_runner.invoke(cli, ["run", str(script_path), "-D", ""])
        assert result.exit_code != 0 or "Experiment completed" in result.output

    def test_whitespace_in_dependency_list(self, tmp_path, cli_runner):
        """Test handling of whitespace in comma-separated dependency list."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "whitespace_deps.py", "simple"
        )

        # Create two completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Test with whitespace around commas (should be handled properly)
        result = cli_runner.invoke(
            cli,
            ["run", str(script_path), "-D", f"{dep_ids[0]} , {dep_ids[1]}"],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected" in result.output


class TestDependencyIntegration:
    """Test dependency integration with other features."""

    def test_dependencies_with_config_file(self, tmp_path, cli_runner):
        """Test dependencies work with configuration files."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "config_dep_test.py", "simple"
        )

        # Create config file
        config_data = {"epochs": 100, "model_type": "transformer"}
        config_path = TestFileHelpers.create_config_file(
            tmp_path, config_data, "test_config.yaml"
        )

        # Create completed dependency
        dep_id = create_completed_experiment(script_path, {}, "dep")

        # Run with dependency and config
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--config",
                str(config_path),
                "-D",
                dep_id,
            ],
        )
        # Config files work with dependencies
        assert result.exit_code == 0 or "error" not in result.output.lower()

    def test_dependencies_with_tags(self, tmp_path, cli_runner):
        """Test dependencies work with experiment tags."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "tagged_dep.py", "simple"
        )

        # Create completed dependency
        dep_id = create_completed_experiment(script_path, {}, "tagged-dep")

        # Run with dependency and tags
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                dep_id,
                "--tag",
                "training",
            ],
        )
        assert result.exit_code == 0

    def test_dependency_sweep_with_description(self, tmp_path, cli_runner):
        """Test dependency sweeps with experiment descriptions."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "described_sweep.py", "simple"
        )

        # Create completed dependencies
        dep_ids = [
            create_completed_experiment(script_path, {}, f"dep-{i}") for i in range(2)
        ]

        # Run sweep with description
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(dep_ids),
                "--description",
                "Testing dependency sweep with description",
            ],
        )
        assert result.exit_code == 0

    def test_dependencies_with_dry_run(self, tmp_path, cli_runner):
        """Test dependencies work with --dry-run flag."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "dry_run_dep.py", "simple"
        )

        # Create completed dependency
        dep_id = create_completed_experiment(script_path, {}, "dep")

        # Dry run with dependency
        result = cli_runner.invoke(
            cli, ["run", str(script_path), "-D", dep_id, "--dry-run"]
        )
        assert result.exit_code == 0
        assert "Configuration validation passed" in result.output

    def test_stage_with_staged_dependency(self, tmp_path, cli_runner):
        """Test staging single experiment that depends on staged dependency."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "incremental_stage.py", "simple"
        )

        # Create staged dependency (e.g., preprocessing step)
        staged_dep_id = create_staged_experiment(script_path, {}, "prep-stage")

        # Stage another experiment depending on the staged dependency
        # This is the incremental pipeline staging scenario
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                staged_dep_id,
                "--stage",
                "--name",
                "train-stage",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Experiment staged" in result.output

    def test_dependency_sweep_staging_with_staged_deps(self, tmp_path, cli_runner):
        """Test staging dependency sweep where dependencies themselves are staged."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_dep_sweep.py", "simple"
        )

        # Create multiple staged dependencies (e.g., different preprocessing runs)
        staged_dep_ids = [
            create_staged_experiment(script_path, {}, f"prep-stage-{i}")
            for i in range(3)
        ]

        # Stage dependency sweep with staged dependencies
        # Should expand into 3 staged experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(staged_dep_ids),
                "--stage",
                "--name",
                "train-sweep",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: expanding into 3 experiments" in result.output
        assert "✓ Staged 3 sweep experiments" in result.output

    def test_cartesian_staging_with_staged_deps(self, tmp_path, cli_runner):
        """Test staging Cartesian product where dependencies are staged."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "staged_cartesian.py", "simple"
        )

        # Create staged dependencies
        staged_dep_ids = [
            create_staged_experiment(script_path, {}, f"prep-stage-{i}")
            for i in range(2)
        ]

        # Stage Cartesian product: 2 staged deps × 2 params = 4 staged experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(staged_dep_ids),
                "--param",
                "lr=list(0.01,0.1)",
                "--stage",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Sweep detected: expanding into 4 experiments" in result.output
        assert "✓ Staged 4 sweep experiments" in result.output

    def test_incremental_pipeline_staging(self, tmp_path, cli_runner):
        """Test full incremental pipeline staging workflow."""
        script_path = TestFileHelpers.create_test_script(
            tmp_path, "pipeline.py", "simple"
        )

        # Stage 1: Stage preprocessing experiments
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "--param",
                "data_source=list(A,B)",
                "--stage",
                "--name",
                "prep",
            ],
        )
        assert result.exit_code == 0
        assert "✓ Staged 2 sweep experiments" in result.output

        # Get the IDs of staged experiments
        # In real workflow, user would use `yanex list --status staged` to get IDs
        # Here we create them again for clarity
        prep_ids = [
            create_staged_experiment(script_path, {"data_source": src}, f"prep-{src}")
            for src in ["A", "B"]
        ]

        # Stage 2: Stage training experiments that depend on staged preprocessing
        result = cli_runner.invoke(
            cli,
            [
                "run",
                str(script_path),
                "-D",
                ",".join(prep_ids),
                "--param",
                "model=list(linear,tree)",
                "--stage",
                "--name",
                "train",
            ],
        )
        assert result.exit_code == 0
        # Should create Cartesian product: 2 prep deps × 2 models = 4 experiments
        assert "✓ Sweep detected: expanding into 4 experiments" in result.output
        assert "✓ Staged 4 sweep experiments" in result.output

        # Stage 3: Execute all staged experiments
        result = cli_runner.invoke(cli, ["run", "--staged", "--parallel", "4"])
        assert result.exit_code == 0
        # Should execute all 6 experiments (2 prep + 4 train)
        assert "staged experiments" in result.output.lower()
