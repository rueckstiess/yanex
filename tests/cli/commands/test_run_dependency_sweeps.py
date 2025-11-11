"""Tests for dependency sweep functionality in run command."""



from tests.test_utils import create_cli_runner
from yanex.cli.commands.run import (
    generate_dependency_combinations,
    parse_dependency_args,
)
from yanex.cli.main import cli


class TestDependencyCombinationsHelper:
    """Test the generate_dependency_combinations helper function."""

    def test_empty_dependencies(self):
        """Test with empty dependencies dict."""
        result = generate_dependency_combinations({})
        assert result == [{}]

    def test_single_slot_single_id(self):
        """Test single slot with single ID."""
        parsed = {"dataprep": ["dp1"]}
        result = generate_dependency_combinations(parsed)
        assert result == [{"dataprep": "dp1"}]

    def test_single_slot_multiple_ids(self):
        """Test single slot with multiple IDs."""
        parsed = {"training": ["tr1", "tr2", "tr3"]}
        result = generate_dependency_combinations(parsed)
        assert len(result) == 3
        assert result == [
            {"training": "tr1"},
            {"training": "tr2"},
            {"training": "tr3"},
        ]

    def test_multiple_slots_single_id_each(self):
        """Test multiple slots, each with single ID."""
        parsed = {"dataprep": ["dp1"], "training": ["tr1"]}
        result = generate_dependency_combinations(parsed)
        assert result == [{"dataprep": "dp1", "training": "tr1"}]

    def test_cartesian_product_2x2(self):
        """Test cartesian product: 2 slots × 2 IDs each."""
        parsed = {"model1": ["tr1", "tr2"], "model2": ["tr3", "tr4"]}
        result = generate_dependency_combinations(parsed)
        assert len(result) == 4
        assert result == [
            {"model1": "tr1", "model2": "tr3"},
            {"model1": "tr1", "model2": "tr4"},
            {"model1": "tr2", "model2": "tr3"},
            {"model1": "tr2", "model2": "tr4"},
        ]

    def test_cartesian_product_3x2(self):
        """Test cartesian product: 3 IDs × 2 IDs."""
        parsed = {"dataprep": ["dp1"], "training": ["tr1", "tr2", "tr3"]}
        result = generate_dependency_combinations(parsed)
        assert len(result) == 3
        assert result == [
            {"dataprep": "dp1", "training": "tr1"},
            {"dataprep": "dp1", "training": "tr2"},
            {"dataprep": "dp1", "training": "tr3"},
        ]

    def test_cartesian_product_2x3x2(self):
        """Test cartesian product: 2 × 3 × 2 = 12 combinations."""
        parsed = {
            "dataprep": ["dp1", "dp2"],
            "training": ["tr1", "tr2", "tr3"],
            "validation": ["val1", "val2"],
        }
        result = generate_dependency_combinations(parsed)
        assert len(result) == 12  # 2 * 3 * 2

        # Check first and last combinations
        assert result[0] == {
            "dataprep": "dp1",
            "training": "tr1",
            "validation": "val1",
        }
        assert result[-1] == {
            "dataprep": "dp2",
            "training": "tr3",
            "validation": "val2",
        }


class TestParseDependencyArgs:
    """Test dependency argument parsing."""

    def test_single_dependency(self):
        """Test parsing single dependency."""
        result = parse_dependency_args(["dataprep=abc12345"])
        assert result == {"dataprep": ["abc12345"]}

    def test_multiple_dependencies_different_slots(self):
        """Test parsing multiple dependencies with different slots."""
        result = parse_dependency_args(["dataprep=abc12345", "training=def67890"])
        assert result == {"dataprep": ["abc12345"], "training": ["def67890"]}

    def test_comma_separated_ids(self):
        """Test parsing comma-separated IDs."""
        result = parse_dependency_args(["training=abc12345,def67890,12345678"])
        assert result == {"training": ["abc12345", "def67890", "12345678"]}

    def test_multiple_slots_with_comma_separated(self):
        """Test multiple slots, some with comma-separated IDs."""
        result = parse_dependency_args(
            ["dataprep=abc12345", "training=def67890,12345678,abcdef12"]
        )
        assert result == {
            "dataprep": ["abc12345"],
            "training": ["def67890", "12345678", "abcdef12"],
        }

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        result = parse_dependency_args(
            ["training = abc12345 , def67890 , 12345678 "]
        )
        assert result == {"training": ["abc12345", "def67890", "12345678"]}


class TestDependencySweepExecution:
    """Test actual execution of dependency sweeps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def _get_latest_experiment_id(self):
        """Get the ID of the most recently created experiment."""
        from yanex.core.manager import ExperimentManager

        manager = ExperimentManager()
        experiment_ids = manager.list_experiments()
        if experiment_ids:
            # Returns most recent first
            return experiment_ids[0]
        return None

    def test_single_dependency_sweep(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test dependency sweep with single slot, multiple IDs."""
        # Create 3 dependency experiments
        result1 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "dep1"]
        )
        assert result1.exit_code == 0
        dep_id1 = self._get_latest_experiment_id()

        result2 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "dep2"]
        )
        assert result2.exit_code == 0
        dep_id2 = self._get_latest_experiment_id()

        result3 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "dep3"]
        )
        assert result3.exit_code == 0
        dep_id3 = self._get_latest_experiment_id()

        # Create a script that declares dependency
        eval_script = temp_dir / "evaluate.py"
        eval_script.write_text(
            """
import yanex
params = yanex.get_params()
print(f"Evaluating with params: {params}")
"""
        )

        # Create config that declares the dependency
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: evaluate.py
      dependencies:
        training: evaluate.py
"""
        )

        # Run with dependency sweep
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(eval_script),
                "--config",
                str(config_file),
                "-d",
                f"training={dep_id1},{dep_id2},{dep_id3}",
            ],
        )

        # Should succeed and create 3 experiments
        assert result.exit_code == 0
        assert "Dependency sweep detected: running 3 experiments" in result.output
        assert "Completed: 3/3" in result.output

    def test_dependency_cartesian_product(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test dependency sweep with cartesian product (2 slots)."""
        # Create 2 experiments for slot1
        result1 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "m1"]
        )
        assert result1.exit_code == 0
        m1_id = self._get_latest_experiment_id()

        result2 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "m2"]
        )
        assert result2.exit_code == 0
        m2_id = self._get_latest_experiment_id()

        # Create 2 experiments for slot2
        result3 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "m3"]
        )
        assert result3.exit_code == 0
        m3_id = self._get_latest_experiment_id()

        result4 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "m4"]
        )
        assert result4.exit_code == 0
        m4_id = self._get_latest_experiment_id()

        # Create comparison script
        compare_script = temp_dir / "compare.py"
        compare_script.write_text(
            """
import yanex
print("Comparing models")
"""
        )

        # Create config
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: compare.py
      dependencies:
        model1: compare.py
        model2: compare.py
"""
        )

        # Run with 2×2 dependency sweep (should create 4 experiments)
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(compare_script),
                "--config",
                str(config_file),
                "-d",
                f"model1={m1_id},{m2_id}",
                "-d",
                f"model2={m3_id},{m4_id}",
            ],
        )

        # Should succeed and create 4 experiments
        assert result.exit_code == 0
        assert "Dependency sweep detected: running 4 experiments" in result.output
        assert "Completed: 4/4" in result.output

    def test_dependency_sweep_with_parallel(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test dependency sweep with parallel execution."""
        # Create 3 dependency experiments
        dep_ids = []
        for i in range(3):
            result = self.runner.invoke(
                cli, ["run", str(sample_experiment_script), "--name", f"dep{i}"]
            )
            assert result.exit_code == 0
            dep_id = self._get_latest_experiment_id()
            dep_ids.append(dep_id)

        # Create eval script
        eval_script = temp_dir / "evaluate.py"
        eval_script.write_text(
            """
import yanex
print("Evaluating")
"""
        )

        # Create config
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: evaluate.py
      dependencies:
        training: evaluate.py
"""
        )

        # Run with dependency sweep in parallel
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(eval_script),
                "--config",
                str(config_file),
                "-d",
                f"training={','.join(dep_ids)}",
                "--parallel",
                "2",
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        assert "Dependency sweep detected: running 3 experiments" in result.output
        assert "Completed: 3/3" in result.output

    def test_combined_parameter_and_dependency_sweep(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test combined parameter and dependency sweeps (cartesian product)."""
        # Create 2 dependency experiments
        result1 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train1"]
        )
        assert result1.exit_code == 0
        tr1_id = self._get_latest_experiment_id()

        result2 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "train2"]
        )
        assert result2.exit_code == 0
        tr2_id = self._get_latest_experiment_id()

        # Create analysis script
        analyze_script = temp_dir / "analyze.py"
        analyze_script.write_text(
            """
import yanex
threshold = yanex.get_param('threshold')
print(f"Analyzing with threshold={threshold}")
"""
        )

        # Create config
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: analyze.py
      dependencies:
        training: analyze.py
"""
        )

        # Run with both parameter sweep (3 values) and dependency sweep (2 deps)
        # Should create 3 × 2 = 6 experiments
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(analyze_script),
                "--config",
                str(config_file),
                "--param",
                "threshold=list(0.3, 0.5, 0.7)",
                "-d",
                f"training={tr1_id},{tr2_id}",
            ],
        )

        # Should succeed and create 6 experiments
        assert result.exit_code == 0
        assert "Combined sweep detected: running 6 experiments" in result.output
        assert "(3 parameter configs × 2 dependency combos)" in result.output
        assert "Completed: 6/6" in result.output


class TestDependencySweepValidation:
    """Test validation and error handling for dependency sweeps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_dependency_sweep_with_nonexistent_id(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test that nonexistent dependency IDs are caught."""
        # Create eval script
        eval_script = temp_dir / "evaluate.py"
        eval_script.write_text(
            """
import yanex
print("Evaluating")
"""
        )

        # Create config
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: evaluate.py
      dependencies:
        training: evaluate.py
"""
        )

        # Run with nonexistent dependency ID
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(eval_script),
                "--config",
                str(config_file),
                "-d",
                "training=nonexist1,nonexist2",
            ],
        )

        # Should fail during first experiment creation (dependency validation)
        assert result.exit_code != 0
        # The error will come from the executor when it tries to validate dependencies
        assert "Failed" in result.output or "Error" in result.output

    def test_dependency_sweep_tags_experiments_as_sweep(
        self, clean_git_repo, sample_experiment_script, temp_dir
    ):
        """Test that dependency sweeps automatically add 'sweep' tag."""
        # Create 2 dependency experiments
        result1 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "dep1"]
        )
        assert result1.exit_code == 0
        dep_id1 = self._get_latest_experiment_id()

        result2 = self.runner.invoke(
            cli, ["run", str(sample_experiment_script), "--name", "dep2"]
        )
        assert result2.exit_code == 0
        dep_id2 = self._get_latest_experiment_id()

        # Create eval script
        eval_script = temp_dir / "evaluate.py"
        eval_script.write_text(
            """
import yanex
print("Evaluating")
"""
        )

        # Create config
        config_file = temp_dir / "config.yaml"
        config_file.write_text(
            """
yanex:
  scripts:
    - name: evaluate.py
      dependencies:
        training: evaluate.py
"""
        )

        # Run dependency sweep
        result = self.runner.invoke(
            cli,
            [
                "run",
                str(eval_script),
                "--config",
                str(config_file),
                "-d",
                f"training={dep_id1},{dep_id2}",
            ],
        )

        assert result.exit_code == 0

        # Check that created experiments have 'sweep' tag
        list_result = self.runner.invoke(cli, ["list", "--tag", "sweep"])
        assert list_result.exit_code == 0
        # Should show 2 sweep experiments
        lines = [line for line in list_result.output.split("\n") if line.strip()]
        # Filter out header and summary lines
        experiment_lines = [
            line for line in lines if not line.startswith("─") and "ID" not in line
        ]
        # Should have at least 2 sweep experiments (could be more if other tests ran)
        assert len(experiment_lines) >= 2
