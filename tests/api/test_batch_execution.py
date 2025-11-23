"""
Tests for batch experiment execution API (yanex.run_multiple).

Tests the programmatic API for running multiple experiments sequentially
or in parallel, including error handling, validation, and context checks.
"""

import os

import pytest

import yanex
from yanex.core.manager import ExperimentManager
from yanex.executor import ExperimentResult, ExperimentSpec


class TestBatchExecutionAPI:
    """Test yanex.run_multiple() batch execution API."""

    def test_run_multiple_sequential(self, per_test_experiments_dir, tmp_path):
        """Test sequential execution of multiple experiments."""
        # Create simple script
        script_content = """
import yanex
print(f"Executed with lr={yanex.get_param('learning_rate')}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)

        # Create experiment specs
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.01},
                name="exp-1",
                tags=["test"],
            ),
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.02},
                name="exp-2",
                tags=["test"],
            ),
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.03},
                name="exp-3",
                tags=["test"],
            ),
        ]

        # Run sequentially (parallel=None)
        results = yanex.run_multiple(experiments, parallel=None)

        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, ExperimentResult) for r in results)
        assert all(r.status == "completed" for r in results)
        assert all(r.experiment_id != "unknown" for r in results)
        assert all(len(r.experiment_id) == 8 for r in results)  # 8-char hex IDs
        assert [r.name for r in results] == ["exp-1", "exp-2", "exp-3"]

        # Verify experiments were created
        manager = ExperimentManager(per_test_experiments_dir)
        all_experiments = manager.list_experiments()
        assert len(all_experiments) == 3

        # Verify metadata and config
        for result, expected_lr in zip(results, [0.01, 0.02, 0.03], strict=True):
            metadata = manager.get_experiment_metadata(result.experiment_id)
            assert metadata["status"] == "completed"

            # Load config separately (stored in params.yaml)
            config = manager.storage.load_config(result.experiment_id)
            assert "learning_rate" in config
            assert config["learning_rate"] == expected_lr

    def test_run_multiple_parallel(self, per_test_experiments_dir, tmp_path):
        """Test parallel execution with ProcessPoolExecutor."""
        # Create script that takes a moment
        script_content = """
import time
import yanex
lr = yanex.get_param('learning_rate')
print(f"Starting with lr={lr}")
time.sleep(0.1)  # Brief delay to ensure parallelism
print(f"Finished with lr={lr}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Create 4 experiments
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": i * 0.01},
                name=f"parallel-{i}",
            )
            for i in range(1, 5)
        ]

        # Run with 2 parallel workers
        results = yanex.run_multiple(experiments, parallel=2, verbose=False)

        # Verify results
        assert len(results) == 4
        assert all(r.status == "completed" for r in results)
        assert all(r.experiment_id != "unknown" for r in results)

        # Verify all experiments were created
        manager = ExperimentManager(per_test_experiments_dir)
        all_experiments = manager.list_experiments()
        assert len(all_experiments) == 4

    def test_parallel_auto_detect_cpus(self, per_test_experiments_dir, tmp_path):
        """Test parallel=0 auto-detects CPU count."""
        script_content = """
print("Executed")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"param": i},
                name=f"exp-{i}",
            )
            for i in range(2)
        ]

        # Run with parallel=0 (auto-detect)
        results = yanex.run_multiple(experiments, parallel=0)

        # Should complete successfully
        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

    def test_error_handling_continues_batch(self, per_test_experiments_dir, tmp_path):
        """Test that individual experiment failures don't abort entire batch."""
        # Create script that fails on certain values
        script_content = """
import yanex
value = yanex.get_param('value')
if value == 2:
    raise ValueError("Intentional failure for value=2")
print(f"Success with value={value}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Create 4 experiments where one will fail
        experiments = [
            ExperimentSpec(
                script_path=script_path, config={"value": i}, name=f"exp-{i}"
            )
            for i in range(1, 5)
        ]

        # Run sequentially
        results = yanex.run_multiple(experiments, parallel=None)

        # Verify results
        assert len(results) == 4

        # Experiment 2 should fail (value=2)
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]

        assert len(completed) == 3
        assert len(failed) == 1

        failed_exp = failed[0]
        assert failed_exp.name == "exp-2"
        assert failed_exp.error_message is not None
        assert "Intentional failure" in failed_exp.error_message

        # Verify successful experiments have valid IDs
        for r in completed:
            assert r.experiment_id != "unknown"
            assert len(r.experiment_id) == 8

    def test_experiment_spec_validation(self, per_test_experiments_dir, tmp_path):
        """Test ExperimentSpec validation logic."""
        # Test: neither script_path nor function specified
        spec1 = ExperimentSpec()
        with pytest.raises(ValueError, match="Must specify exactly one"):
            spec1.validate()

        # Test: both script_path and function specified
        script_path = tmp_path / "script.py"
        script_path.write_text("print('test')")

        spec2 = ExperimentSpec(script_path=script_path, function=lambda: None)
        with pytest.raises(ValueError, match="Must specify exactly one"):
            spec2.validate()

        # Test: function specified (not yet supported)
        spec3 = ExperimentSpec(function=lambda: None)
        with pytest.raises(NotImplementedError, match="Inline function execution"):
            spec3.validate()

        # Test: valid spec with script_path
        spec4 = ExperimentSpec(script_path=script_path)
        spec4.validate()  # Should not raise

    def test_run_multiple_validation_errors(self, per_test_experiments_dir, tmp_path):
        """Test run_multiple validates inputs."""
        # Test: empty experiments list
        with pytest.raises(ValueError, match="experiments list cannot be empty"):
            yanex.run_multiple([])

        # Test: invalid spec in list
        script_path = tmp_path / "script.py"
        script_path.write_text("print('test')")

        invalid_experiments = [
            ExperimentSpec(script_path=script_path, config={"a": 1}),
            ExperimentSpec(),  # Invalid: no script_path or function
        ]

        with pytest.raises(ValueError, match="Invalid ExperimentSpec at index 1"):
            yanex.run_multiple(invalid_experiments)

    def test_cli_context_allowed(self, per_test_experiments_dir, tmp_path):
        """Test that run_multiple works from CLI context (orchestrator pattern)."""
        script_content = """
print("Test")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Simulate CLI context (e.g., orchestrator run via 'yanex run')
        os.environ["YANEX_CLI_ACTIVE"] = "1"

        experiments = [
            ExperimentSpec(script_path=script_path, config={"param": 1}),
        ]

        # Should work fine - orchestrator pattern allows nested execution
        results = yanex.run_multiple(experiments)

        # Verify child experiment succeeded
        assert len(results) == 1
        assert results[0].status == "completed"

    def test_script_args_passthrough(self, per_test_experiments_dir, tmp_path):
        """Test that script_args are passed through to subprocess."""
        # Create script that uses argparse
        script_content = """
import argparse
import yanex

parser = argparse.ArgumentParser()
parser.add_argument('--data-exp', required=True)
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

lr = yanex.get_param('learning_rate', default=0.001)
print(f"lr={lr} data={args.data_exp} fold={args.fold}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Create experiments with script_args
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.01},
                script_args=["--data-exp", "abc123", "--fold", "0"],
                name="fold-0",
            ),
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.01},
                script_args=["--data-exp", "abc123", "--fold", "1"],
                name="fold-1",
            ),
        ]

        # Run sequentially
        results = yanex.run_multiple(experiments, parallel=None)

        # Verify results
        assert len(results) == 2
        assert all(r.status == "completed" for r in results)

        # Verify script_args were stored in metadata
        manager = ExperimentManager(per_test_experiments_dir)
        for result in results:
            metadata = manager.get_experiment_metadata(result.experiment_id)
            assert "script_args" in metadata
            assert "--data-exp" in metadata["script_args"]
            assert "abc123" in metadata["script_args"]

    def test_script_args_with_parallel_execution(
        self, per_test_experiments_dir, tmp_path
    ):
        """Test script_args work correctly in parallel execution."""
        script_content = """
import argparse
import yanex

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, required=True)
args = parser.parse_args()

lr = yanex.get_param('learning_rate')
print(f"Fold {args.fold} with lr={lr}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Create 3 experiments with different fold numbers
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"learning_rate": 0.01},
                script_args=["--fold", str(i)],
                name=f"fold-{i}",
            )
            for i in range(3)
        ]

        # Run in parallel
        results = yanex.run_multiple(experiments, parallel=2)

        # Verify all completed
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

    def test_experiment_result_includes_duration(
        self, per_test_experiments_dir, tmp_path
    ):
        """Test that ExperimentResult includes execution duration."""
        script_content = """
import time
time.sleep(0.05)
print("Done")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        experiments = [
            ExperimentSpec(script_path=script_path, config={}, name="timed"),
        ]

        results = yanex.run_multiple(experiments, parallel=None)

        assert len(results) == 1
        result = results[0]

        # Should have duration
        assert result.duration is not None
        assert result.duration > 0.05  # Should take at least 0.05 seconds
        assert result.duration < 5.0  # But not more than 5 seconds

    def test_tags_and_description_stored(self, per_test_experiments_dir, tmp_path):
        """Test that tags and description are properly stored."""
        script_content = """
print("Test")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"param": 1},
                name="tagged-exp",
                tags=["kfold", "cv", "test"],
                description="K-fold cross-validation experiment",
            ),
        ]

        results = yanex.run_multiple(experiments, parallel=None)

        assert len(results) == 1
        assert results[0].status == "completed"

        # Verify metadata
        manager = ExperimentManager(per_test_experiments_dir)
        metadata = manager.get_experiment_metadata(results[0].experiment_id)

        assert set(metadata["tags"]) == {"kfold", "cv", "test"}
        assert metadata["description"] == "K-fold cross-validation experiment"

    def test_parallel_failure_handling(self, per_test_experiments_dir, tmp_path):
        """Test error handling in parallel execution mode."""
        # Create script that fails for certain values
        script_content = """
import yanex
value = yanex.get_param('value')
if value % 2 == 0:
    raise ValueError(f"Even value {value} not allowed")
print(f"Odd value {value} succeeded")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        # Create 6 experiments (3 will fail, 3 will succeed)
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={"value": i},
                name=f"exp-{i}",
            )
            for i in range(1, 7)
        ]

        # Run in parallel
        results = yanex.run_multiple(experiments, parallel=3)

        # Verify results
        assert len(results) == 6

        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]

        assert len(completed) == 3  # Odd values: 1, 3, 5
        assert len(failed) == 3  # Even values: 2, 4, 6

        # All failed experiments should have error messages
        for r in failed:
            assert r.error_message is not None
            assert "not allowed" in r.error_message

    def test_config_parameters_passed_correctly(
        self, per_test_experiments_dir, tmp_path
    ):
        """Test that config parameters are correctly passed to experiments."""
        script_content = """
import yanex

# Get various parameter types
lr = yanex.get_param('learning_rate')
batch_size = yanex.get_param('batch_size')
optimizer = yanex.get_param('optimizer')
use_cuda = yanex.get_param('use_cuda')

print(f"lr={lr} batch={batch_size} opt={optimizer} cuda={use_cuda}")

# Verify types
assert isinstance(lr, float)
assert isinstance(batch_size, int)
assert isinstance(optimizer, str)
assert isinstance(use_cuda, bool)
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(script_content)
        experiments = [
            ExperimentSpec(
                script_path=script_path,
                config={
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "optimizer": "adam",
                    "use_cuda": True,
                },
                name="config-test",
            ),
        ]

        results = yanex.run_multiple(experiments, parallel=None)

        assert len(results) == 1
        assert results[0].status == "completed"


class TestKFoldOrchestrationPattern:
    """Test k-fold cross-validation orchestration pattern."""

    def test_kfold_execution_mode_detection(self, per_test_experiments_dir, tmp_path):
        """Test that scripts can detect execution mode vs orchestration mode."""
        # Create script that uses _fold_idx parameter to detect mode
        train_script_content = """
import yanex

# Detect mode using _fold_idx parameter
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # ORCHESTRATION MODE: would spawn experiments here
    print("MODE: ORCHESTRATION")
else:
    # EXECUTION MODE: run single fold
    print(f"MODE: EXECUTION fold={fold_idx}")
"""
        script_path = tmp_path / "train.py"
        script_path.write_text(train_script_content)
        # Test 1: Orchestration mode (no _fold_idx)
        orchestration_exp = [
            ExperimentSpec(
                script_path=script_path,
                config={},  # No _fold_idx
                name="orchestration-test",
            ),
        ]

        results = yanex.run_multiple(orchestration_exp, parallel=None)
        assert len(results) == 1
        assert results[0].status == "completed"

        # Test 2: Execution mode (with _fold_idx)
        execution_exps = [
            ExperimentSpec(
                script_path=script_path,
                config={"_fold_idx": i},
                name=f"fold-{i}",
                tags=["kfold"],
            )
            for i in range(3)
        ]

        results = yanex.run_multiple(execution_exps, parallel=2)

        # All folds should complete
        assert len(results) == 3
        assert all(r.status == "completed" for r in results)

        # Verify all experiments were created
        manager = ExperimentManager(per_test_experiments_dir)
        all_experiments = manager.list_experiments()
        assert len(all_experiments) == 4  # 1 orchestration + 3 folds
