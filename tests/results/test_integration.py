"""
Integration tests for the Results API.

These tests verify that the entire Results API works together correctly,
including module-level convenience functions and real-world usage patterns.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

import yanex.results as yr
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import ExperimentNotFoundError


class TestResultsAPIIntegration:
    """Integration tests for the complete Results API."""

    def teardown_method(self, method):
        """Clean up experiments after each test method."""
        try:
            from yanex.core.filtering import UnifiedExperimentFilter
            from yanex.core.manager import ExperimentManager

            manager = ExperimentManager()
            filter_obj = UnifiedExperimentFilter(manager=manager)
            test_experiments = filter_obj.filter_experiments(
                tags=["unit-tests"], limit=100
            )

            for exp in test_experiments:
                try:
                    if exp.get("status") == "running":
                        manager.cancel_experiment(exp["id"], "Test cleanup")
                    manager.delete_experiment(exp["id"])
                except Exception:
                    pass
        except Exception:
            pass

    @pytest.fixture
    def experiment_manager(self, isolated_experiments_dir, clean_git_repo):
        """Create an experiment manager for setup."""
        return ExperimentManager(experiments_dir=isolated_experiments_dir)

    @pytest.fixture
    def custom_manager(self, isolated_experiments_dir):
        """Create a custom results manager."""
        return yr.get_manager(storage_path=isolated_experiments_dir)

    @pytest.fixture
    @patch("yanex.core.manager.validate_clean_working_directory")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def comprehensive_experiments(
        self, mock_capture_env, mock_git_info, mock_validate_git, experiment_manager
    ):
        """Create a comprehensive set of experiments for testing."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        experiments = []

        # Training experiments with different hyperparameters
        for i, lr in enumerate([0.001, 0.01, 0.1], 1):
            exp_id = experiment_manager.create_experiment(
                script_path=Path(f"train_{i}.py"),
                name=f"training_run_{i}",
                config={
                    "learning_rate": lr,
                    "epochs": 100,
                    "batch_size": 32,
                    "model_type": "cnn" if i <= 2 else "rnn",
                },
                tags=["training", "hyperparameter_search", "unit-tests"],
                description=f"Training run {i} with LR={lr}",
            )
            experiment_manager.start_experiment(exp_id)

            # Add metrics (inverse relationship between LR and accuracy for testing)
            accuracy = 0.9 - (lr - 0.001) * 0.3
            loss = 0.1 + (lr - 0.001) * 0.3

            experiment_manager.storage.add_result_step(
                exp_id, {"accuracy": accuracy, "loss": loss, "step": 100}
            )

            # Complete first two experiments, cancel the third
            if i < 3:
                experiment_manager.complete_experiment(exp_id)
            else:
                experiment_manager.cancel_experiment(exp_id, "Test cancellation")

            experiments.append(exp_id)

        # Evaluation experiment
        eval_exp_id = experiment_manager.create_experiment(
            script_path=Path("evaluate.py"),
            name="model_evaluation",
            config={"model_path": "/models/best.pt", "test_split": 0.2},
            tags=["evaluation", "testing", "unit-tests"],
            description="Final model evaluation",
        )
        experiment_manager.start_experiment(eval_exp_id)
        experiment_manager.storage.add_result_step(
            eval_exp_id, {"precision": 0.92, "recall": 0.88, "f1_score": 0.90}
        )
        experiment_manager.complete_experiment(eval_exp_id)
        experiments.append(eval_exp_id)

        # Failed experiment
        failed_exp_id = experiment_manager.create_experiment(
            script_path=Path("broken.py"),
            name="failed_experiment",
            config={"will_fail": True},
            tags=["debug", "unit-tests"],
            description="This experiment was meant to fail",
        )
        experiment_manager.start_experiment(failed_exp_id)
        experiment_manager.fail_experiment(
            failed_exp_id, "Intentional failure for testing"
        )
        experiments.append(failed_exp_id)

        return experiments

    def test_module_level_functions(self, custom_manager, comprehensive_experiments):
        """Test all module-level convenience functions."""
        # Set custom manager as default for this test
        yr._default_manager = custom_manager

        try:
            # Test find()
            all_experiments = yr.find()
            assert len(all_experiments) == 5

            # Test filtering
            training_experiments = yr.find(tags=["training"])
            assert len(training_experiments) == 3

            completed_experiments = yr.find(status="completed")
            assert len(completed_experiments) == 3  # 2 training + 1 evaluation

            # Test combined filters
            completed_training = yr.find(status="completed", tags=["training"])
            assert len(completed_training) == 2

            # Test get_experiment()
            exp_id = all_experiments[0]["id"]
            exp = yr.get_experiment(exp_id)
            assert exp.id == exp_id

            # Test get_experiments()
            training_exp_objects = yr.get_experiments(tags=["training"])
            assert len(training_exp_objects) == 3
            assert all(hasattr(exp, "get_params") for exp in training_exp_objects)

            # Test get_latest()
            latest = yr.get_latest()
            assert latest is not None

            latest_training = yr.get_latest(tags=["training"])
            assert latest_training is not None
            assert "training" in latest_training.tags

            # Test get_best()
            best_accuracy = yr.get_best("accuracy", maximize=True)
            assert best_accuracy is not None
            metrics = best_accuracy.get_metrics()
            accuracy_value = None
            if metrics:
                for entry in reversed(metrics):
                    if "accuracy" in entry:
                        accuracy_value = entry["accuracy"]
                        break
            assert accuracy_value is not None

            # Test list_experiments()
            recent = yr.list_experiments(limit=3)
            assert len(recent) == 3

            # Test experiment_exists()
            assert yr.experiment_exists(exp_id) is True
            assert yr.experiment_exists("nonexistent") is False

            # Test get_experiment_count()
            total_count = yr.get_experiment_count()
            assert total_count == 5

            training_count = yr.get_experiment_count(tags=["training"])
            assert training_count == 3

        finally:
            # Reset default manager
            yr._default_manager = None

    def test_real_world_analysis_workflow(
        self, custom_manager, comprehensive_experiments
    ):
        """Test a realistic analysis workflow."""
        # Set custom manager
        yr._default_manager = custom_manager

        try:
            # 1. Find all completed training experiments
            training_experiments = yr.find(tags=["training"], status="completed")
            assert len(training_experiments) == 2

            # 2. Get them as Experiment objects for detailed analysis
            training_objects = yr.get_experiments(tags=["training"], status="completed")

            # 3. Analyze their performance
            best_training = yr.get_best(
                "accuracy", maximize=True, tags=["training"], status="completed"
            )
            assert best_training is not None

            # 4. Compare hyperparameters of training runs
            learning_rates = []
            accuracies = []

            for exp in training_objects:
                lr = exp.get_param("learning_rate")
                # Get accuracy from the latest metrics entry
                metrics = exp.get_metrics()
                acc = None
                if metrics:
                    for entry in reversed(metrics):
                        if "accuracy" in entry:
                            acc = entry["accuracy"]
                            break
                if lr is not None and acc is not None:
                    learning_rates.append(lr)
                    accuracies.append(acc)

            assert len(learning_rates) == 2
            assert len(accuracies) == 2

            # Verify inverse relationship (lower LR should give higher accuracy)
            lr_acc_pairs = list(zip(learning_rates, accuracies, strict=False))
            lr_acc_pairs.sort(key=lambda x: x[0])  # Sort by LR
            assert (
                lr_acc_pairs[0][1] > lr_acc_pairs[1][1]
            )  # Lower LR has higher accuracy

            # 5. Export results for further analysis
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                yr.export_experiments(
                    f.name, format="json", tags=["training"], status="completed"
                )

                # Verify export
                import json

                f.seek(0)
                with open(f.name) as read_f:
                    exported_data = json.load(read_f)

                assert len(exported_data) == 2
                assert all("params" in exp for exp in exported_data)
                assert all("metrics" in exp for exp in exported_data)

        finally:
            yr._default_manager = None

    def test_experiment_lifecycle_operations(
        self, custom_manager, comprehensive_experiments
    ):
        """Test experiment lifecycle operations."""
        yr._default_manager = custom_manager

        try:
            # Find a completed experiment to modify
            completed = yr.find(status="completed", limit=1)
            assert len(completed) > 0

            exp_id = completed[0]["id"]
            exp = yr.get_experiment(exp_id)

            # Test metadata modifications
            # original_name = exp.name  # Not used in test
            exp.set_name("modified_name")
            assert exp.name == "modified_name"

            # Verify persistence
            exp_reloaded = yr.get_experiment(exp_id)
            assert exp_reloaded.name == "modified_name"

            # Test tag operations
            original_tags = set(exp.tags)
            exp.add_tags(["analysis", "modified"])
            expected_tags = original_tags | {"analysis", "modified"}
            assert set(exp.tags) == expected_tags

            exp.remove_tags(["analysis"])
            expected_tags.remove("analysis")
            assert set(exp.tags) == expected_tags

            # Test description
            exp.set_description("Modified for testing")
            assert exp.description == "Modified for testing"

        finally:
            yr._default_manager = None

    def test_filtering_edge_cases(self, custom_manager, comprehensive_experiments):
        """Test edge cases in filtering."""
        yr._default_manager = custom_manager

        try:
            # Test empty results (using valid status that doesn't match any experiments)
            no_results = yr.find(
                status="staged"
            )  # No experiments should have staged status
            assert len(no_results) == 0

            # Test with None results
            no_latest = yr.get_latest(status="staged")
            assert no_latest is None

            # Test invalid status raises error
            with pytest.raises(ValueError, match="Invalid status"):
                yr.find(status="nonexistent")

            no_best = yr.get_best("nonexistent_metric", status="completed")
            assert no_best is None

            # Test complex filtering
            complex_filter = yr.find(
                tags=["training"],
                status=["completed", "running"],
                name="training_*",
            )
            assert len(complex_filter) >= 2  # Should find training experiments

            # Test ID filtering
            all_exp = yr.find()
            if len(all_exp) >= 2:
                ids_to_find = [all_exp[0]["id"], all_exp[1]["id"]]
                found_by_ids = yr.find(ids=ids_to_find)
                assert len(found_by_ids) == 2
                found_ids = {exp["id"] for exp in found_by_ids}
                assert found_ids == set(ids_to_find)

        finally:
            yr._default_manager = None

    def test_error_handling(self, custom_manager):
        """Test error handling in the API."""
        yr._default_manager = custom_manager

        try:
            # Test non-existent experiment
            with pytest.raises(ExperimentNotFoundError):
                yr.get_experiment("nonexistent")

            # Test invalid export format
            with pytest.raises(ValueError):
                yr.export_experiments("test.xyz", format="invalid")

        finally:
            yr._default_manager = None

    @pytest.mark.skipif(
        condition=True,  # Skip by default
        reason="Requires pandas for DataFrame functionality",
    )
    def test_dataframe_integration(self, custom_manager, comprehensive_experiments):
        """Test pandas DataFrame integration."""
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        yr._default_manager = custom_manager

        try:
            # Test basic comparison
            df = yr.compare(
                tags=["training"],
                params=["learning_rate", "epochs"],
                metrics=["accuracy", "loss"],
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3  # Three training experiments

            # Test hierarchical columns
            assert isinstance(df.columns, pd.MultiIndex)

            # Test parameter access
            learning_rates = df[("param", "learning_rate")]
            assert len(learning_rates) == 3
            assert all(lr > 0 for lr in learning_rates if pd.notna(lr))

            # Test metric access
            accuracies = df[("metric", "accuracy")]
            assert len(accuracies) == 3
            assert all(0 <= acc <= 1 for acc in accuracies if pd.notna(acc))

            # Test filtering with comparison
            completed_df = yr.compare(
                status="completed",
                tags=["training"],
                params=["learning_rate"],
                metrics=["accuracy"],
            )
            assert len(completed_df) == 2  # Only completed training experiments

        finally:
            yr._default_manager = None

    @patch("yanex.core.manager.validate_clean_working_directory")
    @patch("yanex.core.manager.get_current_commit_info")
    @patch("yanex.core.manager.capture_full_environment")
    def test_performance_with_many_experiments(
        self,
        mock_capture_env,
        mock_git_info,
        mock_validate_git,
        custom_manager,
        experiment_manager,
    ):
        """Test performance with a larger number of experiments."""
        # Setup mocks
        mock_validate_git.return_value = None
        mock_git_info.return_value = {"commit": "abc123", "branch": "main"}
        mock_capture_env.return_value = {"python_version": "3.11.0"}

        yr._default_manager = custom_manager

        try:
            # Create many experiments quickly
            many_exp_ids = []
            for i in range(20):
                exp_id = experiment_manager.create_experiment(
                    script_path=Path(f"perf_test_{i}.py"),
                    name=f"perf_test_{i}",
                    config={"iteration": i, "value": i * 0.1},
                    tags=["performance_test", "unit-tests"],
                )
                many_exp_ids.append(exp_id)

            # Test that operations still work efficiently
            all_perf_experiments = yr.find(tags=["performance_test"], limit=50)
            assert len(all_perf_experiments) == 20

            # Test counting
            count = yr.get_experiment_count(tags=["performance_test"])
            assert count == 20

            # Test limiting
            limited = yr.find(tags=["performance_test"], limit=5)
            assert len(limited) == 5

        finally:
            yr._default_manager = None
