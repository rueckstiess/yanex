"""Tests for visualization data extraction layer."""

import pandas as pd
import pytest

from yanex.results.viz.data import (
    _extract_metadata,
    _extract_params,
    detect_plot_type,
    extract_metrics_df,
)


class TestExtractMetricsDF:
    """Tests for extract_metrics_df function."""

    def test_extract_single_experiment_multi_step(self, sample_experiments):
        """Test extracting multi-step metrics from single experiment."""
        exp = sample_experiments["multi_step_single"]
        df = extract_metrics_df([exp], ["accuracy", "loss"])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 3 steps
        assert list(df.columns) == [
            "experiment_id",
            "step",
            "timestamp",
            "accuracy",
            "loss",
            "name",
            "status",
        ]
        assert df["experiment_id"].iloc[0] == exp.id
        assert df["accuracy"].tolist() == [0.8, 0.85, 0.9]
        assert df["loss"].tolist() == [0.2, 0.15, 0.1]

    def test_extract_single_experiment_single_step(self, sample_experiments):
        """Test extracting single-step metrics."""
        exp = sample_experiments["single_step"]
        df = extract_metrics_df([exp], ["final_accuracy"])

        assert len(df) == 1
        assert df["step"].iloc[0] is None
        assert df["final_accuracy"].iloc[0] == 0.95

    def test_extract_multiple_experiments(self, sample_experiments):
        """Test extracting from multiple experiments."""
        exps = [
            sample_experiments["multi_step_single"],
            sample_experiments["multi_step_with_params"],
        ]
        df = extract_metrics_df(exps, ["accuracy"])

        assert len(df) == 6  # 3 steps * 2 experiments
        assert df["experiment_id"].nunique() == 2

    def test_extract_with_params(self, sample_experiments):
        """Test extracting with parameters included."""
        exp = sample_experiments["multi_step_with_params"]
        df = extract_metrics_df(
            [exp], ["accuracy"], include_params=["learning_rate", "batch_size"]
        )

        assert "learning_rate" in df.columns
        assert "batch_size" in df.columns
        assert df["learning_rate"].iloc[0] == 0.001
        assert df["batch_size"].iloc[0] == 32

    def test_extract_auto_discover_params(self, sample_experiments):
        """Test auto-discovering all parameters."""
        exp = sample_experiments["multi_step_with_params"]
        df = extract_metrics_df([exp], ["accuracy"], include_params=None)

        assert "learning_rate" in df.columns
        assert "batch_size" in df.columns

    def test_metric_not_found(self, sample_experiments):
        """Test error when metric doesn't exist."""
        exp = sample_experiments["multi_step_single"]
        with pytest.raises(ValueError, match="Metric 'nonexistent' not found"):
            extract_metrics_df([exp], ["nonexistent"])

    def test_no_experiments(self):
        """Test error when no experiments provided."""
        with pytest.raises(ValueError, match="No experiments provided"):
            extract_metrics_df([], ["accuracy"])

    def test_no_metrics(self, sample_experiments):
        """Test error when no metrics specified."""
        exp = sample_experiments["multi_step_single"]
        with pytest.raises(ValueError, match="No metrics specified"):
            extract_metrics_df([exp], [])

    def test_inconsistent_steps_across_experiments(self, sample_experiments):
        """Test error when mixing single and multi-step metrics."""
        exps = [
            sample_experiments["multi_step_single"],
            sample_experiments["single_step"],
        ]
        # This should work if we're querying different metrics
        # But should fail if same metric is inconsistent
        with pytest.raises(ValueError, match="inconsistent step counts"):
            # Both have "accuracy" but different step structures
            extract_metrics_df(exps, ["accuracy"])


class TestDetectPlotType:
    """Tests for plot type detection."""

    def test_detect_line_plot(self):
        """Test detecting line plot for multi-step data."""
        df = pd.DataFrame({"step": [0, 1, 2], "accuracy": [0.8, 0.85, 0.9]})
        assert detect_plot_type(df) == "line"

    def test_detect_bar_plot(self):
        """Test detecting bar plot for single-step data."""
        df = pd.DataFrame({"step": [None, None], "accuracy": [0.8, 0.85]})
        assert detect_plot_type(df) == "bar"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_metadata(self, sample_experiments):
        """Test metadata extraction."""
        exp = sample_experiments["multi_step_single"]
        metadata = _extract_metadata(exp, ["name", "status"])

        assert "name" in metadata
        assert "status" in metadata
        assert metadata["status"] == "completed"

    def test_extract_metadata_with_tags(self, sample_experiments):
        """Test metadata extraction with list conversion."""
        exp = sample_experiments["with_tags"]
        metadata = _extract_metadata(exp, ["tags"])

        assert "tags" in metadata
        # Tags should be converted to comma-separated string
        assert isinstance(metadata["tags"], str)
        assert "training" in metadata["tags"]

    def test_extract_params(self, sample_experiments):
        """Test parameter extraction."""
        exp = sample_experiments["multi_step_with_params"]
        params = _extract_params(exp, ["learning_rate", "batch_size"])

        assert params["learning_rate"] == 0.001
        assert params["batch_size"] == 32

    def test_extract_all_params(self, sample_experiments):
        """Test extracting all parameters."""
        exp = sample_experiments["multi_step_with_params"]
        params = _extract_params(exp, None)

        assert "learning_rate" in params
        assert "batch_size" in params


@pytest.fixture
def sample_experiments(tmp_path, git_repo):
    """Create sample experiments for testing."""
    from yanex.core.manager import ExperimentManager

    experiments = {}

    # Multi-step single experiment
    manager1 = ExperimentManager(storage_path=tmp_path / "exp1", cwd=git_repo)
    manager1.init()
    manager1.storage.add_result_step({"accuracy": 0.8, "loss": 0.2})
    manager1.storage.add_result_step({"accuracy": 0.85, "loss": 0.15})
    manager1.storage.add_result_step({"accuracy": 0.9, "loss": 0.1})
    manager1.finalize(status="completed")
    from yanex.results.experiment import Experiment

    experiments["multi_step_single"] = Experiment(manager1.experiment_id)

    # Multi-step with parameters
    manager2 = ExperimentManager(
        config={"learning_rate": 0.001, "batch_size": 32},
        storage_path=tmp_path / "exp2",
        cwd=git_repo,
    )
    manager2.init()
    manager2.storage.add_result_step({"accuracy": 0.75})
    manager2.storage.add_result_step({"accuracy": 0.82})
    manager2.storage.add_result_step({"accuracy": 0.88})
    manager2.finalize(status="completed")
    experiments["multi_step_with_params"] = Experiment(manager2.experiment_id)

    # Single-step
    manager3 = ExperimentManager(storage_path=tmp_path / "exp3", cwd=git_repo)
    manager3.init()
    manager3.storage.add_result_step({"final_accuracy": 0.95})
    manager3.finalize(status="completed")
    experiments["single_step"] = Experiment(manager3.experiment_id)

    # With tags
    manager4 = ExperimentManager(
        tags=["training", "cnn"], storage_path=tmp_path / "exp4", cwd=git_repo
    )
    manager4.init()
    manager4.storage.add_result_step({"accuracy": 0.9})
    manager4.finalize(status="completed")
    experiments["with_tags"] = Experiment(manager4.experiment_id)

    return experiments
