"""
Tests for DataFrame API functionality.

This module tests pandas DataFrame integration for experiment comparison.
"""

import pandas as pd
import pytest

from yanex.results.dataframe import (
    correlation_analysis,
    determine_varying_params,
    experiments_to_dataframe,
    find_best_experiments,
    flatten_dataframe_columns,
    format_dataframe_for_analysis,
    get_metric_summary,
    get_parameter_summary,
    metrics_to_long_dataframe,
)


class TestExperimentsToDataFrame:
    """Test experiments_to_dataframe function."""

    def test_basic_conversion(self):
        """Test basic conversion from comparison data to DataFrame."""
        comparison_data = {
            "rows": [
                {
                    "id": "exp1",
                    "name": "test-exp-1",
                    "status": "completed",
                    "param:learning_rate": 0.01,
                    "param:batch_size": 32,
                    "metric:accuracy": 0.95,
                    "metric:loss": 0.05,
                },
                {
                    "id": "exp2",
                    "name": "test-exp-2",
                    "status": "completed",
                    "param:learning_rate": 0.001,
                    "param:batch_size": 64,
                    "metric:accuracy": 0.92,
                    "metric:loss": 0.08,
                },
            ]
        }

        df = experiments_to_dataframe(comparison_data)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.names == ["category", "name"]

        # Check dimensions
        assert len(df) == 2

        # Check columns exist
        assert ("param", "learning_rate") in df.columns
        assert ("param", "batch_size") in df.columns
        assert ("metric", "accuracy") in df.columns
        assert ("metric", "loss") in df.columns
        assert ("meta", "name") in df.columns
        assert ("meta", "status") in df.columns

        # Check data
        assert df.loc["exp1", ("param", "learning_rate")] == 0.01
        assert df.loc["exp2", ("metric", "accuracy")] == 0.92

    def test_empty_comparison_data(self):
        """Test with empty comparison data."""
        comparison_data = {"rows": []}

        df = experiments_to_dataframe(comparison_data)

        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)
        assert len(df) == 0

    def test_missing_rows_key(self):
        """Test with missing rows key."""
        comparison_data = {}

        df = experiments_to_dataframe(comparison_data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_with_missing_values(self):
        """Test conversion with missing metric values."""
        comparison_data = {
            "rows": [
                {
                    "id": "exp1",
                    "param:learning_rate": 0.01,
                    "metric:accuracy": 0.95,
                },
                {
                    "id": "exp2",
                    "param:learning_rate": 0.001,
                    "metric:accuracy": None,  # Missing value
                },
            ]
        }

        df = experiments_to_dataframe(comparison_data)

        assert len(df) == 2
        assert pd.isna(df.loc["exp2", ("metric", "accuracy")])

    def test_metadata_columns(self):
        """Test that metadata columns are properly categorized."""
        comparison_data = {
            "rows": [
                {
                    "id": "exp1",
                    "name": "test",
                    "status": "completed",
                    "created_at": "2025-01-01T00:00:00",
                    "param:x": 1,
                }
            ]
        }

        df = experiments_to_dataframe(comparison_data)

        # All non-param/metric columns should be in "meta" category
        meta_cols = [col for col in df.columns if col[0] == "meta"]
        assert len(meta_cols) == 3  # id (index), name, status, created_at
        assert ("meta", "name") in df.columns
        assert ("meta", "status") in df.columns


class TestFormatDataFrameForAnalysis:
    """Test format_dataframe_for_analysis function."""

    def test_numeric_conversion(self):
        """Test that numeric strings are converted to numbers."""
        data = {
            ("param", "lr"): ["0.01", "0.001", "0.0001"],
            ("metric", "acc"): ["0.95", "0.92", "0.90"],
        }
        df = pd.DataFrame(data)

        formatted_df = format_dataframe_for_analysis(df)

        assert pd.api.types.is_numeric_dtype(formatted_df[("param", "lr")])
        assert pd.api.types.is_numeric_dtype(formatted_df[("metric", "acc")])

    def test_datetime_conversion(self):
        """Test that datetime columns are converted."""
        data = {
            ("meta", "started_at"): ["2025-01-01T00:00:00", "2025-01-02T00:00:00"],
            ("meta", "completed_at"): ["2025-01-01T01:00:00", "2025-01-02T01:00:00"],
        }
        df = pd.DataFrame(data)

        formatted_df = format_dataframe_for_analysis(df)

        assert pd.api.types.is_datetime64_any_dtype(
            formatted_df[("meta", "started_at")]
        )
        assert pd.api.types.is_datetime64_any_dtype(
            formatted_df[("meta", "completed_at")]
        )

    def test_duration_conversion(self):
        """Test that duration is converted to timedelta."""
        data = {("meta", "duration"): ["00:05:30", "00:10:15", "01:00:00"]}
        df = pd.DataFrame(data)

        formatted_df = format_dataframe_for_analysis(df)

        assert pd.api.types.is_timedelta64_dtype(formatted_df[("meta", "duration")])

    def test_categorical_conversion(self):
        """Test that repeated values are converted to categorical."""
        # Status with repeated values (> 50% repeated)
        data = {("meta", "status"): ["completed"] * 8 + ["failed"] * 2}
        df = pd.DataFrame(data)

        formatted_df = format_dataframe_for_analysis(df)

        assert formatted_df[("meta", "status")].dtype.name == "category"

    def test_non_categorical_unique_values(self):
        """Test that columns with many unique values stay as object."""
        # Each value is unique (100% unique)
        data = {("meta", "name"): [f"exp-{i}" for i in range(10)]}
        df = pd.DataFrame(data)

        formatted_df = format_dataframe_for_analysis(df)

        assert formatted_df[("meta", "name")].dtype == "object"

    def test_handles_non_convertible_values(self):
        """Test graceful handling of non-convertible values."""
        data = {
            ("param", "value"): ["abc", "def", "ghi"],  # Non-numeric strings
            ("meta", "bad_date"): ["not-a-date", "also-not-a-date", "nope"],
        }
        df = pd.DataFrame(data)

        # Should not raise - just keep original dtypes
        formatted_df = format_dataframe_for_analysis(df)

        assert formatted_df[("param", "value")].dtype == "object"


class TestFlattenDataFrameColumns:
    """Test flatten_dataframe_columns function."""

    def test_basic_flattening(self):
        """Test basic column flattening."""
        data = {
            ("param", "learning_rate"): [0.01, 0.001],
            ("metric", "accuracy"): [0.95, 0.92],
            ("meta", "name"): ["exp1", "exp2"],
        }
        df = pd.DataFrame(data)

        flat_df = flatten_dataframe_columns(df)

        assert not isinstance(flat_df.columns, pd.MultiIndex)
        assert "param_learning_rate" in flat_df.columns
        assert "metric_accuracy" in flat_df.columns
        assert "name" in flat_df.columns  # meta prefix removed

    def test_already_flat(self):
        """Test that already flat DataFrame is returned unchanged."""
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        flat_df = flatten_dataframe_columns(df)

        assert flat_df.equals(df)


class TestGetParameterSummary:
    """Test get_parameter_summary function."""

    def test_basic_summary(self):
        """Test basic parameter summary."""
        data = {
            ("param", "learning_rate"): [0.01, 0.001, 0.0001],
            ("param", "batch_size"): [32, 64, 128],
            ("metric", "accuracy"): [0.95, 0.92, 0.90],
        }
        df = pd.DataFrame(data)

        summary = get_parameter_summary(df)

        # Check that only parameters are in summary
        assert "learning_rate" in summary.columns
        assert "batch_size" in summary.columns
        assert "accuracy" not in summary.columns

        # Check standard statistics
        assert "mean" in summary.index
        assert "std" in summary.index
        assert "min" in summary.index
        assert "max" in summary.index

    def test_empty_when_no_params(self):
        """Test returns empty DataFrame when no parameters."""
        data = {("metric", "accuracy"): [0.95, 0.92]}
        df = pd.DataFrame(data)

        summary = get_parameter_summary(df)

        assert summary.empty

    def test_non_hierarchical_columns(self):
        """Test with non-hierarchical columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        summary = get_parameter_summary(df)

        assert summary.empty


class TestGetMetricSummary:
    """Test get_metric_summary function."""

    def test_basic_summary(self):
        """Test basic metric summary."""
        data = {
            ("param", "learning_rate"): [0.01, 0.001, 0.0001],
            ("metric", "accuracy"): [0.95, 0.92, 0.90],
            ("metric", "loss"): [0.05, 0.08, 0.10],
        }
        df = pd.DataFrame(data)

        summary = get_metric_summary(df)

        # Check that only metrics are in summary
        assert "accuracy" in summary.columns
        assert "loss" in summary.columns
        assert "learning_rate" not in summary.columns

        # Check standard statistics
        assert "mean" in summary.index
        assert "std" in summary.index

    def test_empty_when_no_metrics(self):
        """Test returns empty DataFrame when no metrics."""
        data = {("param", "learning_rate"): [0.01, 0.001]}
        df = pd.DataFrame(data)

        summary = get_metric_summary(df)

        assert summary.empty


class TestCorrelationAnalysis:
    """Test correlation_analysis function."""

    def test_basic_correlation(self):
        """Test basic correlation analysis."""
        data = {
            ("param", "learning_rate"): [0.01, 0.005, 0.001, 0.0005],
            ("param", "batch_size"): [32, 64, 128, 256],
            ("metric", "accuracy"): [0.90, 0.92, 0.95, 0.96],
            ("metric", "loss"): [0.10, 0.08, 0.05, 0.04],
        }
        df = pd.DataFrame(data)

        corr = correlation_analysis(df)

        # Check structure
        assert isinstance(corr, pd.DataFrame)
        assert "learning_rate" in corr.columns
        assert "accuracy" in corr.columns
        assert "loss" in corr.columns

        # Check diagonal is 1.0
        assert corr.loc["accuracy", "accuracy"] == pytest.approx(1.0)

        # Check correlation is symmetric
        assert (
            corr.loc["learning_rate", "accuracy"]
            == corr.loc["accuracy", "learning_rate"]
        )

    def test_empty_when_no_numeric_columns(self):
        """Test returns empty when no numeric columns."""
        data = {
            ("param", "name"): ["a", "b", "c"],
            ("metric", "status"): ["ok", "ok", "fail"],
        }
        df = pd.DataFrame(data)

        corr = correlation_analysis(df)

        assert corr.empty

    def test_non_hierarchical_columns(self):
        """Test with non-hierarchical columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        corr = correlation_analysis(df)

        assert corr.empty


class TestFindBestExperiments:
    """Test find_best_experiments function."""

    def test_maximize_metric(self):
        """Test finding best experiments (maximize)."""
        data = {
            ("metric", "accuracy"): [0.90, 0.95, 0.92, 0.88, 0.93],
            ("param", "lr"): [0.01, 0.005, 0.001, 0.02, 0.002],
        }
        df = pd.DataFrame(data, index=[f"exp{i}" for i in range(5)])

        best = find_best_experiments(df, "accuracy", maximize=True, top_n=3)

        # Check top 3 are returned in descending order
        assert len(best) == 3
        assert best.iloc[0][("metric", "accuracy")] == 0.95
        assert best.iloc[1][("metric", "accuracy")] == 0.93
        assert best.iloc[2][("metric", "accuracy")] == 0.92

    def test_minimize_metric(self):
        """Test finding best experiments (minimize)."""
        data = {
            ("metric", "loss"): [0.10, 0.05, 0.08, 0.12, 0.06],
            ("param", "lr"): [0.01, 0.005, 0.001, 0.02, 0.002],
        }
        df = pd.DataFrame(data, index=[f"exp{i}" for i in range(5)])

        best = find_best_experiments(df, "loss", maximize=False, top_n=2)

        # Check top 2 are returned in ascending order
        assert len(best) == 2
        assert best.iloc[0][("metric", "loss")] == 0.05
        assert best.iloc[1][("metric", "loss")] == 0.06

    def test_metric_not_found(self):
        """Test error when metric doesn't exist."""
        data = {
            ("metric", "accuracy"): [0.90, 0.95],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError, match="Metric 'loss' not found"):
            find_best_experiments(df, "loss", maximize=True)

    def test_handles_missing_values(self):
        """Test that rows with missing metric values are filtered out."""
        data = {
            ("metric", "accuracy"): [0.90, None, 0.92, 0.88, None],
            ("param", "lr"): [0.01, 0.005, 0.001, 0.02, 0.002],
        }
        df = pd.DataFrame(data, index=[f"exp{i}" for i in range(5)])

        best = find_best_experiments(df, "accuracy", maximize=True, top_n=5)

        # Should only return 3 (non-None values)
        assert len(best) == 3
        assert not best[("metric", "accuracy")].isna().any()

    def test_non_hierarchical_columns(self):
        """Test with non-hierarchical columns."""
        df = pd.DataFrame({"col1": [1, 2, 3]})

        result = find_best_experiments(df, "col1", maximize=True)

        assert result.empty


# Note: export_comparison_summary tests are intentionally omitted
# Excel export is optional convenience functionality that requires openpyxl
# Core DataFrame API functionality is tested above


class TestDetermineVaryingParams:
    """Test determine_varying_params function."""

    def test_all_params_same(self):
        """Test when all parameters are the same across experiments."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.get_params.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}
        exp2 = Mock()
        exp2.get_params.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}
        exp3 = Mock()
        exp3.get_params.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}

        experiments = [exp1, exp2, exp3]
        varying = determine_varying_params(experiments)

        assert varying == []

    def test_some_params_varying(self):
        """Test when some parameters vary."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.get_params.return_value = {"lr": 0.01, "batch_size": 32, "epochs": 10}
        exp2 = Mock()
        exp2.get_params.return_value = {"lr": 0.001, "batch_size": 32, "epochs": 20}
        exp3 = Mock()
        exp3.get_params.return_value = {"lr": 0.1, "batch_size": 32, "epochs": 10}

        experiments = [exp1, exp2, exp3]
        varying = determine_varying_params(experiments)

        # lr and epochs vary, batch_size does not
        assert set(varying) == {"lr", "epochs"}

    def test_all_params_varying(self):
        """Test when all parameters vary."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.get_params.return_value = {"lr": 0.01, "batch_size": 32}
        exp2 = Mock()
        exp2.get_params.return_value = {"lr": 0.001, "batch_size": 64}
        exp3 = Mock()
        exp3.get_params.return_value = {"lr": 0.1, "batch_size": 128}

        experiments = [exp1, exp2, exp3]
        varying = determine_varying_params(experiments)

        assert set(varying) == {"lr", "batch_size"}

    def test_empty_experiment_list(self):
        """Test with empty experiment list."""
        varying = determine_varying_params([])
        assert varying == []

    def test_single_experiment(self):
        """Test with single experiment."""
        from unittest.mock import Mock

        exp = Mock()
        exp.get_params.return_value = {"lr": 0.01, "batch_size": 32}

        varying = determine_varying_params([exp])
        assert varying == []

    def test_numeric_string_distinction(self):
        """Test that numeric values are properly compared as strings."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.get_params.return_value = {"lr": 0.01, "name": "exp1"}
        exp2 = Mock()
        exp2.get_params.return_value = {"lr": 0.010, "name": "exp2"}  # Same as 0.01

        experiments = [exp1, exp2]
        varying = determine_varying_params(experiments)

        # name varies, lr does not (0.01 == 0.010)
        assert varying == ["name"]


class TestMetricsToLongDataFrame:
    """Test metrics_to_long_dataframe function."""

    def test_basic_conversion(self):
        """Test basic conversion to long format."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.id = "exp1"
        exp1.get_params.return_value = {"lr": 0.01, "epochs": 10}
        # get_metrics(as_dataframe=False) returns list of dicts
        exp1.get_metrics.return_value = [
            {"step": 0, "train_loss": 0.5, "train_acc": 0.8},
            {"step": 1, "train_loss": 0.4, "train_acc": 0.85},
            {"step": 2, "train_loss": 0.3, "train_acc": 0.9},
        ]

        exp2 = Mock()
        exp2.id = "exp2"
        exp2.get_params.return_value = {"lr": 0.001, "epochs": 10}
        exp2.get_metrics.return_value = [
            {"step": 0, "train_loss": 0.6, "train_acc": 0.75},
            {"step": 1, "train_loss": 0.5, "train_acc": 0.8},
            {"step": 2, "train_loss": 0.4, "train_acc": 0.85},
        ]

        experiments = [exp1, exp2]
        df = metrics_to_long_dataframe(experiments)

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert "experiment_id" in df.columns
        assert "step" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns

        # Check data
        assert len(df) == 12  # 2 experiments × 2 metrics × 3 steps
        assert set(df["experiment_id"].unique()) == {"exp1", "exp2"}
        assert set(df["metric_name"].unique()) == {"train_loss", "train_acc"}
        assert df["step"].min() == 0
        assert df["step"].max() == 2

    def test_with_param_cols(self):
        """Test with specific parameter columns."""
        from unittest.mock import Mock

        exp = Mock()
        exp.id = "exp1"
        exp.get_params.return_value = {"lr": 0.01, "epochs": 10, "batch_size": 32}
        exp.get_metrics.return_value = [
            {"step": 0, "loss": 0.5},
            {"step": 1, "loss": 0.4},
        ]

        df = metrics_to_long_dataframe([exp], param_cols=["lr", "epochs"])

        # Should include specified params
        assert "lr" in df.columns
        assert "epochs" in df.columns
        # Should not include batch_size
        assert "batch_size" not in df.columns

    def test_with_empty_param_cols(self):
        """Test with empty param_cols (no parameters)."""
        from unittest.mock import Mock

        exp = Mock()
        exp.id = "exp1"
        exp.get_params.return_value = {"lr": 0.01}
        exp.get_metrics.return_value = [
            {"step": 0, "loss": 0.5},
            {"step": 1, "loss": 0.4},
        ]

        df = metrics_to_long_dataframe([exp], param_cols=[])

        # Should not include any parameter columns
        assert "lr" not in df.columns
        assert "experiment_id" in df.columns
        assert "metric_name" in df.columns

    def test_filter_specific_metrics(self):
        """Test filtering to specific metrics."""
        from unittest.mock import Mock

        exp = Mock()
        exp.id = "exp1"
        exp.get_params.return_value = {}
        exp.get_metrics.return_value = [
            {"step": 0, "train_loss": 0.5, "train_acc": 0.8, "val_loss": 0.6},
            {"step": 1, "train_loss": 0.4, "train_acc": 0.85, "val_loss": 0.5},
        ]

        df = metrics_to_long_dataframe([exp], metrics=["train_loss", "train_acc"])

        # Should only include specified metrics
        assert set(df["metric_name"].unique()) == {"train_loss", "train_acc"}
        assert "val_loss" not in df["metric_name"].values

    def test_empty_experiment_list(self):
        """Test with empty experiment list."""
        df = metrics_to_long_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "experiment_id" in df.columns
        assert "step" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns

    def test_single_experiment(self):
        """Test with single experiment."""
        from unittest.mock import Mock

        exp = Mock()
        exp.id = "exp1"
        exp.get_params.return_value = {"lr": 0.01}
        exp.get_metrics.return_value = [
            {"step": 0, "loss": 0.5},
            {"step": 1, "loss": 0.4},
            {"step": 2, "loss": 0.3},
        ]

        df = metrics_to_long_dataframe([exp], param_cols=["lr"])

        assert len(df) == 3  # 3 steps
        assert df["experiment_id"].unique()[0] == "exp1"
        assert df["lr"].unique()[0] == 0.01

    def test_experiments_with_no_metrics(self):
        """Test handling experiments with no metrics."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.id = "exp1"
        exp1.get_params.return_value = {}
        exp1.get_metrics.return_value = []  # Empty list

        exp2 = Mock()
        exp2.id = "exp2"
        exp2.get_params.return_value = {}
        exp2.get_metrics.return_value = [{"step": 0, "loss": 0.5}]

        df = metrics_to_long_dataframe([exp1, exp2])

        # Should only include exp2 which has metrics
        assert len(df) == 1
        assert df["experiment_id"].unique()[0] == "exp2"

    def test_missing_metrics_in_some_experiments(self):
        """Test when different experiments log different metrics."""
        from unittest.mock import Mock

        exp1 = Mock()
        exp1.id = "exp1"
        exp1.get_params.return_value = {}
        exp1.get_metrics.return_value = [
            {"step": 0, "loss": 0.5},
            {"step": 1, "loss": 0.4},
        ]

        exp2 = Mock()
        exp2.id = "exp2"
        exp2.get_params.return_value = {}
        exp2.get_metrics.return_value = [
            {"step": 0, "loss": 0.6, "accuracy": 0.8},
            {"step": 1, "loss": 0.5, "accuracy": 0.85},
        ]

        df = metrics_to_long_dataframe([exp1, exp2])

        # exp1 contributes 2 rows (loss only)
        # exp2 contributes 4 rows (loss + accuracy)
        assert len(df) == 6
        assert set(df["metric_name"].unique()) == {"loss", "accuracy"}

        # Check exp1 has no accuracy entries
        exp1_data = df[df["experiment_id"] == "exp1"]
        assert "accuracy" not in exp1_data["metric_name"].values
