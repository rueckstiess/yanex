"""Integration tests for visualization module."""

import matplotlib
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import yanex
import yanex.results as yr


class TestPlotMetricsIntegration:
    """Integration tests for plot_metrics()."""

    @pytest.fixture(autouse=True)
    def setup_experiments(self, tmp_path, clean_git_repo):
        """Create sample experiments for testing."""
        # Set up storage
        yanex.set_experiments_dir(tmp_path)

        # Create experiments with multi-step metrics
        with yanex.create_experiment(
            script_path="train.py",
            config={"learning_rate": 0.001, "model": "cnn"},
            tags=["training"],
        ):
            yanex.log_metrics({"accuracy": 0.8, "loss": 0.2})
            yanex.log_metrics({"accuracy": 0.85, "loss": 0.15})
            yanex.log_metrics({"accuracy": 0.9, "loss": 0.1})

        with yanex.create_experiment(
            script_path="train.py",
            config={"learning_rate": 0.01, "model": "cnn"},
            tags=["training"],
        ):
            yanex.log_metrics({"accuracy": 0.75, "loss": 0.25})
            yanex.log_metrics({"accuracy": 0.82, "loss": 0.18})
            yanex.log_metrics({"accuracy": 0.88, "loss": 0.12})

        with yanex.create_experiment(
            script_path="train.py",
            config={"learning_rate": 0.001, "model": "rnn"},
            tags=["training"],
        ):
            yanex.log_metrics({"accuracy": 0.78, "loss": 0.22})
            yanex.log_metrics({"accuracy": 0.84, "loss": 0.16})
            yanex.log_metrics({"accuracy": 0.89, "loss": 0.11})

        yield

        # Cleanup
        plt.close("all")

    def test_plot_single_experiment_single_metric(self):
        """Test plotting single experiment with single metric."""
        experiments = yr.get_experiments(tags=["training"], limit=1)
        exp_id = experiments[0].id

        fig = yr.plot_metrics("accuracy", ids=[exp_id], show=False)

        assert fig is not None
        assert len(fig.axes) == 1
        assert fig.axes[0].get_ylabel() == "Accuracy"
        plt.close(fig)

    def test_plot_single_experiment_multiple_metrics(self):
        """Test plotting single experiment with multiple metrics."""
        experiments = yr.get_experiments(tags=["training"], limit=1)
        exp_id = experiments[0].id

        fig = yr.plot_metrics(["accuracy", "loss"], ids=[exp_id], show=False)

        assert fig is not None
        assert len(fig.axes) == 2  # 2 subplots for 2 metrics
        plt.close(fig)

    def test_plot_multiple_experiments_single_metric(self):
        """Test plotting multiple experiments with single metric."""
        fig = yr.plot_metrics("accuracy", tags=["training"], show=False)

        assert fig is not None
        assert len(fig.axes) == 1
        # Should have 3 lines (3 experiments)
        lines = fig.axes[0].get_lines()
        assert len(lines) == 3
        plt.close(fig)

    def test_plot_with_label_by(self):
        """Test plotting with custom label_by."""
        fig = yr.plot_metrics(
            "accuracy", tags=["training"], label_by="learning_rate", show=False
        )

        assert fig is not None
        # Check legend
        legend = fig.axes[0].get_legend()
        assert legend is not None
        plt.close(fig)

    def test_plot_with_subplot_by(self):
        """Test plotting with subplot_by."""
        fig = yr.plot_metrics(
            "accuracy",
            tags=["training"],
            label_by="learning_rate",
            subplot_by="model",
            show=False,
        )

        assert fig is not None
        # Should have 2 subplots (cnn, rnn)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_with_return_axes(self):
        """Test returning axes for customization."""
        fig, axes = yr.plot_metrics(
            "accuracy",
            tags=["training"],
            show=False,
            return_axes=True,
        )

        assert fig is not None
        assert axes is not None
        # Can customize axes
        axes.set_yscale("linear")
        plt.close(fig)

    def test_plot_no_experiments_found(self):
        """Test error when no experiments match filters."""
        with pytest.raises(ValueError, match="No experiments found"):
            yr.plot_metrics("accuracy", tags=["nonexistent"], show=False)

    def test_plot_metric_not_found(self):
        """Test error when metric doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            yr.plot_metrics("nonexistent_metric", tags=["training"], show=False)

    def test_plot_with_custom_styling(self):
        """Test plotting with custom styling options."""
        fig = yr.plot_metrics(
            "accuracy",
            tags=["training"],
            limit=1,
            title="Test Plot",
            xlabel="Epoch",
            ylabel="Validation Accuracy",
            figsize=(10, 6),
            grid=False,
            show=False,
        )

        assert fig is not None
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        assert fig.axes[0].get_title() == "Test Plot"
        assert fig.axes[0].get_xlabel() == "Epoch"
        plt.close(fig)

    def test_plot_with_subplot_layout(self):
        """Test specifying custom subplot layout."""
        fig = yr.plot_metrics(
            ["accuracy", "loss"],
            tags=["training"],
            limit=1,
            subplot_layout=(2, 1),  # 2 rows, 1 column
            show=False,
        )

        assert fig is not None
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_invalid_subplot_layout(self):
        """Test error with mismatched subplot layout."""
        with pytest.raises(ValueError, match="subplot_layout"):
            # 2 metrics need 2 subplots, but (2, 2) = 4
            yr.plot_metrics(
                ["accuracy", "loss"],
                tags=["training"],
                limit=1,
                subplot_layout=(2, 2),
                show=False,
            )


class TestVisualizationBuildingBlocks:
    """Tests for lower-level visualization building blocks."""

    @pytest.fixture(autouse=True)
    def setup_experiments(self, tmp_path, clean_git_repo):
        """Create sample experiments."""
        yanex.set_experiments_dir(tmp_path)

        with yanex.create_experiment(
            script_path="train.py",
            config={"lr": 0.001},
            tags=["test"],
        ):
            yanex.log_metrics({"acc": 0.8})
            yanex.log_metrics({"acc": 0.9})

    def test_extract_metrics_df(self):
        """Test extracting metrics to DataFrame."""
        from yanex.results.viz import extract_metrics_df

        experiments = yr.get_experiments(tags=["test"])
        df = extract_metrics_df(experiments, ["acc"])

        assert len(df) == 2  # 2 steps
        assert "acc" in df.columns
        assert "lr" in df.columns  # Auto-discovered params

    def test_organize_for_plotting(self):
        """Test organizing data for plotting."""
        from yanex.results.viz import extract_metrics_df, organize_for_plotting

        experiments = yr.get_experiments(tags=["test"])
        df = extract_metrics_df(experiments, ["acc"])
        result = organize_for_plotting(df, ["acc"])

        assert "plot_type" in result
        assert "subplots" in result
        assert "metadata" in result
        assert result["plot_type"] == "line"
