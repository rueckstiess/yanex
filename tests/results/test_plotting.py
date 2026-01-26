"""
Tests for metrics plotting utilities.

This module tests the plot_metrics function and its helper functions.
"""

import numpy as np
import pandas as pd
import pytest

from yanex.results.plotting import (
    _calculate_layout,
    _get_colors,
    _is_single_step_metric,
    _parse_group_sort_key,
    _resolve_column,
    _resolve_group_by,
    _sort_groups,
    plot_metrics,
)


class TestResolveColumn:
    """Test _resolve_column function for suffix-based column resolution."""

    @pytest.fixture
    def df_with_nested_cols(self):
        """Create a DataFrame with nested column names."""
        return pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2"],
                "step": [0, 0],
                "metric_name": ["loss", "loss"],
                "value": [1.0, 0.8],
                "origami.pipeline.lr": [0.01, 0.001],
                "origami.train.epochs": [10, 20],
                "name": ["run-a", "run-b"],
            }
        )

    def test_exact_match(self, df_with_nested_cols):
        """Exact column name is returned as-is."""
        result = _resolve_column(df_with_nested_cols, "origami.pipeline.lr")
        assert result == "origami.pipeline.lr"

    def test_suffix_match(self, df_with_nested_cols):
        """Short name resolves to full path via suffix matching."""
        result = _resolve_column(df_with_nested_cols, "lr")
        assert result == "origami.pipeline.lr"

    def test_suffix_match_epochs(self, df_with_nested_cols):
        """Another suffix match example."""
        result = _resolve_column(df_with_nested_cols, "epochs")
        assert result == "origami.train.epochs"

    def test_non_nested_column(self, df_with_nested_cols):
        """Non-nested column name works directly."""
        result = _resolve_column(df_with_nested_cols, "name")
        assert result == "name"

    def test_ambiguous_raises(self):
        """Ambiguous column name raises ValueError with helpful message."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1"],
                "step": [0],
                "metric_name": ["loss"],
                "value": [1.0],
                "model.lr": [0.01],
                "optimizer.lr": [0.001],
            }
        )
        with pytest.raises(ValueError, match="Ambiguous column 'lr'"):
            _resolve_column(df, "lr")

    def test_not_found_raises(self, df_with_nested_cols):
        """Missing column raises ValueError with available columns."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            _resolve_column(df_with_nested_cols, "nonexistent")

    def test_standard_cols_excluded(self, df_with_nested_cols):
        """Standard columns like experiment_id are excluded from resolution."""
        with pytest.raises(ValueError, match="not found"):
            _resolve_column(df_with_nested_cols, "experiment_id")


class TestResolveGroupBy:
    """Test _resolve_group_by function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp1", "exp2", "exp2"],
                "step": [0, 1, 0, 1],
                "metric_name": ["loss", "loss", "loss", "loss"],
                "value": [1.0, 0.8, 1.2, 0.9],
                "lr": [0.01, 0.01, 0.001, 0.001],
                "name": ["run-a", "run-a", "run-b", "run-b"],
            }
        )

    def test_none_groups_by_experiment_id(self, sample_df):
        """When group_by is None, each experiment is its own group."""
        result = _resolve_group_by(sample_df, None)
        assert list(result) == ["exp1", "exp1", "exp2", "exp2"]

    def test_single_column(self, sample_df):
        """Group by a single column."""
        result = _resolve_group_by(sample_df, "lr")
        assert list(result) == ["lr=0.01", "lr=0.01", "lr=0.001", "lr=0.001"]

    def test_multiple_columns(self, sample_df):
        """Group by multiple columns."""
        result = _resolve_group_by(sample_df, ["lr", "name"])
        assert result.iloc[0] == "lr=0.01, name=run-a"
        assert result.iloc[2] == "lr=0.001, name=run-b"

    def test_params_auto_detection(self, sample_df):
        """Group by 'params' auto-detects param columns."""
        result = _resolve_group_by(sample_df, "params")
        # Should group by 'lr' and 'name' (non-standard columns)
        assert "lr=" in result.iloc[0]
        assert "name=" in result.iloc[0]

    def test_params_with_no_extra_columns(self):
        """When 'params' but no extra columns, fall back to experiment_id."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2"],
                "step": [0, 0],
                "metric_name": ["loss", "loss"],
                "value": [1.0, 1.2],
            }
        )
        result = _resolve_group_by(df, "params")
        assert list(result) == ["exp1", "exp2"]

    def test_callable_grouping(self, sample_df):
        """Group by a custom callable."""
        result = _resolve_group_by(
            sample_df, lambda row: "high" if row["lr"] > 0.005 else "low"
        )
        assert list(result) == ["high", "high", "low", "low"]

    def test_missing_column_raises(self, sample_df):
        """Raise error if column not found."""
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            _resolve_group_by(sample_df, "nonexistent")

    def test_suffix_column_resolution(self):
        """Group by suffix-resolved column names."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp1", "exp2", "exp2"],
                "step": [0, 1, 0, 1],
                "metric_name": ["loss", "loss", "loss", "loss"],
                "value": [1.0, 0.8, 1.2, 0.9],
                "origami.train.lr": [0.01, 0.01, 0.001, 0.001],
            }
        )
        # Use "lr" instead of "origami.train.lr"
        result = _resolve_group_by(df, "lr")
        assert "lr=0.01" in result.iloc[0]
        assert "lr=0.001" in result.iloc[2]

    def test_shortens_dotted_names(self):
        """Column names with dots are shortened to last part."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1"],
                "step": [0],
                "metric_name": ["loss"],
                "value": [1.0],
                "train.optimizer.lr": [0.01],
            }
        )
        result = _resolve_group_by(df, ["train.optimizer.lr"])
        assert result.iloc[0] == "lr=0.01"


class TestGetColors:
    """Test _get_colors function."""

    def test_none_uses_default_cycle(self):
        """None returns matplotlib default color cycle."""
        colors = _get_colors(None, 5)
        assert colors == ["C0", "C1", "C2", "C3", "C4"]

    def test_string_uses_colormap(self):
        """String is treated as colormap name."""
        colors = _get_colors("viridis", 3)
        assert len(colors) == 3
        # Colors should be tuples (RGBA)
        assert all(len(c) == 4 for c in colors)

    def test_list_cycles_through(self):
        """List cycles through provided colors."""
        colors = _get_colors(["red", "blue"], 5)
        assert colors == ["red", "blue", "red", "blue", "red"]

    def test_single_group_colormap(self):
        """Single group with colormap gets middle color."""
        colors = _get_colors("viridis", 1)
        assert len(colors) == 1


class TestCalculateLayout:
    """Test _calculate_layout function."""

    def test_single_metric(self):
        """Single metric gets 1x1 layout."""
        rows, cols, figsize = _calculate_layout(1)
        assert (rows, cols) == (1, 1)
        assert figsize == (10, 6)

    def test_two_metrics(self):
        """Two metrics get 1x2 layout."""
        rows, cols, figsize = _calculate_layout(2)
        assert (rows, cols) == (1, 2)
        assert figsize == (14, 5)

    def test_three_metrics(self):
        """Three metrics get 2x2 layout."""
        rows, cols, figsize = _calculate_layout(3)
        assert (rows, cols) == (2, 2)

    def test_four_metrics(self):
        """Four metrics get 2x2 layout."""
        rows, cols, figsize = _calculate_layout(4)
        assert (rows, cols) == (2, 2)

    def test_five_metrics(self):
        """Five metrics get 2x3 layout."""
        rows, cols, figsize = _calculate_layout(5)
        assert (rows, cols) == (2, 3)

    def test_seven_metrics(self):
        """Seven metrics get 3x3 layout."""
        rows, cols, figsize = _calculate_layout(7)
        assert (rows, cols) == (3, 3)


class TestIsSingleStepMetric:
    """Test _is_single_step_metric function."""

    def test_multi_step_metric(self):
        """Multi-step metric returns False."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp1", "exp1"],
                "step": [0, 1, 2],
                "value": [1.0, 0.8, 0.6],
            }
        )
        assert not _is_single_step_metric(df)

    def test_single_step_metric(self):
        """Single-step metric returns True."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2"],
                "step": [0, 0],
                "value": [1.0, 0.8],
            }
        )
        assert _is_single_step_metric(df)

    def test_mixed_steps(self):
        """If any experiment has multiple steps, returns False."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp1", "exp2"],
                "step": [0, 1, 0],
                "value": [1.0, 0.8, 0.9],
            }
        )
        assert not _is_single_step_metric(df)


class TestParseGroupSortKey:
    """Test _parse_group_sort_key function for smart numeric sorting."""

    def test_single_numeric_param(self):
        """Single numeric parameter parsed correctly."""
        result = _parse_group_sort_key("lr=0.01")
        assert result == ((0, 0.01),)

    def test_single_string_param(self):
        """Single string parameter parsed correctly."""
        result = _parse_group_sort_key("model=resnet")
        assert result == ((1, "resnet"),)

    def test_multiple_numeric_params(self):
        """Multiple numeric parameters parsed correctly."""
        result = _parse_group_sort_key("lr=0.01, epochs=10")
        assert result == ((0, 0.01), (0, 10.0))

    def test_mixed_params(self):
        """Mixed numeric and string parameters parsed correctly."""
        result = _parse_group_sort_key("lr=0.01, model=resnet")
        assert result == ((0, 0.01), (1, "resnet"))

    def test_no_equals_sign(self):
        """Value without equals sign treated as string."""
        result = _parse_group_sort_key("exp1")
        assert result == ((1, "exp1"),)

    def test_sorting_behavior_numeric(self):
        """Numeric values sort numerically, not alphabetically."""
        groups = ["epochs=10", "epochs=2", "epochs=1"]
        sorted_groups = sorted(groups, key=_parse_group_sort_key)
        assert sorted_groups == ["epochs=1", "epochs=2", "epochs=10"]

    def test_sorting_behavior_float(self):
        """Float values sort correctly."""
        groups = ["lr=0.1", "lr=0.01", "lr=0.001"]
        sorted_groups = sorted(groups, key=_parse_group_sort_key)
        assert sorted_groups == ["lr=0.001", "lr=0.01", "lr=0.1"]

    def test_sorting_behavior_multi_param(self):
        """Multi-param groups sort by all params in order."""
        groups = ["lr=0.1, bs=32", "lr=0.01, bs=16", "lr=0.01, bs=32"]
        sorted_groups = sorted(groups, key=_parse_group_sort_key)
        # Primary sort by lr, secondary by bs
        assert sorted_groups == ["lr=0.01, bs=16", "lr=0.01, bs=32", "lr=0.1, bs=32"]


class TestSortGroups:
    """Test _sort_groups function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with groups and values."""
        return pd.DataFrame(
            {
                "_group": ["lr=0.1", "lr=0.1", "lr=0.01", "lr=0.01", "lr=0.001"],
                "value": [0.8, 0.85, 0.9, 0.95, 0.7],
            }
        )

    def test_sort_by_none_alphabetical(self, sample_df):
        """sort_by=None returns alphabetical order."""
        groups = ["lr=0.1", "lr=0.01", "lr=0.001"]
        result = _sort_groups(sample_df, groups, None)
        assert result == ["lr=0.001", "lr=0.01", "lr=0.1"]

    def test_sort_by_value_ascending(self, sample_df):
        """sort_by='value' sorts by mean value ascending."""
        groups = ["lr=0.1", "lr=0.01", "lr=0.001"]
        result = _sort_groups(sample_df, groups, "value")
        # lr=0.001: mean=0.7, lr=0.1: mean=0.825, lr=0.01: mean=0.925
        assert result == ["lr=0.001", "lr=0.1", "lr=0.01"]

    def test_sort_by_group_numeric(self, sample_df):
        """sort_by='group' uses smart numeric sorting."""
        groups = ["lr=0.1", "lr=0.01", "lr=0.001"]
        result = _sort_groups(sample_df, groups, "group")
        assert result == ["lr=0.001", "lr=0.01", "lr=0.1"]

    def test_sort_by_group_epochs(self):
        """sort_by='group' handles integer-like values."""
        df = pd.DataFrame(
            {
                "_group": ["epochs=10", "epochs=2", "epochs=1"],
                "value": [1.0, 1.0, 1.0],
            }
        )
        groups = ["epochs=10", "epochs=2", "epochs=1"]
        result = _sort_groups(df, groups, "group")
        assert result == ["epochs=1", "epochs=2", "epochs=10"]


class TestPlotMetrics:
    """Test plot_metrics function."""

    @pytest.fixture
    def multi_step_df(self):
        """DataFrame with multi-step metrics."""
        data = []
        for exp_id in ["exp1", "exp2"]:
            for step in range(10):
                data.append(
                    {
                        "experiment_id": exp_id,
                        "step": step,
                        "metric_name": "train_loss",
                        "value": 1.0 - step * 0.08 + (0.1 if exp_id == "exp2" else 0),
                        "lr": 0.01 if exp_id == "exp1" else 0.001,
                    }
                )
        return pd.DataFrame(data)

    @pytest.fixture
    def single_step_df(self):
        """DataFrame with single-step metrics (final accuracy)."""
        return pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2", "exp3"],
                "step": [0, 0, 0],
                "metric_name": ["accuracy", "accuracy", "accuracy"],
                "value": [0.92, 0.95, 0.88],
                "lr": [0.01, 0.001, 0.01],
            }
        )

    def test_missing_columns_raises(self):
        """Raises ValueError if required columns missing."""
        df = pd.DataFrame({"experiment_id": ["exp1"], "value": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_metrics(df)

    def test_empty_dataframe(self):
        """Empty DataFrame shows 'No data' message."""
        df = pd.DataFrame(columns=["experiment_id", "step", "metric_name", "value"])
        fig, axes = plot_metrics(df)
        assert len(axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_metric_line_chart(self, multi_step_df):
        """Multi-step metric creates line chart."""
        fig, axes = plot_metrics(multi_step_df)
        assert len(axes) == 1
        # Should have lines plotted
        assert len(axes[0].lines) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_metric_bar_chart(self, single_step_df):
        """Single-step metric creates bar chart."""
        fig, axes = plot_metrics(single_step_df)
        assert len(axes) == 1
        # Should have bars (patches) plotted
        assert len(axes[0].patches) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multiple_metrics(self, multi_step_df):
        """Multiple metrics create multiple subplots."""
        # Add another metric
        df2 = multi_step_df.copy()
        df2["metric_name"] = "val_loss"
        df2["value"] = df2["value"] + 0.1
        df = pd.concat([multi_step_df, df2])

        fig, axes = plot_metrics(df)
        assert len(axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_specific_metrics(self, multi_step_df):
        """Can specify which metrics to plot."""
        df2 = multi_step_df.copy()
        df2["metric_name"] = "val_loss"
        df = pd.concat([multi_step_df, df2])

        fig, axes = plot_metrics(df, metrics=["train_loss"])
        assert len(axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_group_by_column(self, multi_step_df):
        """Can group by a column."""
        fig, axes = plot_metrics(multi_step_df, group_by="lr")
        # Should have legend with lr values
        legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert any("lr=" in t for t in legend_texts)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_group_by_params(self, multi_step_df):
        """Can group by 'params' for auto-detection."""
        fig, axes = plot_metrics(multi_step_df, group_by="params")
        # Should have legend with lr values (the only param column)
        legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert any("lr=" in t for t in legend_texts)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_group_by_callable(self, multi_step_df):
        """Can group by a callable."""
        fig, axes = plot_metrics(
            multi_step_df, group_by=lambda row: "high" if row["lr"] > 0.005 else "low"
        )
        legend_texts = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert set(legend_texts) == {"high", "low"}
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_smooth_window(self, multi_step_df):
        """Smoothing applies to lines."""
        fig1, axes1 = plot_metrics(multi_step_df, smooth_window=None)
        fig2, axes2 = plot_metrics(multi_step_df, smooth_window=3)

        # Get y-data from first line
        y1 = axes1[0].lines[0].get_ydata()
        y2 = axes2[0].lines[0].get_ydata()

        # Smoothed line should be different
        assert not np.allclose(y1, y2)
        import matplotlib.pyplot as plt

        plt.close(fig1)
        plt.close(fig2)

    def test_show_individual_false(self, multi_step_df):
        """Can hide individual runs."""
        # Create a grouping where multiple experiments are in same group
        multi_step_df["group_col"] = "all"

        fig_with, axes_with = plot_metrics(
            multi_step_df, group_by="group_col", show_individual=True
        )
        fig_without, axes_without = plot_metrics(
            multi_step_df, group_by="group_col", show_individual=False
        )

        # With individual runs should have more lines
        assert len(axes_with[0].lines) > len(axes_without[0].lines)
        import matplotlib.pyplot as plt

        plt.close(fig_with)
        plt.close(fig_without)

    def test_custom_colors_list(self, multi_step_df):
        """Can provide custom color list."""
        fig, axes = plot_metrics(multi_step_df, colors=["red", "blue"])
        # Should use custom colors
        line_colors = [line.get_color() for line in axes[0].lines]
        assert "red" in line_colors or "blue" in line_colors
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_colors_colormap(self, multi_step_df):
        """Can provide colormap name."""
        fig, axes = plot_metrics(multi_step_df, colors="Set2")
        # Should not raise error
        assert len(axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_custom_figsize(self, multi_step_df):
        """Can provide custom figsize."""
        fig, axes = plot_metrics(multi_step_df, figsize=(20, 10))
        assert fig.get_figwidth() == 20
        assert fig.get_figheight() == 10
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_no_data_for_metric(self, multi_step_df):
        """Shows message when metric has no data."""
        fig, axes = plot_metrics(multi_step_df, metrics=["nonexistent"])
        # Axis should still exist
        assert len(axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_returns_fig_and_axes(self, multi_step_df):
        """Returns tuple of (Figure, array of Axes)."""
        fig, axes = plot_metrics(multi_step_df)

        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)
        assert isinstance(axes, np.ndarray)
        plt.close(fig)

    def test_bar_chart_with_groups(self, single_step_df):
        """Bar chart groups correctly."""
        fig, axes = plot_metrics(single_step_df, group_by="lr")
        # Should have 2 bars (2 unique lr values)
        assert len(axes[0].patches) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_bar_chart_shows_individual_points(self, single_step_df):
        """Bar chart shows individual points when multiple per group."""
        fig, axes = plot_metrics(single_step_df, group_by="lr", show_individual=True)
        # Should have scatter points (collections)
        collections = axes[0].collections
        # The group with lr=0.01 has 2 experiments, so should have points
        assert len(collections) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_sort_by_value_bar_chart(self):
        """sort_by='value' orders bars by metric value."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2", "exp3"],
                "step": [0, 0, 0],
                "metric_name": ["accuracy", "accuracy", "accuracy"],
                "value": [0.7, 0.9, 0.5],
                "group": ["B", "A", "C"],
            }
        )
        fig, axes = plot_metrics(df, group_by="group", sort_by="value")
        # Get x-axis labels (should be sorted by value: C, B, A)
        labels = [t.get_text() for t in axes[0].get_xticklabels()]
        assert labels == ["group=C", "group=B", "group=A"]
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_sort_by_group_bar_chart(self):
        """sort_by='group' uses smart numeric sorting."""
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2", "exp3"],
                "step": [0, 0, 0],
                "metric_name": ["accuracy", "accuracy", "accuracy"],
                "value": [0.9, 0.8, 0.7],
                "epochs": [10, 2, 1],
            }
        )
        fig, axes = plot_metrics(df, group_by="epochs", sort_by="group")
        # Get x-axis labels (should be sorted numerically: 1, 2, 10)
        labels = [t.get_text() for t in axes[0].get_xticklabels()]
        assert labels == ["epochs=1", "epochs=2", "epochs=10"]
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_sort_by_value_line_chart(self):
        """sort_by='value' orders legend by metric value."""
        data = []
        for exp_id, lr, base_val in [("exp1", 0.1, 0.9), ("exp2", 0.01, 0.7)]:
            for step in range(3):
                data.append(
                    {
                        "experiment_id": exp_id,
                        "step": step,
                        "metric_name": "loss",
                        "value": base_val - step * 0.1,
                        "lr": lr,
                    }
                )
        df = pd.DataFrame(data)
        fig, axes = plot_metrics(df, group_by="lr", sort_by="value")
        legend_labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        # lr=0.01 has lower mean value (0.6), lr=0.1 has higher (0.8)
        assert legend_labels == ["lr=0.01", "lr=0.1"]
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_sort_by_none_default(self, multi_step_df):
        """sort_by=None (default) uses alphabetical order."""
        fig, axes = plot_metrics(multi_step_df, group_by="lr", sort_by=None)
        legend_labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        # Alphabetical: "lr=0.001" < "lr=0.01"
        assert legend_labels == ["lr=0.001", "lr=0.01"]
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_colors_consistent_across_sort_modes(self):
        """Colors stay consistent when sort_by changes."""
        # Create data with 3 groups
        df = pd.DataFrame(
            {
                "experiment_id": ["exp1", "exp2", "exp3"],
                "step": [0, 0, 0],
                "metric_name": ["accuracy", "accuracy", "accuracy"],
                "value": [0.5, 0.9, 0.7],  # B has highest, A has lowest
                "group": ["A", "B", "C"],
            }
        )
        import matplotlib.pyplot as plt

        # Get colors with sort_by=None (alphabetical)
        fig1, axes1 = plot_metrics(df, group_by="group", sort_by=None)
        bars1 = axes1[0].patches
        colors_none = {
            axes1[0].get_xticklabels()[i].get_text(): bars1[i].get_facecolor()
            for i in range(len(bars1))
        }
        plt.close(fig1)

        # Get colors with sort_by="value"
        fig2, axes2 = plot_metrics(df, group_by="group", sort_by="value")
        bars2 = axes2[0].patches
        colors_value = {
            axes2[0].get_xticklabels()[i].get_text(): bars2[i].get_facecolor()
            for i in range(len(bars2))
        }
        plt.close(fig2)

        # Same group should have same color regardless of sort mode
        assert colors_none["group=A"] == colors_value["group=A"]
        assert colors_none["group=B"] == colors_value["group=B"]
        assert colors_none["group=C"] == colors_value["group=C"]
