"""Visualization utilities for yanex experiment results.

This module provides building blocks for visualizing experiment metrics:
- Data extraction: extract_metrics_df()
- Organization: organize_for_plotting()
- Plotting: Individual plotting functions
- Styling: Color palettes and plot styles

Example
-------
Low-level API (building blocks):
    >>> from yanex.results.viz import extract_metrics_df, organize_for_plotting
    >>> import yanex.results as yr
    >>> experiments = yr.get_experiments(tags=["training"])
    >>> df = extract_metrics_df(experiments, ["accuracy", "loss"])
    >>> # Use pandas operations on df
    >>> df_filtered = df[df['step'] < 100]

High-level API (convenience):
    >>> import yanex.results as yr
    >>> yr.plot_metrics("accuracy", tags=["training"])
"""

from yanex.results.viz.data import extract_metrics_df
from yanex.results.viz.grouping import organize_for_plotting
from yanex.results.viz.styles import (
    apply_yanex_style,
    get_color_palette,
    get_plot_style,
)

__all__ = [
    "extract_metrics_df",
    "organize_for_plotting",
    "get_color_palette",
    "get_plot_style",
    "apply_yanex_style",
]
