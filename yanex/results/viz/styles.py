"""Styling utilities for yanex visualizations.

Provides color palettes, plot styles, and theming for publication-ready plots.
"""

import warnings

# Colorblind-safe palettes (from ColorBrewer and seaborn)
# 10-color palette (Tableau 10)
PALETTE_10 = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]

# 20-color palette (Tableau 20)
PALETTE_20 = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]

# Alternative colorblind-safe palette (for colorblind_safe=True)
COLORBLIND_PALETTE_10 = [
    "#0173B2",  # Blue
    "#DE8F05",  # Orange
    "#029E73",  # Green
    "#CC78BC",  # Pink
    "#CA9161",  # Brown
    "#949494",  # Gray
    "#ECE133",  # Yellow
    "#56B4E9",  # Light Blue
    "#F0E442",  # Light Yellow
    "#D55E00",  # Red-Orange
]


def get_color_palette(n_colors: int, colorblind_safe: bool = True) -> list[str]:
    """
    Get color palette for n_colors.

    Strategy:
    - n <= 10: Use 10-color palette
    - 10 < n <= 20: Use 20-color palette
    - n > 20: Use 20-color palette and warn about cycling

    Parameters
    ----------
    n_colors : int
        Number of colors needed
    colorblind_safe : bool, default=True
        Use colorblind-friendly palette

    Returns
    -------
    list[str]
        List of hex color codes

    Examples
    --------
    >>> colors = get_color_palette(5)
    >>> len(colors)
    5
    >>> colors = get_color_palette(25)  # doctest: +SKIP
    Warning: 25 unique labels but only 20 colors available...
    """
    if n_colors <= 0:
        return []

    # Select base palette
    if n_colors <= 10:
        palette = COLORBLIND_PALETTE_10 if colorblind_safe else PALETTE_10
    else:
        palette = PALETTE_20

    # Warn if cycling needed
    if n_colors > len(palette):
        warnings.warn(
            f"{n_colors} unique labels but only {len(palette)} colors available. "
            f"Colors will cycle. Consider using group_by or subplot_by to reduce "
            f"the number of lines per plot.",
            UserWarning,
            stacklevel=2,
        )

    # Cycle through palette if needed
    colors = []
    for i in range(n_colors):
        colors.append(palette[i % len(palette)])

    return colors


def get_plot_style() -> dict[str, str | float | bool]:
    """
    Get yanex default plot style.

    Returns dictionary suitable for plt.rcParams.update().

    The style is designed for:
    - Publication-ready figures
    - Clear readability on screen and in print
    - PDF/PNG export
    - White background with professional appearance

    Returns
    -------
    dict
        Matplotlib rcParams-compatible dictionary

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> plt.rcParams.update(get_plot_style())
    """
    return {
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        # Grid
        "grid.color": "#CCCCCC",
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Font
        "font.size": 10,
        "font.family": "sans-serif",
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.fontsize": 9,
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }


def apply_yanex_style() -> None:
    """
    Apply yanex style to all matplotlib plots.

    This sets matplotlib rcParams globally, affecting all subsequent plots.

    Examples
    --------
    >>> from yanex.results.viz import apply_yanex_style
    >>> apply_yanex_style()
    >>> # All matplotlib plots now use yanex style
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(get_plot_style())
