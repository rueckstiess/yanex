"""Centralized theme constants for yanex CLI output.

This module defines all color, style, and formatting constants used across
CLI commands to ensure consistent output formatting.
"""

from rich import box

# =============================================================================
# Element Styles
# =============================================================================

# Column/field styles
ID_STYLE = "dim"
SCRIPT_STYLE = "dim cyan"
NAME_STYLE = "white"
TAGS_STYLE = "blue"
PARAMS_STYLE = "cyan"
METRICS_STYLE = "green"
TIMESTAMP_STYLE = "dim"

# Additional element styles
SLOT_STYLE = "cyan"  # Dependency slots: <data>, <model>
TARGET_STYLE = "yellow bold"  # Target experiment markers: <*>
DESCRIPTION_STYLE = "dim"  # Experiment descriptions
STEP_STYLE = "cyan"  # Step/index columns in tables
DURATION_STYLE = "dim"  # Duration display (when not status-colored)
LABEL_STYLE = "bold"  # Field labels (e.g., "Status:", "Created:")

# Message styles (for CLI feedback)
WARNING_STYLE = "yellow"
ERROR_STYLE = "red"
SUCCESS_STYLE = "green"
VERBOSE_STYLE = "dim"  # Verbose/debug output

# =============================================================================
# Status Colors and Symbols
# =============================================================================

STATUS_COLORS: dict[str, str] = {
    "completed": "green",
    "failed": "red",
    "running": "yellow",
    "created": "white",
    "cancelled": "bright_red",
    "staged": "cyan",
}

STATUS_SYMBOLS: dict[str, str] = {
    "completed": "✓",
    "failed": "✗",
    "running": "⚡",
    "created": "○",
    "cancelled": "✖",
    "staged": "⏲",
}

# =============================================================================
# Table Styles
# =============================================================================

# Use SIMPLE (borderless) for data tables
DATA_TABLE_BOX = box.SIMPLE

# Use ROUNDED for panel containers
PANEL_BOX = box.ROUNDED

# Header style for tables
TABLE_HEADER_STYLE = "bold"

# =============================================================================
# Operation Symbols
# =============================================================================

SUCCESS_SYMBOL = "✓"
FAILURE_SYMBOL = "✗"
WARNING_SYMBOL = "⚠️"
