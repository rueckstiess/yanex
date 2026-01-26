"""Styled field formatters combining color + transformation.

This module provides consistent formatting functions for experiment fields,
combining styling (colors) with value transformations (e.g., relative timestamps).

These formatters are used by CLI commands to ensure consistent output across
all yanex commands when using the default output format.
"""

from datetime import UTC
from typing import Any

from rich.text import Text

from yanex.utils.datetime_utils import (
    format_datetime_for_display,
    format_duration,
    format_relative_time,
    parse_iso_timestamp,
)

from .theme import (
    DESCRIPTION_STYLE,
    DURATION_STYLE,
    FAILURE_SYMBOL,
    ID_STYLE,
    NAME_STYLE,
    SCRIPT_STYLE,
    SLOT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    SUCCESS_STYLE,
    SUCCESS_SYMBOL,
    TAGS_STYLE,
    TARGET_STYLE,
    TIMESTAMP_STYLE,
    VERBOSE_STYLE,
    WARNING_STYLE,
    WARNING_SYMBOL,
)

# =============================================================================
# Internal Helpers
# =============================================================================


def _get_exp_value(exp: dict, key: str, default: Any = None) -> Any:
    """Get a value from experiment dict, handling both prefixed and unprefixed keys.

    Comparison data uses prefixed keys (meta:id, meta:status), while list data
    uses unprefixed keys (id, status). This helper checks both formats.

    Args:
        exp: Experiment data dictionary.
        key: Key name without prefix (e.g., "id", "status", "started_at").
        default: Default value if key not found.

    Returns:
        Value from experiment dict or default.
    """
    prefixed_key = f"meta:{key}"
    if prefixed_key in exp:
        return exp[prefixed_key]
    return exp.get(key, default)


# =============================================================================
# Text Utilities
# =============================================================================


def truncate_middle(text: str, max_width: int) -> str:
    """Truncate text in the middle, preserving start and end.

    Useful for long names where both prefix and suffix are important,
    like nested parameter names (e.g., "optimizer.learning_rate" -> "optim...rate").

    Args:
        text: Text to truncate.
        max_width: Maximum width including ellipsis.

    Returns:
        Truncated text with "..." in the middle, or original if short enough.
    """
    if len(text) <= max_width:
        return text

    if max_width < 5:
        # Too short for meaningful truncation
        return text[:max_width]

    # Reserve 3 chars for "..."
    available = max_width - 3
    # Split: ~60% at start, ~40% at end (slightly favor start for readability)
    start_len = (available * 3) // 5
    end_len = available - start_len

    return text[:start_len] + "..." + text[-end_len:]


# =============================================================================
# Field Formatters (return Rich Text objects)
# =============================================================================


def format_experiment_id(exp_id: str) -> Text:
    """Format experiment ID with consistent styling.

    Args:
        exp_id: 8-character hex experiment ID.

    Returns:
        Styled Text object (dim gray).
    """
    return Text(exp_id, style=ID_STYLE)


def format_experiment_name(name: str | None, fallback: str = "(unnamed)") -> Text:
    """Format experiment name, handling None/empty.

    Args:
        name: Experiment name or None.
        fallback: Text to show when name is empty.

    Returns:
        Styled Text object (white, or dim for fallback).
    """
    if not name:
        return Text(fallback, style="dim")
    return Text(name, style=NAME_STYLE)


def format_script(script_path: str | None) -> Text:
    """Format script name from full path.

    Extracts just the filename from the full path.

    Args:
        script_path: Full path to script or None.

    Returns:
        Styled Text object (dim cyan) or dim "-" if empty.
    """
    from pathlib import Path

    if not script_path:
        return Text("-", style="dim")

    script_name = Path(script_path).name
    return Text(script_name, style=SCRIPT_STYLE)


def format_status(status: str, include_symbol: bool = True) -> Text:
    """Format status with symbol and color.

    Args:
        status: Experiment status (completed, failed, running, etc.).
        include_symbol: Whether to include the status symbol.

    Returns:
        Styled Text object with appropriate color.
    """
    color = STATUS_COLORS.get(status, "white")
    if include_symbol:
        symbol = STATUS_SYMBOLS.get(status, "?")
        return Text(f"{symbol} {status}", style=color)
    return Text(status, style=color)


def format_status_symbol(status: str) -> Text:
    """Format just the status symbol with color.

    Args:
        status: Experiment status.

    Returns:
        Styled Text with just the symbol.
    """
    symbol = STATUS_SYMBOLS.get(status, "?")
    color = STATUS_COLORS.get(status, "white")
    return Text(symbol, style=color)


def format_timestamp_relative(iso_timestamp: str | None) -> Text:
    """Format timestamp as relative time ('3 hours ago').

    Args:
        iso_timestamp: ISO format timestamp string.

    Returns:
        Styled Text object with relative time (dim style).
    """
    if not iso_timestamp:
        return Text("-", style="dim")

    dt = parse_iso_timestamp(iso_timestamp)
    if dt:
        relative = format_relative_time(dt)
        return Text(relative, style=TIMESTAMP_STYLE)
    return Text(iso_timestamp, style=TIMESTAMP_STYLE)


def format_timestamp_absolute(iso_timestamp: str | None) -> Text:
    """Format timestamp as absolute time ('2023-01-01 12:00:00').

    Args:
        iso_timestamp: ISO format timestamp string.

    Returns:
        Styled Text object with formatted datetime (dim style).
    """
    if not iso_timestamp:
        return Text("-", style="dim")

    display = format_datetime_for_display(iso_timestamp)
    return Text(display, style=TIMESTAMP_STYLE)


def format_duration_styled(
    start_time: str | None,
    end_time: str | None,
    status: str,
) -> Text:
    """Format duration with status-based coloring.

    Colors: green=completed, yellow=running, red=failed/cancelled.

    Args:
        start_time: ISO format start timestamp.
        end_time: ISO format end timestamp (None for running experiments).
        status: Experiment status for color selection.

    Returns:
        Styled Text object with duration.
    """
    if not start_time:
        return Text("-", style="dim")

    style = STATUS_COLORS.get(status, DURATION_STYLE)

    start_dt = parse_iso_timestamp(start_time)
    end_dt = parse_iso_timestamp(end_time) if end_time else None

    if start_dt:
        duration_str = format_duration(start_dt, end_dt)
        return Text(duration_str, style=style)
    return Text("-", style="dim")


def format_experiment_duration(experiment: dict) -> Text:
    """Format experiment duration from experiment metadata dict.

    Handles the logic of determining end time from various fields
    (ended_at, completed_at, failed_at, cancelled_at) and colors
    based on status.

    Supports both prefixed (meta:started_at) and unprefixed (started_at) keys.

    Args:
        experiment: Experiment metadata dictionary with started_at, status,
                   and optional end time fields.

    Returns:
        Styled Text object with duration.
    """
    from datetime import datetime

    started_at = _get_exp_value(experiment, "started_at")
    status = _get_exp_value(experiment, "status", "")

    # Try different possible end time fields (check both prefixed and unprefixed)
    ended_at = (
        _get_exp_value(experiment, "ended_at")
        or _get_exp_value(experiment, "completed_at")
        or _get_exp_value(experiment, "failed_at")
        or _get_exp_value(experiment, "cancelled_at")
    )

    if not started_at:
        return Text("-", style="dim")

    start_time = parse_iso_timestamp(started_at)
    if start_time is None:
        return Text("-", style="dim")

    end_time = None
    if ended_at:
        end_time = parse_iso_timestamp(ended_at)
    elif status == "running":
        # For running experiments, end_time stays None to show "(ongoing)"
        pass
    else:
        # For non-running experiments without end time, use current time
        end_time = datetime.now(UTC)

    duration_str = format_duration(start_time, end_time)

    # Color coding based on status
    style = STATUS_COLORS.get(status, DURATION_STYLE)
    return Text(duration_str, style=style)


def format_tags(tags: list[str] | None, max_tags: int = 3, max_width: int = 23) -> Text:
    """Format tags list with optional truncation.

    Args:
        tags: List of tag strings.
        max_tags: Maximum number of tags to show before truncating.
        max_width: Maximum total width before truncating.

    Returns:
        Styled Text object (blue) or dim "-" if empty.
    """
    if not tags:
        return Text("-", style="dim")

    # Limit displayed tags
    display_tags = tags[:max_tags]

    if len(tags) > max_tags:
        tag_str = ", ".join(display_tags) + f" (+{len(tags) - max_tags})"
    else:
        tag_str = ", ".join(display_tags)

    # Truncate if still too long
    if len(tag_str) > max_width:
        tag_str = tag_str[: max_width - 3] + "..."

    return Text(tag_str, style=TAGS_STYLE)


def format_slot_name(slot: str) -> Text:
    """Format dependency slot name like <data>, <model>.

    Args:
        slot: Slot name without angle brackets.

    Returns:
        Styled Text object (cyan).
    """
    return Text(f"<{slot}>", style=SLOT_STYLE)


def format_target_marker() -> Text:
    """Format target experiment marker.

    Returns:
        Styled Text object "<*>" (magenta).
    """
    return Text("<*>", style=TARGET_STYLE)


def format_description(description: str | None) -> Text:
    """Format experiment description.

    Args:
        description: Description text or None.

    Returns:
        Styled Text object (dim) or dim "-" if empty.
    """
    if not description:
        return Text("-", style="dim")
    return Text(description, style=DESCRIPTION_STYLE)


# =============================================================================
# Message Formatters (return Rich markup strings)
# =============================================================================


def format_success_message(message: str) -> str:
    """Format success message with Rich markup.

    Args:
        message: Message text.

    Returns:
        Rich markup string with green color and success symbol.
    """
    return f"[{SUCCESS_STYLE}]{SUCCESS_SYMBOL} {message}[/{SUCCESS_STYLE}]"


def format_error_message(message: str) -> str:
    """Format error message with Rich markup.

    Args:
        message: Message text.

    Returns:
        Rich markup string with red color and failure symbol.
    """
    return f"[{STATUS_COLORS['failed']}]{FAILURE_SYMBOL} {message}[/{STATUS_COLORS['failed']}]"


def format_warning_message(message: str) -> str:
    """Format warning message with Rich markup.

    Args:
        message: Message text.

    Returns:
        Rich markup string with yellow color and warning symbol.
    """
    return f"[{WARNING_STYLE}]{WARNING_SYMBOL} {message}[/{WARNING_STYLE}]"


def format_verbose(message: str) -> str:
    """Format verbose/debug message with Rich markup.

    Args:
        message: Message text.

    Returns:
        Rich markup string with dim style.
    """
    return f"[{VERBOSE_STYLE}]{message}[/{VERBOSE_STYLE}]"


def format_cancelled_message(message: str) -> str:
    """Format cancellation message with Rich markup.

    Args:
        message: Message text.

    Returns:
        Rich markup string with cancelled status color and symbol.
    """
    color = STATUS_COLORS["cancelled"]
    symbol = STATUS_SYMBOLS["cancelled"]
    return f"[{color}]{symbol} {message}[/{color}]"
