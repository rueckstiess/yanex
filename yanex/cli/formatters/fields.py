"""Styled field formatters combining color + transformation.

This module provides consistent formatting functions for experiment fields,
combining styling (colors) with value transformations (e.g., relative timestamps).

These formatters are used by CLI commands to ensure consistent output across
all yanex commands when using the default output format.
"""

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


def format_tags(tags: list[str] | None) -> Text:
    """Format tags list.

    Args:
        tags: List of tag strings.

    Returns:
        Styled Text object (blue) or dim "-" if empty.
    """
    if not tags:
        return Text("-", style="dim")
    return Text(", ".join(tags), style=TAGS_STYLE)


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
