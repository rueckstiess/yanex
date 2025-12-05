"""Centralized date/time parsing and formatting utilities for Yanex."""

from datetime import UTC, datetime, timezone


def parse_iso_timestamp(timestamp: str) -> datetime | None:
    """Parse ISO format timestamp with proper timezone handling.

    Handles various ISO timestamp formats commonly found in experiment data:
    - ISO with Z suffix (Zulu time): "2023-01-01T12:00:00Z"
    - ISO with timezone offset: "2023-01-01T12:00:00+00:00"
    - ISO without timezone (assumes UTC): "2023-01-01T12:00:00"

    Args:
        timestamp: ISO format timestamp string

    Returns:
        Parsed datetime object with timezone info, or None if parsing failed

    Examples:
        >>> parse_iso_timestamp("2023-01-01T12:00:00Z")
        datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)

        >>> parse_iso_timestamp("2023-01-01T12:00:00+05:00")
        datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone(datetime.timedelta(seconds=18000)))

        >>> parse_iso_timestamp("2023-01-01T12:00:00")
        datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if not timestamp or not isinstance(timestamp, str):
        return None

    timestamp = timestamp.strip()
    if not timestamp:
        return None

    try:
        # Handle Z suffix (Zulu time / UTC)
        if timestamp.endswith("Z"):
            return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        # Handle explicit timezone offset (e.g., +00:00, -05:00)
        elif (
            "+" in timestamp or timestamp.count("-") > 2
        ):  # Account for date separators
            return datetime.fromisoformat(timestamp)

        # No timezone info - assume UTC for consistency
        else:
            naive_dt = datetime.fromisoformat(timestamp)
            return naive_dt.replace(tzinfo=UTC)

    except (ValueError, TypeError):
        return None


def ensure_timezone_aware(dt: datetime, default_tz: timezone = UTC) -> datetime:
    """Ensure datetime object has timezone information.

    Args:
        dt: Datetime object that may or may not have timezone info
        default_tz: Default timezone to apply if none present (defaults to UTC)

    Returns:
        Timezone-aware datetime object

    Examples:
        >>> naive_dt = datetime(2023, 1, 1, 12, 0)
        >>> ensure_timezone_aware(naive_dt)
        datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz)
    return dt


def format_duration(start_time: datetime, end_time: datetime | None = None) -> str:
    """Format duration between two times in human-readable format.

    Args:
        start_time: Start datetime
        end_time: End datetime (if None, use current time)

    Returns:
        Human-readable duration string

    Examples:
        >>> start = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
        >>> end = datetime(2023, 1, 1, 12, 2, 34, tzinfo=timezone.utc)
        >>> format_duration(start, end)
        "2m 34s"

        >>> format_duration(start, None)  # Still running
        "+ 5m 12s"
    """
    if end_time is None:
        end_time = datetime.now(UTC)
        is_ongoing = True
    else:
        is_ongoing = False

    # Ensure both times have timezone info
    start_time = ensure_timezone_aware(start_time)
    end_time = ensure_timezone_aware(end_time)

    # Calculate duration
    duration = end_time - start_time
    total_seconds = int(duration.total_seconds())

    if total_seconds < 0:
        return "0s"

    # Format as human-readable
    if total_seconds < 60:
        result = f"{total_seconds}s"
    elif total_seconds < 3600:  # Less than 1 hour
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        result = f"{minutes}m {seconds}s"
    elif total_seconds < 86400:  # Less than 1 day
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        result = f"{hours}h {minutes}m"
    else:  # 1 day or more
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        result = f"{days}d {hours}h"

    if is_ongoing:
        result = "+ " + result

    return result


def format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time from now.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable relative time string

    Examples:
        >>> past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        >>> format_relative_time(past_time)
        "2 hours ago"

        >>> old_time = datetime.now(timezone.utc) - timedelta(days=30)
        >>> format_relative_time(old_time)
        "2023-01-01"  # Shows actual date for old times
    """
    now = datetime.now(UTC)

    # Ensure dt has timezone info
    dt = ensure_timezone_aware(dt)

    # Calculate difference
    diff = now - dt
    total_seconds = int(diff.total_seconds())

    if total_seconds < 0:
        return "in the future"

    if total_seconds < 60:
        return "just now"
    elif total_seconds < 3600:  # Less than 1 hour
        minutes = total_seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif total_seconds < 86400:  # Less than 1 day
        hours = total_seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif total_seconds < 604800:  # Less than 1 week
        days = total_seconds // 86400
        return f"{days} day{'s' if days != 1 else ''} ago"
    else:
        # For longer periods, show the actual date
        return dt.strftime("%Y-%m-%d")


def calculate_duration_seconds(start_str: str, end_str: str) -> float | None:
    """Calculate duration in seconds between two ISO timestamp strings.

    Args:
        start_str: Start time as ISO timestamp string
        end_str: End time as ISO timestamp string

    Returns:
        Duration in seconds, or None if parsing failed

    Examples:
        >>> calculate_duration_seconds("2023-01-01T12:00:00Z", "2023-01-01T12:02:34Z")
        154.0
    """
    start_dt = parse_iso_timestamp(start_str)
    end_dt = parse_iso_timestamp(end_str)

    if start_dt is None or end_dt is None:
        return None

    return (end_dt - start_dt).total_seconds()


def format_datetime_for_display(dt_str: str) -> str:
    """Format ISO timestamp string for human-readable display.

    Args:
        dt_str: ISO format datetime string

    Returns:
        Formatted datetime string for display

    Examples:
        >>> format_datetime_for_display("2023-01-01T12:00:00Z")
        "2023-01-01 12:00:00"
    """
    dt = parse_iso_timestamp(dt_str)
    if dt is None:
        return dt_str  # Return original if parsing failed

    # Format for display (remove timezone info for cleaner look)
    return dt.strftime("%Y-%m-%d %H:%M:%S")
