"""
Time parsing utilities for human-readable date specifications.
"""

from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

import dateparser


def parse_time_spec(time_spec: str) -> Optional[datetime]:
    """
    Parse human-readable time specification into datetime object.

    Args:
        time_spec: Human-readable time string (e.g., "today", "2 hours ago", "2023-01-01")

    Returns:
        Parsed datetime object with timezone info, or None if parsing failed

    Examples:
        >>> parse_time_spec("today")
        datetime(2025, 6, 28, 0, 0, tzinfo=...)

        >>> parse_time_spec("2 hours ago")
        datetime(2025, 6, 28, 10, 0, tzinfo=...)

        >>> parse_time_spec("2023-01-01")
        datetime(2023, 1, 1, 0, 0, tzinfo=...)
    """
    if not time_spec or not time_spec.strip():
        return None

    time_spec = time_spec.strip().lower()

    try:
        # Handle special cases for relative day terms to return beginning of day
        if time_spec in ["today"]:
            today = date.today()
            return datetime.combine(today, time.min, tzinfo=timezone.utc)
        elif time_spec in ["yesterday"]:
            yesterday = date.today() - timedelta(days=1)
            return datetime.combine(yesterday, time.min, tzinfo=timezone.utc)
        elif time_spec in ["tomorrow"]:
            tomorrow = date.today() + timedelta(days=1)
            return datetime.combine(tomorrow, time.min, tzinfo=timezone.utc)

        # Use dateparser to handle natural language and various formats
        parsed_dt = dateparser.parse(
            time_spec,
            settings={
                "TIMEZONE": "local",  # Use local timezone
                "RETURN_AS_TIMEZONE_AWARE": True,  # Always return timezone-aware datetime
                "PREFER_DATES_FROM": "past",  # For ambiguous dates, prefer past
                "STRICT_PARSING": False,  # Allow flexible parsing
            },
        )

        if parsed_dt is None:
            return None

        # Ensure we have timezone info
        if parsed_dt.tzinfo is None:
            # Add local timezone if not present

            local_tz = timezone.utc.replace(tzinfo=timezone.utc).astimezone().tzinfo
            parsed_dt = parsed_dt.replace(tzinfo=local_tz)

        return parsed_dt

    except Exception:
        # Return None for any parsing errors
        return None


def format_duration(start_time: datetime, end_time: Optional[datetime] = None) -> str:
    """
    Format duration between two times in human-readable format.

    Args:
        start_time: Start datetime
        end_time: End datetime (if None, use current time)

    Returns:
        Human-readable duration string

    Examples:
        >>> format_duration(start, end)
        "2m 34s"

        >>> format_duration(start, None)  # Still running
        "5m 12s (ongoing)"
    """
    if end_time is None:
        end_time = datetime.now(timezone.utc)
        is_ongoing = True
    else:
        is_ongoing = False

    # Ensure both times have timezone info
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

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
        result += " (ongoing)"

    return result


def format_relative_time(dt: datetime) -> str:
    """
    Format datetime as relative time from now.

    Args:
        dt: Datetime to format

    Returns:
        Human-readable relative time string

    Examples:
        >>> format_relative_time(datetime.now() - timedelta(hours=2))
        "2 hours ago"

        >>> format_relative_time(datetime.now() - timedelta(days=1))
        "1 day ago"
    """
    now = datetime.now(timezone.utc)

    # Ensure dt has timezone info
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

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
