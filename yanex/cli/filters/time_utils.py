"""
Time parsing utilities for human-readable date specifications.
"""

from datetime import UTC, date, datetime, time, timedelta

import dateparser

from ...utils.datetime_utils import (
    format_duration as _format_duration,
)
from ...utils.datetime_utils import (
    format_relative_time as _format_relative_time,
)


def parse_time_spec(time_spec: str) -> datetime | None:
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
            return datetime.combine(today, time.min, tzinfo=UTC)
        elif time_spec in ["yesterday"]:
            yesterday = date.today() - timedelta(days=1)
            return datetime.combine(yesterday, time.min, tzinfo=UTC)
        elif time_spec in ["tomorrow"]:
            tomorrow = date.today() + timedelta(days=1)
            return datetime.combine(tomorrow, time.min, tzinfo=UTC)

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

            local_tz = UTC.replace(tzinfo=UTC).astimezone().tzinfo
            parsed_dt = parsed_dt.replace(tzinfo=local_tz)

        return parsed_dt

    except Exception:
        # Return None for any parsing errors
        return None


# Delegate to centralized datetime utilities for consistency
def format_duration(start_time: datetime, end_time: datetime | None = None) -> str:
    """Format duration between two times in human-readable format.

    This function delegates to the centralized datetime utilities.
    See yanex.utils.datetime_utils.format_duration for full documentation.
    """
    return _format_duration(start_time, end_time)


def format_relative_time(dt: datetime) -> str:
    """Format datetime as relative time from now.

    This function delegates to the centralized datetime utilities.
    See yanex.utils.datetime_utils.format_relative_time for full documentation.
    """
    return _format_relative_time(dt)
