"""Tests for datetime utilities module."""

from datetime import datetime, timezone

from yanex.utils.datetime_utils import (
    calculate_duration_seconds,
    ensure_timezone_aware,
    format_datetime_for_display,
    format_duration,
    format_relative_time,
    parse_iso_timestamp,
)


class TestParseIsoTimestamp:
    """Test ISO timestamp parsing functionality."""

    def test_parse_z_suffix(self):
        """Test parsing timestamp with Z suffix."""
        result = parse_iso_timestamp("2023-01-01T12:00:00Z")
        expected = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_parse_timezone_offset(self):
        """Test parsing timestamp with timezone offset."""
        result = parse_iso_timestamp("2023-01-01T12:00:00+05:00")
        assert result is not None
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12
        assert result.tzinfo is not None

    def test_parse_naive_timestamp(self):
        """Test parsing timestamp without timezone (assumes UTC)."""
        result = parse_iso_timestamp("2023-01-01T12:00:00")
        expected = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_parse_invalid_timestamp(self):
        """Test parsing invalid timestamp returns None."""
        assert parse_iso_timestamp("invalid") is None
        assert parse_iso_timestamp("") is None
        assert parse_iso_timestamp(None) is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert parse_iso_timestamp("") is None
        assert parse_iso_timestamp("   ") is None


class TestEnsureTimezoneAware:
    """Test timezone awareness utility."""

    def test_naive_datetime_gets_timezone(self):
        """Test that naive datetime gets UTC timezone."""
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)
        result = ensure_timezone_aware(naive_dt)
        assert result.tzinfo == timezone.utc

    def test_aware_datetime_unchanged(self):
        """Test that timezone-aware datetime is unchanged."""
        aware_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = ensure_timezone_aware(aware_dt)
        assert result == aware_dt
        assert result.tzinfo == timezone.utc


class TestCalculateDurationSeconds:
    """Test duration calculation utility."""

    def test_calculate_valid_duration(self):
        """Test calculating duration between valid timestamps."""
        start = "2023-01-01T12:00:00Z"
        end = "2023-01-01T12:02:34Z"
        result = calculate_duration_seconds(start, end)
        assert result == 154.0  # 2 minutes 34 seconds

    def test_calculate_invalid_timestamps(self):
        """Test calculating duration with invalid timestamps."""
        assert calculate_duration_seconds("invalid", "2023-01-01T12:00:00Z") is None
        assert calculate_duration_seconds("2023-01-01T12:00:00Z", "invalid") is None


class TestFormatDatetimeForDisplay:
    """Test datetime display formatting."""

    def test_format_valid_timestamp(self):
        """Test formatting valid timestamp for display."""
        result = format_datetime_for_display("2023-01-01T12:00:00Z")
        assert result == "2023-01-01 12:00:00"

    def test_format_invalid_timestamp(self):
        """Test formatting invalid timestamp returns original."""
        result = format_datetime_for_display("invalid")
        assert result == "invalid"


class TestFormatDuration:
    """Test duration formatting."""

    def test_format_short_duration(self):
        """Test formatting short duration."""
        start = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        result = format_duration(start, end)
        assert result == "30s"

    def test_format_medium_duration(self):
        """Test formatting medium duration."""
        start = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2023, 1, 1, 12, 2, 30, tzinfo=timezone.utc)
        result = format_duration(start, end)
        assert result == "2m 30s"

    def test_format_ongoing_duration(self):
        """Test formatting ongoing duration (no end time)."""
        start = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_duration(start, None)
        assert "+" in result


class TestFormatRelativeTime:
    """Test relative time formatting."""

    def test_format_recent_time(self):
        """Test formatting very recent time."""
        # Create a datetime that's just a few seconds ago
        import datetime as dt

        recent = datetime.now(timezone.utc) - dt.timedelta(seconds=30)
        result = format_relative_time(recent)
        assert result == "just now"

    def test_format_old_time(self):
        """Test formatting old time shows date."""
        old_time = datetime(2020, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_relative_time(old_time)
        assert "2020-01-01" in result
