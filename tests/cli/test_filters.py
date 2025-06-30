"""
Tests for yanex CLI filtering functionality.
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from yanex.cli.filters import ExperimentFilter, parse_time_spec
from yanex.core.manager import ExperimentManager
from yanex.core.storage import ExperimentStorage


class TestExperimentFilter:
    """Test ExperimentFilter class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock manager with storage
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = ExperimentStorage(self.temp_dir)
        self.manager = Mock(spec=ExperimentManager)
        self.manager.storage = self.storage

        # Create filter with mocked manager
        self.filter = ExperimentFilter(self.manager)

        # Create test experiment metadata
        self.sample_experiments = [
            {
                "id": "exp12345",
                "name": "test-experiment-1",
                "status": "completed",
                "tags": ["ml", "training"],
                "started_at": "2025-06-28T10:00:00",
                "completed_at": "2025-06-28T10:05:00",
                "created_at": "2025-06-28T09:59:00",
            },
            {
                "id": "exp67890",
                "name": "test-experiment-2",
                "status": "failed",
                "tags": ["ml", "debug"],
                "started_at": "2025-06-28T11:00:00",
                "failed_at": "2025-06-28T11:02:00",
                "created_at": "2025-06-28T10:59:00",
            },
            {
                "id": "exp11111",
                "name": None,
                "status": "running",
                "tags": ["experiment"],
                "started_at": "2025-06-28T12:00:00",
                "created_at": "2025-06-28T11:59:00",
            },
            {
                "id": "exp22222",
                "name": "hyperparameter-tuning",
                "status": "completed",
                "tags": ["ml", "hyperopt", "training"],
                "started_at": "2025-06-27T15:00:00",
                "completed_at": "2025-06-27T15:30:00",
                "created_at": "2025-06-27T14:59:00",
            },
            {
                "id": "exp33333",
                "name": "baseline-model",
                "status": "cancelled",
                "tags": ["baseline"],
                "started_at": "2025-06-26T09:00:00",
                "cancelled_at": "2025-06-26T09:15:00",
                "created_at": "2025-06-26T08:59:00",
            },
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_filter_by_status_completed(self):
        """Test filtering by completed status."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(status="completed")

            assert len(results) == 2
            assert all(exp["status"] == "completed" for exp in results)
            assert "exp12345" in [exp["id"] for exp in results]
            assert "exp22222" in [exp["id"] for exp in results]

    def test_filter_by_status_failed(self):
        """Test filtering by failed status."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(status="failed")

            assert len(results) == 1
            assert results[0]["id"] == "exp67890"
            assert results[0]["status"] == "failed"

    def test_filter_by_status_running(self):
        """Test filtering by running status."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(status="running")

            assert len(results) == 1
            assert results[0]["id"] == "exp11111"
            assert results[0]["status"] == "running"

    def test_filter_by_invalid_status(self):
        """Test filtering with invalid status raises ValueError."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            with pytest.raises(ValueError, match="Invalid status 'invalid'"):
                self.filter.filter_experiments(status="invalid")

    def test_filter_by_name_pattern_exact(self):
        """Test filtering by exact name pattern."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(name_pattern="test-experiment-1")

            assert len(results) == 1
            assert results[0]["id"] == "exp12345"

    def test_filter_by_name_pattern_wildcard(self):
        """Test filtering by wildcard name pattern."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(name_pattern="test-*")

            assert len(results) == 2
            ids = [exp["id"] for exp in results]
            assert "exp12345" in ids
            assert "exp67890" in ids

    def test_filter_by_name_pattern_tuning(self):
        """Test filtering by name pattern containing 'tuning'."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(name_pattern="*tuning*")

            assert len(results) == 1
            assert results[0]["id"] == "exp22222"
            assert "tuning" in results[0]["name"]

    def test_filter_by_name_pattern_unnamed(self):
        """Test filtering includes unnamed experiments."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(name_pattern="*unnamed*")

            assert len(results) == 1
            assert results[0]["id"] == "exp11111"
            assert results[0]["name"] is None

    def test_filter_by_single_tag(self):
        """Test filtering by single tag."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(tags=["ml"])

            assert len(results) == 3
            ids = [exp["id"] for exp in results]
            assert "exp12345" in ids
            assert "exp67890" in ids
            assert "exp22222" in ids

    def test_filter_by_multiple_tags_and_logic(self):
        """Test filtering by multiple tags with AND logic."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(tags=["ml", "training"])

            assert len(results) == 2
            ids = [exp["id"] for exp in results]
            assert "exp12345" in ids
            assert "exp22222" in ids

            # Verify both tags are present in results
            for exp in results:
                assert "ml" in exp["tags"]
                assert "training" in exp["tags"]

    def test_filter_by_tags_no_matches(self):
        """Test filtering by tags with no matches."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(tags=["nonexistent"])

            assert len(results) == 0

    def test_filter_by_started_after(self):
        """Test filtering by started after time."""
        # Create a time that should match some experiments
        after_time = datetime(2025, 6, 28, 10, 30, 0, tzinfo=timezone.utc)

        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(started_after=after_time)

            # Should match experiments started at 11:00 and 12:00
            assert len(results) == 2
            ids = [exp["id"] for exp in results]
            assert "exp67890" in ids  # started at 11:00
            assert "exp11111" in ids  # started at 12:00

    def test_filter_by_started_before(self):
        """Test filtering by started before time."""
        # Create a time that should match some experiments
        before_time = datetime(2025, 6, 28, 10, 30, 0, tzinfo=timezone.utc)

        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(started_before=before_time)

            # Should match experiments started before 10:30
            assert len(results) == 3
            ids = [exp["id"] for exp in results]
            assert "exp12345" in ids  # started at 10:00
            assert "exp22222" in ids  # started at 15:00 on 27th (day before)
            assert "exp33333" in ids  # started at 09:00 on 26th

    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria combined."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(
                status="completed", tags=["ml"], name_pattern="*experiment*"
            )

            # Should match only test-experiment-1 (completed, has ml tag, name matches pattern)
            assert len(results) == 1
            assert results[0]["id"] == "exp12345"

    def test_filter_with_limit(self):
        """Test filtering with result limit."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(limit=2)

            assert len(results) == 2

    def test_filter_with_include_all_flag(self):
        """Test filtering with include_all flag."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(include_all=True)

            # Should return all experiments
            assert len(results) == 5

    def test_filter_default_limit(self):
        """Test default limit of 10 when no limit specified."""
        # Create more than 10 experiments
        many_experiments = []
        for i in range(15):
            many_experiments.append(
                {
                    "id": f"exp{i:05d}",
                    "name": f"experiment-{i}",
                    "status": "completed",
                    "tags": [],
                    "started_at": f"2025-06-28T{10 + i % 14:02d}:00:00",
                    "created_at": f"2025-06-28T{9 + i % 14:02d}:59:00",
                }
            )

        with patch.object(
            self.filter, "_load_all_experiments", return_value=many_experiments
        ):
            results = self.filter.filter_experiments()

            # Should apply default limit of 10
            assert len(results) == 10

    def test_experiments_sorted_by_creation_time(self):
        """Test that experiments are sorted by creation time (newest first)."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(include_all=True)

            # Verify sorted by created_at in descending order
            created_times = [exp["created_at"] for exp in results]
            assert created_times == sorted(created_times, reverse=True)

    def test_timezone_handling_in_time_filters(self):
        """Test proper timezone handling in time comparison methods."""
        # Test experiment with no timezone info
        exp_no_tz = {
            "id": "exp_no_tz",
            "name": "no-timezone",
            "status": "completed",
            "tags": [],
            "started_at": "2025-06-28T10:00:00",  # No timezone
            "created_at": "2025-06-28T09:59:00",
        }

        # Test with timezone-aware comparison time
        after_time = datetime(2025, 6, 28, 9, 30, 0, tzinfo=timezone.utc)

        with patch.object(
            self.filter, "_load_all_experiments", return_value=[exp_no_tz]
        ):
            results = self.filter.filter_experiments(started_after=after_time)

            # Should handle timezone conversion and find the experiment
            assert len(results) == 1
            assert results[0]["id"] == "exp_no_tz"


class TestTimeUtils:
    """Test time parsing utilities."""

    def test_parse_time_spec_natural_language(self):
        """Test parsing natural language time specifications."""
        # These tests depend on dateparser behavior
        result = parse_time_spec("1 hour ago")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_time_spec_iso_format(self):
        """Test parsing ISO format dates."""
        result = parse_time_spec("2025-01-01T12:00:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

    def test_parse_time_spec_invalid(self):
        """Test parsing invalid time specifications."""
        result = parse_time_spec("not a valid time")
        assert result is None

        result = parse_time_spec("")
        assert result is None

        result = parse_time_spec(None)
        assert result is None

    def test_parse_time_spec_with_timezone(self):
        """Test parsing time with timezone info."""
        result = parse_time_spec("2025-01-01T12:00:00+05:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_parse_time_spec_relative_days(self):
        """Test parsing relative day terms returns beginning of day."""
        from datetime import date, time, timezone

        # Test "today" returns beginning of today
        today_result = parse_time_spec("today")
        expected_today = datetime.combine(date.today(), time.min, tzinfo=timezone.utc)
        assert today_result == expected_today

        # Test "yesterday" returns beginning of yesterday
        yesterday_result = parse_time_spec("yesterday")
        expected_yesterday = datetime.combine(
            date.today() - timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        assert yesterday_result == expected_yesterday

        # Test "tomorrow" returns beginning of tomorrow
        tomorrow_result = parse_time_spec("tomorrow")
        expected_tomorrow = datetime.combine(
            date.today() + timedelta(days=1), time.min, tzinfo=timezone.utc
        )
        assert tomorrow_result == expected_tomorrow

        # Test case insensitive
        today_upper = parse_time_spec("TODAY")
        assert today_upper == expected_today
