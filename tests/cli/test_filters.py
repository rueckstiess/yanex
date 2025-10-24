"""
Tests for yanex CLI filtering functionality.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from tests.test_utils import TestDataFactory, TestFileHelpers
from yanex.cli.filters import ExperimentFilter, parse_time_spec
from yanex.core.manager import ExperimentManager


class TestExperimentFilter:
    """Test ExperimentFilter class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test storage using utilities
        self.storage = TestFileHelpers.create_temp_storage()
        self.manager = Mock(spec=ExperimentManager)
        self.manager.storage = self.storage

        # Create filter with mocked manager
        self.filter = ExperimentFilter(self.manager)

        # Create test experiment metadata using factory
        self.sample_experiments = [
            TestDataFactory.create_experiment_metadata(
                experiment_id="exp12345",
                name="test-experiment-1",
                status="completed",
                tags=["ml", "training"],
                started_at="2025-06-28T10:00:00",
                completed_at="2025-06-28T10:05:00",
                created_at="2025-06-28T09:59:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="exp67890",
                name="test-experiment-2",
                status="failed",
                tags=["ml", "debug"],
                started_at="2025-06-28T11:00:00",
                failed_at="2025-06-28T11:02:00",
                created_at="2025-06-28T10:59:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="exp11111",
                name=None,
                status="running",
                tags=["experiment"],
                started_at="2025-06-28T12:00:00",
                created_at="2025-06-28T11:59:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="exp22222",
                name="hyperparameter-tuning",
                status="completed",
                tags=["ml", "hyperopt", "training"],
                started_at="2025-06-27T15:00:00",
                completed_at="2025-06-27T15:30:00",
                created_at="2025-06-27T14:59:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="exp33333",
                name="baseline-model",
                status="cancelled",
                tags=["baseline"],
                started_at="2025-06-26T09:00:00",
                cancelled_at="2025-06-26T09:15:00",
                created_at="2025-06-26T08:59:00",
            ),
        ]

    @pytest.mark.parametrize(
        "status,expected_count,expected_ids",
        [
            ("completed", 2, ["exp12345", "exp22222"]),
            ("failed", 1, ["exp67890"]),
            ("running", 1, ["exp11111"]),
            ("cancelled", 1, ["exp33333"]),
        ],
    )
    def test_filter_by_status(self, status, expected_count, expected_ids):
        """Test filtering by various status values."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(status=status)

            assert len(results) == expected_count
            assert all(exp["status"] == status for exp in results)
            result_ids = [exp["id"] for exp in results]
            for exp_id in expected_ids:
                assert exp_id in result_ids

    def test_filter_by_invalid_status(self):
        """Test filtering with invalid status raises ValueError."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            with pytest.raises(ValueError, match="Invalid status 'invalid'"):
                self.filter.filter_experiments(status="invalid")

    @pytest.mark.parametrize(
        "name_pattern,expected_count,expected_ids",
        [
            ("test-experiment-1", 1, ["exp12345"]),
            ("test-*", 2, ["exp12345", "exp67890"]),
            ("*tuning*", 1, ["exp22222"]),
            ("*unnamed*", 1, ["exp11111"]),  # For None names
            ("nonexistent", 0, []),
        ],
    )
    def test_filter_by_name_pattern(self, name_pattern, expected_count, expected_ids):
        """Test filtering by various name patterns."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(name=name_pattern)

            assert len(results) == expected_count
            result_ids = [exp["id"] for exp in results]
            for exp_id in expected_ids:
                assert exp_id in result_ids

    def test_filter_by_empty_name_pattern(self):
        """Test filtering by empty pattern matches only unnamed experiments."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            # Test empty string pattern
            results = self.filter.filter_experiments(name="")

            # Should only match the unnamed experiment (exp11111)
            assert len(results) == 1
            assert results[0]["id"] == "exp11111"
            assert results[0].get("name") is None

    def test_filter_empty_name_edge_cases(self):
        """Test edge cases for empty name filtering."""
        # Create additional test cases including empty string names
        test_experiments = [
            {
                **TestDataFactory.create_experiment_metadata(
                    experiment_id="exp_none", status="completed"
                ),
                # Explicitly omit name key to simulate None name
            },
            {
                **TestDataFactory.create_experiment_metadata(
                    experiment_id="exp_empty", status="completed"
                ),
                "name": "",  # Empty string name
            },
            {
                **TestDataFactory.create_experiment_metadata(
                    experiment_id="exp_named", status="completed"
                ),
                "name": "actual-name",
            },
        ]

        with patch.object(
            self.filter, "_load_all_experiments", return_value=test_experiments
        ):
            # Empty pattern should match both None and empty string names
            results = self.filter.filter_experiments(name="")

            assert len(results) == 2
            result_ids = [exp["id"] for exp in results]
            assert "exp_none" in result_ids
            assert "exp_empty" in result_ids
            assert "exp_named" not in result_ids

    def test_filter_unnamed_vs_named_patterns(self):
        """Test that unnamed filtering works correctly vs named filtering."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            # Empty pattern - should get only unnamed
            unnamed_results = self.filter.filter_experiments(name="")
            assert len(unnamed_results) == 1
            assert unnamed_results[0].get("name") is None

            # Pattern that matches named experiments
            named_results = self.filter.filter_experiments(name="test-*")
            assert len(named_results) == 2
            assert all(exp["name"] is not None for exp in named_results)
            assert all(exp["name"].startswith("test-") for exp in named_results)

            # Pattern that should match unnamed (alternative method)
            unnamed_alt_results = self.filter.filter_experiments(name="*unnamed*")
            assert len(unnamed_alt_results) == 1
            assert unnamed_alt_results[0].get("name") is None

    @pytest.mark.parametrize(
        "tags,expected_count,expected_ids",
        [
            (["ml"], 3, ["exp12345", "exp67890", "exp22222"]),
            (["ml", "training"], 2, ["exp12345", "exp22222"]),
            (["nonexistent"], 0, []),
            (["baseline"], 1, ["exp33333"]),
        ],
    )
    def test_filter_by_tags(self, tags, expected_count, expected_ids):
        """Test filtering by various tag combinations."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(tags=tags)

            assert len(results) == expected_count
            result_ids = [exp["id"] for exp in results]
            for exp_id in expected_ids:
                assert exp_id in result_ids

            # Verify all tags are present in results (AND logic)
            if tags and expected_count > 0:
                for exp in results:
                    for tag in tags:
                        assert tag in exp["tags"]

    @pytest.mark.parametrize(
        "time_filter,filter_time,expected_ids",
        [
            # started_after 10:30 should match 11:00 and 12:00
            (
                "started_after",
                datetime(2025, 6, 28, 10, 30, 0, tzinfo=timezone.utc),
                ["exp67890", "exp11111"],
            ),
            # started_before 10:30 should match 10:00, 15:00 on 27th, and 09:00 on 26th
            (
                "started_before",
                datetime(2025, 6, 28, 10, 30, 0, tzinfo=timezone.utc),
                ["exp12345", "exp22222", "exp33333"],
            ),
        ],
    )
    def test_filter_by_time(self, time_filter, filter_time, expected_ids):
        """Test filtering by time ranges."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            kwargs = {time_filter: filter_time}
            results = self.filter.filter_experiments(**kwargs)

            result_ids = [exp["id"] for exp in results]
            assert len(result_ids) == len(expected_ids)
            for exp_id in expected_ids:
                assert exp_id in result_ids

    def test_filter_combined_criteria(self):
        """Test filtering with multiple criteria combined."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            results = self.filter.filter_experiments(
                status="completed", tags=["ml"], name="*experiment*"
            )

            # Should match only test-experiment-1 (completed, has ml tag, name matches pattern)
            assert len(results) == 1
            assert results[0]["id"] == "exp12345"

    @pytest.mark.parametrize(
        "limit,include_all,expected_count",
        [
            (2, False, 2),  # Explicit limit
            (None, True, 5),  # Include all flag
            (None, False, 5),  # Default for small dataset
        ],
    )
    def test_filter_with_limits(self, limit, include_all, expected_count):
        """Test filtering with various limit configurations."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            kwargs = {}
            if limit is not None:
                kwargs["limit"] = limit
            if include_all:
                kwargs["include_all"] = include_all

            results = self.filter.filter_experiments(**kwargs)
            assert len(results) == expected_count

    def test_filter_default_limit_with_many_experiments(self):
        """Test default limit of 10 when no limit specified."""
        # Create more than 10 experiments using factory
        many_experiments = []
        for i in range(15):
            many_experiments.append(
                TestDataFactory.create_experiment_metadata(
                    experiment_id=f"exp{i:05d}",
                    name=f"experiment-{i}",
                    status="completed",
                    tags=[],
                    started_at=f"2025-06-28T{10 + i % 14:02d}:00:00",
                    created_at=f"2025-06-28T{9 + i % 14:02d}:59:00",
                )
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
        # Test experiment with no timezone info using factory
        exp_no_tz = TestDataFactory.create_experiment_metadata(
            experiment_id="exp_no_tz",
            name="no-timezone",
            status="completed",
            tags=[],
            started_at="2025-06-28T10:00:00",  # No timezone
            created_at="2025-06-28T09:59:00",
        )

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

    @pytest.mark.parametrize(
        "time_spec,expected_valid",
        [
            ("1 hour ago", True),
            ("2025-01-01T12:00:00", True),
            ("2025-01-01T12:00:00+05:00", True),
            ("not a valid time", False),
            ("", False),
            (None, False),
        ],
    )
    def test_parse_time_spec_validity(self, time_spec, expected_valid):
        """Test parsing various time specifications for validity."""
        result = parse_time_spec(time_spec)

        if expected_valid:
            assert result is not None
            assert isinstance(result, datetime)
            if time_spec not in [None, ""]:
                assert result.tzinfo is not None
        else:
            assert result is None

    def test_parse_time_spec_iso_format_details(self):
        """Test parsing ISO format dates with specific details."""
        result = parse_time_spec("2025-01-01T12:00:00")
        assert result is not None
        assert result.year == 2025
        assert result.month == 1
        assert result.day == 1
        assert result.hour == 12

    @pytest.mark.parametrize(
        "day_term,expected_time",
        [
            (
                "today",
                lambda: datetime.combine(
                    datetime.now().date(), datetime.min.time(), timezone.utc
                ),
            ),
            (
                "yesterday",
                lambda: datetime.combine(
                    datetime.now().date() - timedelta(days=1),
                    datetime.min.time(),
                    timezone.utc,
                ),
            ),
            (
                "tomorrow",
                lambda: datetime.combine(
                    datetime.now().date() + timedelta(days=1),
                    datetime.min.time(),
                    timezone.utc,
                ),
            ),
            (
                "TODAY",
                lambda: datetime.combine(
                    datetime.now().date(), datetime.min.time(), timezone.utc
                ),
            ),  # Case insensitive
        ],
    )
    def test_parse_time_spec_relative_days(self, day_term, expected_time):
        """Test parsing relative day terms returns beginning of day."""
        result = parse_time_spec(day_term)
        expected = expected_time()

        assert result is not None
        assert result.date() == expected.date()
        assert result.time() == expected.time()
        assert result.tzinfo == expected.tzinfo

    def test_parse_time_spec_natural_language(self):
        """Test parsing natural language time specifications."""
        # These tests depend on dateparser behavior
        result = parse_time_spec("1 hour ago")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

        # Should be approximately 1 hour ago (within 5 minutes tolerance)
        now = datetime.now(timezone.utc)
        time_diff = abs((now - result).total_seconds() - 3600)  # 1 hour = 3600 seconds
        assert time_diff < 300  # 5 minutes tolerance
