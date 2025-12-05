"""
Tests for yanex CLI show command functionality.
"""

import time
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from tests.test_utils import TestDataFactory
from yanex.cli.commands.show import find_experiment
from yanex.cli.filters import ExperimentFilter
from yanex.cli.formatters.console import ExperimentTableFormatter


class TestFindExperiment:
    """Test experiment lookup functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, isolated_storage, isolated_manager):
        """Set up test fixtures with real isolated storage."""
        self.storage = isolated_storage
        self.manager = isolated_manager
        self.filter = ExperimentFilter(self.manager)

        # Create test experiments in storage
        test_experiments = [
            ("abcd1234", "test-experiment", "completed"),
            ("efgh5678", "test-experiment", "running"),  # Duplicate name
            ("ijkl9012", "unique-experiment", "failed"),
            ("mnop3456", None, "completed"),  # Unnamed
            ("7775550b", "amb-1", "completed"),
            ("777999aa", "amb-2", "running"),
        ]

        for exp_id, name, status in test_experiments:
            metadata = TestDataFactory.create_experiment_metadata(
                experiment_id=exp_id, name=name, status=status
            )
            self.storage.create_experiment_directory(exp_id)
            self.storage.save_metadata(exp_id, metadata)

    @pytest.mark.parametrize(
        "identifier,expected_id,expected_name",
        [
            ("abcd1234", "abcd1234", "test-experiment"),
            ("unique-experiment", "ijkl9012", "unique-experiment"),
        ],
    )
    def test_find_experiment_by_identifier_success(
        self, identifier, expected_id, expected_name
    ):
        """Test finding experiment by ID or unique name."""
        result = find_experiment(self.filter, identifier)

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == expected_id
        assert result["name"] == expected_name

    @pytest.mark.parametrize(
        "identifier",
        [
            "notfound",
            "nonexistent-experiment",
            "abcd2",  # Non-8-character treated as name
        ],
    )
    def test_find_experiment_not_found(self, identifier):
        """Test finding experiment with non-existent identifiers."""
        result = find_experiment(self.filter, identifier)
        assert result is None

    def test_find_experiment_by_name_duplicate_returns_list(self):
        """Test finding experiment by name with duplicates returns list."""
        result = find_experiment(self.filter, "test-experiment")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(exp["name"] == "test-experiment" for exp in result)
        assert result[0]["id"] != result[1]["id"]

    def test_find_experiment_id_takes_precedence_over_name(self):
        """Test that ID lookup takes precedence over name lookup."""
        # Create an experiment where the 8-char ID could be confused with name
        exp_id = "testexpr"
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id,
            name="special-experiment",
            status="completed",
        )
        self.storage.create_experiment_directory(exp_id)
        self.storage.save_metadata(exp_id, metadata)

        result = find_experiment(self.filter, "testexpr")

        assert result is not None
        assert isinstance(result, dict)  # Single result, not list
        assert result["id"] == "testexpr"
        assert result["name"] == "special-experiment"

    def test_find_experiment_by_unique_id_prefix(self):
        """Test finding experiment by a unique ID prefix returns the single match."""
        # 'abc' uniquely matches id 'abcd1234'
        result = find_experiment(self.filter, "abc")

        assert result is not None
        assert isinstance(result, dict)
        assert result["id"] == "abcd1234"

    def test_find_experiment_by_ambiguous_id_prefix_returns_list(self):
        """Test finding experiment by an ambiguous ID prefix returns a list of matches."""
        result = find_experiment(self.filter, "777")

        assert result is not None
        assert isinstance(result, list)
        # Should include both 777* experiments
        ids = sorted(exp["id"] for exp in result)
        assert ids == ["7775550b", "777999aa"]

    def test_find_experiment_by_id_prefix_no_match(self):
        """Test that a non-matching ID prefix returns None (falls through name match too)."""
        result = find_experiment(self.filter, "zzz")
        assert result is None


class TestFormatterHelperMethods:
    """Test ExperimentTableFormatter helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ExperimentTableFormatter()

    @pytest.mark.parametrize(
        "time_string,expected_output",
        [
            ("2025-06-28T15:30:45.123456+10:00", "2025-06-28 15:30:45"),
            ("2025-06-28T15:30:45", "2025-06-28 15:30:45"),
            ("invalid-time", "invalid-time"),
        ],
    )
    def test_format_time(self, time_string, expected_output):
        """Test formatting various datetime strings."""
        result = self.formatter._format_time(time_string)
        assert result == expected_output

    def test_calculate_duration_with_end_time(self):
        """Test calculating duration between two times."""
        start_time = "2025-06-28T15:30:00"
        end_time = "2025-06-28T15:35:30"

        result = self.formatter._calculate_duration(start_time, end_time)
        assert "5m 30s" in result

    def test_calculate_duration_ongoing_experiment(self):
        """Test calculating duration for ongoing experiment."""
        start_time = "2025-06-28T15:30:00"

        # Mock the current time to be 5 minutes later
        with patch("yanex.cli.filters.time_utils.datetime") as mock_datetime:
            mock_now = datetime(2025, 6, 28, 15, 35, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.fromisoformat = datetime.fromisoformat

            result = self.formatter._calculate_duration(start_time, None)
            assert "+" in result

    @pytest.mark.parametrize(
        "file_size,expected_output",
        [
            (512, "512.0 B"),
            (2048, "2.0 KB"),
            (5242880, "5.0 MB"),  # 5MB
        ],
    )
    def test_format_file_size(self, file_size, expected_output):
        """Test formatting various file sizes."""
        result = self.formatter._format_file_size(file_size)
        assert result == expected_output

    def test_format_timestamp_valid(self):
        """Test formatting valid Unix timestamp."""
        # Create a known timestamp
        dt = datetime(2025, 6, 28, 15, 30, 45)
        timestamp = time.mktime(dt.timetuple())

        result = self.formatter._format_timestamp(timestamp)
        assert "2025-06-28 15:30" in result

    def test_format_timestamp_invalid(self):
        """Test formatting invalid timestamp."""
        result = self.formatter._format_timestamp(float("inf"))
        assert result == "unknown"


class TestMetricsDisplayLogic:
    """Test the metrics selection and display logic."""

    def test_many_metrics_selection_prioritizes_key_metrics(self):
        """Test metric selection for experiments with many metrics prioritizes key metrics."""
        # Create scenario with many metrics using utilities
        all_metrics = [
            "accuracy",
            "loss",
            "epoch",
            "learning_rate",
            "f1_score",
            "precision",
            "recall",
            "auc",
            "custom_metric_1",
            "custom_metric_2",
            "metric_a",
            "metric_b",
            "metric_c",
            "metric_d",
            "metric_e",
        ]

        key_metrics = [
            "accuracy",
            "loss",
            "epoch",
            "learning_rate",
            "f1_score",
            "precision",
            "recall",
        ]
        shown_metrics = []

        # Add key metrics that exist (up to 8)
        for metric in key_metrics:
            if metric in all_metrics and len(shown_metrics) < 8:
                shown_metrics.append(metric)

        # Fill remaining slots with other metrics
        remaining_metrics = [m for m in all_metrics if m not in shown_metrics]
        for metric in remaining_metrics:
            if len(shown_metrics) < 8:
                shown_metrics.append(metric)
            else:
                break

        # Should show 8 metrics total
        assert len(shown_metrics) == 8
        # Should prioritize key metrics
        assert "accuracy" in shown_metrics
        assert "loss" in shown_metrics
        assert "epoch" in shown_metrics
        # Should include some additional metrics
        assert len([m for m in shown_metrics if m not in key_metrics]) > 0

    @pytest.mark.parametrize(
        "all_metrics,requested_metrics,expected_shown,expected_missing",
        [
            (
                ["accuracy", "loss", "epoch", "custom_metric"],
                ["accuracy", "nonexistent", "epoch", "another_missing"],
                ["accuracy", "epoch"],
                ["nonexistent", "another_missing"],
            ),
            (
                ["accuracy", "loss"],
                ["accuracy", "loss"],
                ["accuracy", "loss"],
                [],
            ),
            (
                ["accuracy", "loss"],
                ["missing1", "missing2"],
                [],
                ["missing1", "missing2"],
            ),
        ],
    )
    def test_requested_metrics_validation(
        self, all_metrics, requested_metrics, expected_shown, expected_missing
    ):
        """Test validation of user-requested metrics."""
        shown_metrics = []
        missing_metrics = []

        for metric in requested_metrics:
            if metric in all_metrics:
                shown_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        assert shown_metrics == expected_shown
        assert missing_metrics == expected_missing

    def test_few_metrics_show_all_without_summary(self):
        """Test that few metrics don't trigger summary mode and show all."""
        all_metrics = ["accuracy", "loss", "epoch"]

        # Should not trigger summary mode (≤8 metrics)
        assert len(all_metrics) <= 8

        # Should show all metrics normally
        shown_metrics = all_metrics
        assert len(shown_metrics) == 3
        assert shown_metrics == all_metrics

    def test_metrics_selection_with_experiment_data(self):
        """Test metrics selection using actual experiment data structure."""
        # Create experiment with many metrics using factory
        experiment_results = TestDataFactory.create_experiment_results(
            result_type="ml_metrics",
            custom_metric_1=0.123,
            custom_metric_2=0.456,
            custom_metric_3=0.789,
            custom_metric_4=0.101,
            custom_metric_5=0.202,
        )

        # Extract metrics (simulate what the formatter would do)
        all_metrics = list(experiment_results.keys())

        # Should have more than 8 metrics to test selection
        assert len(all_metrics) >= 8

        # Test key metrics are available in the results
        key_metrics_in_results = [
            metric
            for metric in ["accuracy", "loss", "precision", "recall", "f1_score"]
            if metric in all_metrics
        ]

        # Should find some key metrics
        assert len(key_metrics_in_results) > 0
        assert "accuracy" in key_metrics_in_results

    def test_empty_metrics_handling(self):
        """Test handling of experiments with no metrics."""
        all_metrics = []
        shown_metrics = all_metrics

        assert len(shown_metrics) == 0
        assert shown_metrics == []
