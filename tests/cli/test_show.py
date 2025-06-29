"""
Tests for yanex CLI show command functionality.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

import pytest

from yanex.cli.commands.show import find_experiment
from yanex.cli.filters import ExperimentFilter
from yanex.cli.formatters.console import ExperimentTableFormatter
from yanex.core.manager import ExperimentManager
from yanex.core.storage import ExperimentStorage


class TestFindExperiment:
    """Test experiment lookup functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock filter with test experiments
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = ExperimentStorage(self.temp_dir)
        self.manager = Mock(spec=ExperimentManager)
        self.manager.storage = self.storage

        self.filter = ExperimentFilter(self.manager)

        # Create test experiment data
        self.sample_experiments = [
            {
                "id": "abcd1234",
                "name": "test-experiment",
                "status": "completed",
                "created_at": "2025-06-28T10:00:00",
            },
            {
                "id": "efgh5678",
                "name": "test-experiment",  # Duplicate name
                "status": "running",
                "created_at": "2025-06-28T11:00:00",
            },
            {
                "id": "ijkl9012",
                "name": "unique-experiment",
                "status": "failed",
                "created_at": "2025-06-28T12:00:00",
            },
            {
                "id": "mnop3456",
                "name": None,  # Unnamed experiment
                "status": "completed",
                "created_at": "2025-06-28T13:00:00",
            },
        ]

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_experiment_by_id_success(self):
        """Test finding experiment by exact ID match."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            result = find_experiment(self.filter, "abcd1234")

            assert result is not None
            assert isinstance(result, dict)
            assert result["id"] == "abcd1234"
            assert result["name"] == "test-experiment"

    def test_find_experiment_by_id_not_found(self):
        """Test finding experiment by non-existent ID."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            result = find_experiment(self.filter, "notfound")

            assert result is None

    def test_find_experiment_by_name_unique(self):
        """Test finding experiment by unique name."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            result = find_experiment(self.filter, "unique-experiment")

            assert result is not None
            assert isinstance(result, dict)
            assert result["id"] == "ijkl9012"
            assert result["name"] == "unique-experiment"

    def test_find_experiment_by_name_duplicate(self):
        """Test finding experiment by name with duplicates returns list."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            result = find_experiment(self.filter, "test-experiment")

            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["name"] == "test-experiment"
            assert result[1]["name"] == "test-experiment"
            assert result[0]["id"] != result[1]["id"]

    def test_find_experiment_by_name_not_found(self):
        """Test finding experiment by non-existent name."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            result = find_experiment(self.filter, "nonexistent-experiment")

            assert result is None

    def test_find_experiment_id_takes_precedence(self):
        """Test that ID lookup takes precedence over name lookup."""
        # Create an experiment where the 8-char ID happens to match another experiment's name
        special_experiments = self.sample_experiments + [
            {
                "id": "testexpr",  # 8-char ID that could be confused with name
                "name": "special-experiment",
                "status": "completed",
                "created_at": "2025-06-28T14:00:00",
            }
        ]

        with patch.object(
            self.filter, "_load_all_experiments", return_value=special_experiments
        ):
            # Should find by ID (8 chars), not by name
            result = find_experiment(self.filter, "testexpr")

            assert result is not None
            assert isinstance(result, dict)  # Single result, not list
            assert result["id"] == "testexpr"
            assert result["name"] == "special-experiment"

    def test_find_experiment_invalid_length_id(self):
        """Test that non-8-character identifiers are treated as names."""
        with patch.object(
            self.filter, "_load_all_experiments", return_value=self.sample_experiments
        ):
            # Try with a short identifier - should be treated as name
            result = find_experiment(self.filter, "abcd")

            assert result is None  # No experiment with name "abcd"


class TestFormatterHelperMethods:
    """Test ExperimentTableFormatter helper methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ExperimentTableFormatter()

    def test_format_time_with_timezone(self):
        """Test formatting datetime with timezone."""
        result = self.formatter._format_time("2025-06-28T15:30:45.123456+10:00")
        assert result == "2025-06-28 15:30:45"

    def test_format_time_without_timezone(self):
        """Test formatting datetime without timezone."""
        result = self.formatter._format_time("2025-06-28T15:30:45")
        assert result == "2025-06-28 15:30:45"

    def test_format_time_invalid(self):
        """Test formatting invalid time string."""
        result = self.formatter._format_time("invalid-time")
        assert result == "invalid-time"

    def test_calculate_duration_with_end_time(self):
        """Test calculating duration between two times."""
        start_time = "2025-06-28T15:30:00"
        end_time = "2025-06-28T15:35:30"

        result = self.formatter._calculate_duration(start_time, end_time)
        assert "5m 30s" in result

    def test_calculate_duration_ongoing(self):
        """Test calculating duration for ongoing experiment."""
        start_time = "2025-06-28T15:30:00"

        # Mock the current time to be 5 minutes later
        with patch("yanex.cli.filters.time_utils.datetime") as mock_datetime:
            mock_now = datetime(2025, 6, 28, 15, 35, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = mock_now
            mock_datetime.fromisoformat = datetime.fromisoformat

            result = self.formatter._calculate_duration(start_time, None)
            assert "(ongoing)" in result

    def test_format_file_size_bytes(self):
        """Test formatting file size in bytes."""
        result = self.formatter._format_file_size(512)
        assert result == "512.0 B"

    def test_format_file_size_kilobytes(self):
        """Test formatting file size in kilobytes."""
        result = self.formatter._format_file_size(2048)
        assert result == "2.0 KB"

    def test_format_file_size_megabytes(self):
        """Test formatting file size in megabytes."""
        result = self.formatter._format_file_size(5242880)  # 5MB
        assert result == "5.0 MB"

    def test_format_timestamp(self):
        """Test formatting Unix timestamp."""
        import time

        # Create a known timestamp
        dt = datetime(2025, 6, 28, 15, 30, 45)
        timestamp = time.mktime(dt.timetuple())

        result = self.formatter._format_timestamp(timestamp)
        assert "2025-06-28 15:30" in result

    def test_format_timestamp_invalid(self):
        """Test formatting invalid timestamp."""
        # Use a timestamp that will cause an exception (way beyond valid range)
        result = self.formatter._format_timestamp(float("inf"))
        assert result == "unknown"


class TestMetricsDisplayLogic:
    """Test the metrics selection and display logic."""

    def test_many_metrics_selection(self):
        """Test metric selection for experiments with many metrics."""
        # Create a scenario with many metrics
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

        # Simulate the key metrics selection logic
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

    def test_requested_metrics_validation(self):
        """Test validation of user-requested metrics."""
        all_metrics = ["accuracy", "loss", "epoch", "custom_metric"]
        requested_metrics = ["accuracy", "nonexistent", "epoch", "another_missing"]

        # Simulate the validation logic
        shown_metrics = []
        missing_metrics = []

        for metric in requested_metrics:
            if metric in all_metrics:
                shown_metrics.append(metric)
            else:
                missing_metrics.append(metric)

        assert shown_metrics == ["accuracy", "epoch"]
        assert missing_metrics == ["nonexistent", "another_missing"]

    def test_few_metrics_no_summary(self):
        """Test that few metrics don't trigger summary mode."""
        all_metrics = ["accuracy", "loss", "epoch"]

        # Should not trigger summary mode (≤8 metrics)
        assert len(all_metrics) <= 8

        # Should show all metrics normally
        shown_metrics = all_metrics
        assert len(shown_metrics) == 3
        assert shown_metrics == all_metrics
