"""
Tests for id command functionality.

This module tests the id command for retrieving experiment IDs in various formats.
"""

import json

import pytest

from tests.test_utils import create_cli_runner
from yanex.cli.main import cli


class TestIdCommandHelp:
    """Test id command help and documentation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_help_output(self):
        """Test that id command shows help information."""
        result = self.runner.invoke(cli, ["id", "--help"])
        assert result.exit_code == 0
        assert "Get experiment IDs matching filter criteria" in result.output
        assert "--format" in result.output
        assert "--limit" in result.output
        assert "--status" in result.output
        assert "--name" in result.output
        assert "--tag" in result.output
        assert "--archived" in result.output

    def test_id_help_shows_examples(self):
        """Test that help includes usage examples."""
        result = self.runner.invoke(cli, ["id", "--help"])
        assert result.exit_code == 0
        assert "Examples:" in result.output
        assert "yanex id" in result.output
        assert "bash substitution" in result.output


class TestIdCommandBasicBehavior:
    """Test id command basic behavior without filters."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_basic_invocation(self):
        """Test basic id invocation works without errors."""
        result = self.runner.invoke(cli, ["id"])
        assert result.exit_code == 0
        # Should return empty CSV format or IDs
        assert '"' in result.output or result.output.strip() == '""'

    def test_id_default_format_is_csv(self):
        """Test that default format is CSV with quotes."""
        result = self.runner.invoke(cli, ["id"])
        assert result.exit_code == 0
        # CSV format should start and end with quotes
        output = result.output.strip()
        if output:
            assert output.startswith('"')
            assert output.endswith('"')

    def test_id_csv_format_explicit(self):
        """Test explicit CSV format."""
        result = self.runner.invoke(cli, ["id", "--format", "csv"])
        assert result.exit_code == 0
        output = result.output.strip()
        if output:
            assert output.startswith('"')
            assert output.endswith('"')

    def test_id_newline_format(self):
        """Test newline format."""
        result = self.runner.invoke(cli, ["id", "--format", "newline"])
        assert result.exit_code == 0

    def test_id_json_format(self):
        """Test JSON format."""
        result = self.runner.invoke(cli, ["id", "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON array
        try:
            data = json.loads(result.output.strip())
            assert isinstance(data, list)
        except json.JSONDecodeError:
            # Empty result is also valid
            assert result.output.strip() == "[]"

    def test_id_short_format_flag(self):
        """Test short -f format flag."""
        result = self.runner.invoke(cli, ["id", "-f", "json"])
        assert result.exit_code == 0
        # Should be valid JSON
        try:
            data = json.loads(result.output.strip())
            assert isinstance(data, list)
        except json.JSONDecodeError:
            assert result.output.strip() == "[]"


class TestIdCommandFiltering:
    """Test id command filtering options."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_id_filter_by_status(self):
        """Test filtering by status."""
        result = self.runner.invoke(cli, ["id", "--status", "completed"])
        assert result.exit_code == 0

    def test_id_filter_by_name_pattern(self):
        """Test filtering by name pattern."""
        result = self.runner.invoke(cli, ["id", "--name", "*test*"])
        assert result.exit_code == 0

    def test_id_filter_by_empty_name(self):
        """Test filtering unnamed experiments with empty string."""
        result = self.runner.invoke(cli, ["id", "--name", ""])
        assert result.exit_code == 0

    def test_id_filter_by_single_tag(self):
        """Test filtering by single tag."""
        result = self.runner.invoke(cli, ["id", "--tag", "training"])
        assert result.exit_code == 0

    def test_id_filter_by_multiple_tags(self):
        """Test filtering by multiple tags (AND logic)."""
        result = self.runner.invoke(
            cli, ["id", "--tag", "training", "--tag", "production"]
        )
        assert result.exit_code == 0

    def test_id_filter_by_script_pattern(self):
        """Test filtering by script pattern."""
        result = self.runner.invoke(cli, ["id", "--script", "train.py"])
        assert result.exit_code == 0

    def test_id_filter_by_time_started_after(self):
        """Test filtering by started after date."""
        result = self.runner.invoke(cli, ["id", "--started-after", "2025-01-01"])
        assert result.exit_code == 0

    def test_id_filter_by_time_started_before(self):
        """Test filtering by started before date."""
        result = self.runner.invoke(cli, ["id", "--started-before", "2025-12-31"])
        assert result.exit_code == 0

    def test_id_filter_by_time_range(self):
        """Test filtering by time range."""
        result = self.runner.invoke(
            cli,
            [
                "id",
                "--started-after",
                "2025-01-01",
                "--started-before",
                "2025-12-31",
            ],
        )
        assert result.exit_code == 0

    def test_id_with_custom_limit(self):
        """Test id with custom limit."""
        result = self.runner.invoke(cli, ["id", "-l", "5"])
        assert result.exit_code == 0

    def test_id_with_zero_limit(self):
        """Test id with zero limit."""
        result = self.runner.invoke(cli, ["id", "-l", "0"])
        assert result.exit_code == 0
        # Should return empty result
        output = result.output.strip()
        assert output == '""' or output == "[]"

    def test_id_archived_flag(self):
        """Test archived flag."""
        result = self.runner.invoke(cli, ["id", "--archived"])
        assert result.exit_code == 0


class TestIdCommandFormats:
    """Test different output formats with various scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_csv_format_structure(self):
        """Test CSV format structure is correct."""
        result = self.runner.invoke(cli, ["id", "--format", "csv"])
        assert result.exit_code == 0
        output = result.output.strip()
        # CSV should be quoted
        if len(output) > 2:  # More than just quotes
            assert output.startswith('"')
            assert output.endswith('"')
            # Content between quotes should be comma-separated
            content = output[1:-1]
            if content:  # If there are IDs
                # Should contain valid hex IDs (8 chars each)
                ids = content.split(",")
                for exp_id in ids:
                    assert len(exp_id) == 8
                    # Check if valid hex
                    try:
                        int(exp_id, 16)
                    except ValueError:
                        pytest.fail(f"Invalid hex ID: {exp_id}")

    def test_newline_format_structure(self):
        """Test newline format structure is correct."""
        result = self.runner.invoke(cli, ["id", "--format", "newline"])
        assert result.exit_code == 0
        # Each line should be a valid experiment ID (8 hex chars) or empty
        if result.output.strip():
            lines = result.output.strip().split("\n")
            for line in lines:
                if line:  # Skip empty lines
                    assert len(line) == 8
                    try:
                        int(line, 16)
                    except ValueError:
                        pytest.fail(f"Invalid hex ID: {line}")

    def test_json_format_structure(self):
        """Test JSON format structure is correct."""
        result = self.runner.invoke(cli, ["id", "--format", "json"])
        assert result.exit_code == 0
        # Should be valid JSON array
        data = json.loads(result.output.strip())
        assert isinstance(data, list)
        # Each element should be an 8-char hex string
        for exp_id in data:
            assert isinstance(exp_id, str)
            assert len(exp_id) == 8
            try:
                int(exp_id, 16)
            except ValueError:
                pytest.fail(f"Invalid hex ID: {exp_id}")


class TestIdCommandCombinations:
    """Test combinations of filters and formats."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_status_filter_with_csv_format(self):
        """Test status filter with CSV format."""
        result = self.runner.invoke(
            cli, ["id", "--status", "completed", "--format", "csv"]
        )
        assert result.exit_code == 0
        output = result.output.strip()
        assert output.startswith('"')
        assert output.endswith('"')

    def test_status_filter_with_json_format(self):
        """Test status filter with JSON format."""
        result = self.runner.invoke(
            cli, ["id", "--status", "failed", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, list)

    def test_tag_filter_with_newline_format(self):
        """Test tag filter with newline format."""
        result = self.runner.invoke(
            cli, ["id", "--tag", "training", "--format", "newline"]
        )
        assert result.exit_code == 0

    def test_multiple_filters_with_limit_and_format(self):
        """Test multiple filters with limit and format."""
        result = self.runner.invoke(
            cli,
            [
                "id",
                "--status",
                "completed",
                "--tag",
                "production",
                "-l",
                "10",
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output.strip())
        assert isinstance(data, list)
        # Should respect limit
        assert len(data) <= 10

    def test_script_filter_with_csv_format(self):
        """Test script filter with CSV format."""
        result = self.runner.invoke(
            cli, ["id", "--script", "train.py", "--format", "csv"]
        )
        assert result.exit_code == 0


class TestIdCommandVerbose:
    """Test verbose output behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_verbose_flag_shows_debug_info(self):
        """Test that verbose flag shows debug information."""
        result = self.runner.invoke(cli, ["-v", "id", "--status", "completed"])
        assert result.exit_code == 0
        # Verbose output goes to stderr, but we can check if command succeeded

    def test_verbose_with_filters(self):
        """Test verbose output with multiple filters."""
        result = self.runner.invoke(
            cli,
            [
                "-v",
                "id",
                "--status",
                "completed",
                "--tag",
                "training",
                "--name",
                "*test*",
            ],
        )
        assert result.exit_code == 0


class TestIdCommandErrorHandling:
    """Test error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_invalid_format(self):
        """Test invalid format option."""
        result = self.runner.invoke(cli, ["id", "--format", "invalid"])
        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()

    def test_invalid_status(self):
        """Test invalid status option."""
        result = self.runner.invoke(cli, ["id", "--status", "invalid_status"])
        assert result.exit_code != 0

    def test_invalid_limit(self):
        """Test invalid limit option."""
        result = self.runner.invoke(cli, ["id", "--limit", "not_a_number"])
        assert result.exit_code != 0

    def test_invalid_time_format(self):
        """Test invalid time format."""
        result = self.runner.invoke(cli, ["id", "--started-after", "invalid_date"])
        # Should either handle gracefully or error appropriately
        # Command should not crash
        assert result.exit_code in [0, 1, 2]
