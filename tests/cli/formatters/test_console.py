"""
Tests for console formatter including script name display.
"""

import pytest
from rich.text import Text

from yanex.cli.formatters.console import ExperimentTableFormatter
from yanex.cli.formatters.theme import SCRIPT_STYLE


class TestExperimentTableFormatter:
    """Test the ExperimentTableFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create a formatter instance."""
        return ExperimentTableFormatter()

    def test_format_script_with_normal_filename(self, formatter):
        """Test formatting script name with normal filename."""
        script_path = "/path/to/train.py"
        result = formatter._format_script(script_path)

        assert isinstance(result, Text)
        assert str(result.plain) == "train.py"
        assert result.style == SCRIPT_STYLE

    def test_format_script_without_path(self, formatter):
        """Test formatting script when path is None or empty."""
        # Test None
        result = formatter._format_script(None)
        assert isinstance(result, Text)
        assert str(result.plain) == "-"
        assert result.style == "dim"

        # Test empty string
        result = formatter._format_script("")
        assert isinstance(result, Text)
        assert str(result.plain) == "-"
        assert result.style == "dim"

    def test_format_script_with_long_filename(self, formatter):
        """Test that long script names are shown in full without truncation."""
        # Script name within 15 char limit
        script_path = "/path/to/short_name.py"  # 13 chars
        result = formatter._format_script(script_path)
        assert str(result.plain) == "short_name.py"
        assert len(result.plain) == 13

        # Script name > 15 chars (should NOT truncate - shown in full)
        script_path = "/path/to/very_long_script_name.py"
        result = formatter._format_script(script_path)
        # Should show full name without truncation
        assert str(result.plain) == "very_long_script_name.py"
        assert len(result.plain) == 24
        assert result.plain.endswith(".py")

    def test_format_script_extracts_filename_only(self, formatter):
        """Test that only filename is extracted, not full path."""
        script_path = "/very/long/path/to/script/directory/train.py"
        result = formatter._format_script(script_path)

        assert str(result.plain) == "train.py"
        assert "path" not in result.plain
        assert "directory" not in result.plain

    def test_format_script_with_various_extensions(self, formatter):
        """Test formatting script names with different extensions."""
        # .py extension
        result = formatter._format_script("/path/to/script.py")
        assert str(result.plain) == "script.py"

        # Other extensions should also work
        result = formatter._format_script("/path/to/script.sh")
        assert str(result.plain) == "script.sh"

        # No extension
        result = formatter._format_script("/path/to/script")
        assert str(result.plain) == "script"

    def test_format_script_with_very_long_filename(self, formatter):
        """Test that even very long script names are shown in full."""
        # Create a very long script name
        long_name = "a" * 20 + ".py"  # 23 chars total
        script_path = f"/path/to/{long_name}"
        result = formatter._format_script(script_path)

        # Should show full name without truncation
        assert result.plain.endswith(".py")
        assert len(result.plain) == 23
        assert str(result.plain) == long_name

    def test_format_experiments_table_includes_script_column(self, formatter):
        """Test that the experiments table includes the Script column."""
        experiments = [
            {
                "id": "abc12345",
                "script_path": "/path/to/train.py",
                "name": "test_experiment",
                "status": "completed",
                "started_at": "2024-01-01T10:00:00",
                "ended_at": "2024-01-01T10:30:00",
                "tags": ["test"],
            }
        ]

        table = formatter.format_experiments_table(experiments)

        # Check that Script column exists (2nd column after ID)
        assert (
            len(table.columns) == 7
        )  # ID, Script, Name, Status, Duration, Tags, Started
        assert table.columns[1].header == "Script"
        assert table.columns[1].style == SCRIPT_STYLE
        # Note: width is set internally but not exposed as a public attribute

    def test_format_experiments_table_with_missing_script(self, formatter):
        """Test table formatting when experiment has no script_path."""
        experiments = [
            {
                "id": "abc12345",
                # No script_path
                "name": "test_experiment",
                "status": "completed",
                "started_at": "2024-01-01T10:00:00",
                "ended_at": "2024-01-01T10:30:00",
                "tags": [],
            }
        ]

        table = formatter.format_experiments_table(experiments)

        # Should handle missing script_path gracefully
        assert len(table.columns) == 7
        # The table should still render without errors
        assert table is not None

    def test_format_experiments_table_script_column_position(self, formatter):
        """Test that Script column is in the correct position (2nd, after ID)."""
        table = formatter.format_experiments_table([])

        # Verify column order
        column_headers = [col.header for col in table.columns]
        assert column_headers[0] == "ID"
        assert column_headers[1] == "Script"
        assert column_headers[2] == "Name"
        assert column_headers[3] == "Status"
        assert column_headers[4] == "Duration"
        assert column_headers[5] == "Tags"
        assert column_headers[6] == "Started"

    def test_calculate_column_width(self, formatter):
        """Test that column width is calculated based on longest value."""
        # Empty experiments should return minimum width
        assert formatter._calculate_column_width([], "script_path", min_width=15) == 15

        # Single experiment with short script name
        experiments = [{"script_path": "/path/to/train.py"}]
        assert (
            formatter._calculate_column_width(experiments, "script_path", min_width=15)
            == 15
        )  # Minimum

        # Single experiment with long script name
        experiments = [{"script_path": "/path/to/very_long_script_name.py"}]
        assert (
            formatter._calculate_column_width(experiments, "script_path", min_width=15)
            == 24
        )

        # Multiple experiments - should use longest
        experiments = [
            {"script_path": "/path/to/short.py"},
            {"script_path": "/path/to/very_long_script_name.py"},
            {"script_path": "/path/to/medium_length.py"},
        ]
        assert (
            formatter._calculate_column_width(experiments, "script_path", min_width=15)
            == 24
        )

        # Experiment with no script_path
        experiments = [{"script_path": None}, {"script_path": "/path/to/train.py"}]
        assert (
            formatter._calculate_column_width(experiments, "script_path", min_width=15)
            == 15
        )

        # Test with name field
        experiments = [{"name": "short"}, {"name": "very-long-experiment-name"}]
        assert (
            formatter._calculate_column_width(experiments, "name", min_width=12) == 25
        )  # "very-long-experiment-name" is 25 chars

        # Test max_width cap
        experiments = [{"name": "a" * 100}]
        assert (
            formatter._calculate_column_width(
                experiments, "name", min_width=12, max_width=50
            )
            == 50
        )
