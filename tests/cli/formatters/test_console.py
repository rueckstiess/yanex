"""
Tests for console formatter including script name display.
"""

import pytest
from rich.text import Text

from yanex.cli.formatters.console import ExperimentTableFormatter


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
        assert result.style == "cyan"

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
        """Test truncation of long script names while preserving extension."""
        # Script name within 15 char limit (should not truncate)
        script_path = "/path/to/short_name.py"  # 13 chars - fits within 15
        result = formatter._format_script(script_path)
        assert str(result.plain) == "short_name.py"
        assert len(result.plain) == 13

        # Script name > 15 chars (should truncate)
        script_path = "/path/to/very_long_script_name.py"
        result = formatter._format_script(script_path)
        # Should be truncated to "very_long....py" (15 chars: 9 + 3 + 3)
        assert str(result.plain) == "very_long....py"
        assert len(result.plain) == 15
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

    def test_format_script_preserves_extension_in_truncation(self, formatter):
        """Test that truncation preserves the file extension."""
        # Create a very long script name
        long_name = "a" * 20 + ".py"  # 22 chars total
        script_path = f"/path/to/{long_name}"
        result = formatter._format_script(script_path)

        # Should be truncated but preserve .py
        assert result.plain.endswith(".py")
        assert len(result.plain) == 15
        assert "..." in result.plain

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
            len(table.columns) == 8
        )  # ID, Script, Name, Status, Deps, Duration, Tags, Started
        assert table.columns[1].header == "Script"
        assert table.columns[1].style == "cyan"
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
        assert len(table.columns) == 8
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
        assert column_headers[4] == "Deps"
        assert column_headers[5] == "Duration"
        assert column_headers[6] == "Tags"
        assert column_headers[7] == "Started"

    def test_format_dependencies_with_no_dependencies(self, formatter):
        """Test formatting dependencies when experiment has none."""
        # None
        result = formatter._format_dependencies(None)
        assert isinstance(result, Text)
        assert str(result.plain) == "-"
        assert result.style == "dim"

        # Empty dict
        result = formatter._format_dependencies({})
        assert str(result.plain) == "-"

        # has_dependencies = False
        result = formatter._format_dependencies({"has_dependencies": False})
        assert str(result.plain) == "-"

        # has_dependencies = True but count = 0
        result = formatter._format_dependencies(
            {"has_dependencies": True, "dependency_count": 0}
        )
        assert str(result.plain) == "-"

    def test_format_dependencies_with_dependencies(self, formatter):
        """Test formatting dependencies when experiment has them."""
        # Single dependency
        result = formatter._format_dependencies(
            {"has_dependencies": True, "dependency_count": 1}
        )
        assert isinstance(result, Text)
        assert str(result.plain) == "→ 1"
        assert result.style == "yellow"

        # Multiple dependencies
        result = formatter._format_dependencies(
            {"has_dependencies": True, "dependency_count": 3}
        )
        assert str(result.plain) == "→ 3"
        assert result.style == "yellow"
