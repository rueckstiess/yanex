"""
Tests for filter arguments module.

This module tests CLI filter argument decorators and utilities.
"""

from unittest.mock import patch

import click
import pytest

from yanex.cli.filters.arguments import (
    experiment_filter_options,
    format_filter_summary,
    parse_cli_time_filters,
    require_filters_or_confirmation,
    validate_filter_arguments,
)


class TestExperimentFilterOptionsDecorator:
    """Test experiment_filter_options decorator functionality."""

    def test_decorator_adds_all_options_by_default(self):
        """Test that decorator adds all options when all flags are True."""

        @experiment_filter_options()
        @click.command()
        def test_command(**kwargs):
            """Test command."""
            pass

        # Check that all expected parameters are present
        param_names = {p.name for p in test_command.params}

        assert "script_pattern" in param_names
        assert "status" in param_names
        assert "name_pattern" in param_names
        assert "tags" in param_names
        assert "started_after" in param_names
        assert "started_before" in param_names
        assert "ended_after" in param_names
        assert "ended_before" in param_names
        assert "ids" in param_names
        assert "archived" in param_names
        assert "limit" in param_names

    def test_decorator_excludes_ids_when_disabled(self):
        """Test that --ids option is excluded when include_ids=False."""

        @experiment_filter_options(include_ids=False)
        @click.command()
        def test_command(**kwargs):
            """Test command."""
            pass

        param_names = {p.name for p in test_command.params}
        assert "ids" not in param_names
        # Other options should still be present
        assert "status" in param_names

    def test_decorator_excludes_archived_when_disabled(self):
        """Test that --archived option is excluded when include_archived=False."""

        @experiment_filter_options(include_archived=False)
        @click.command()
        def test_command(**kwargs):
            """Test command."""
            pass

        param_names = {p.name for p in test_command.params}
        assert "archived" not in param_names
        # Other options should still be present
        assert "status" in param_names

    def test_decorator_excludes_limit_when_disabled(self):
        """Test that --limit option is excluded when include_limit=False."""

        @experiment_filter_options(include_limit=False)
        @click.command()
        def test_command(**kwargs):
            """Test command."""
            pass

        param_names = {p.name for p in test_command.params}
        assert "limit" not in param_names
        # Other options should still be present
        assert "status" in param_names

    def test_decorator_with_default_limit(self):
        """Test that default_limit sets the limit default value."""

        @experiment_filter_options(default_limit=10)
        @click.command()
        def test_command(**kwargs):
            """Test command."""
            pass

        # Find the limit parameter
        limit_param = None
        for param in test_command.params:
            if param.name == "limit":
                limit_param = param
                break

        assert limit_param is not None
        assert limit_param.default == 10

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function docstring and name."""

        @experiment_filter_options()
        @click.command()
        def my_custom_command(**kwargs):
            """My custom docstring."""
            pass

        # Function name should be preserved by Click
        assert "my_custom_command" in str(my_custom_command.callback)


class TestValidateFilterArguments:
    """Test validate_filter_arguments function."""

    def test_validate_empty_arguments(self):
        """Test validation with no arguments."""
        result = validate_filter_arguments()
        assert result == {}

    def test_validate_converts_ids_tuple_to_list(self):
        """Test that IDs tuple is converted to list."""
        result = validate_filter_arguments(ids=("exp1", "exp2", "exp3"))
        assert result == {"ids": ["exp1", "exp2", "exp3"]}

    def test_validate_ignores_empty_ids_tuple(self):
        """Test that empty IDs tuple is ignored."""
        result = validate_filter_arguments(ids=())
        assert result == {}

    def test_validate_preserves_status(self):
        """Test that status is preserved."""
        result = validate_filter_arguments(status="completed")
        assert result == {"status": "completed"}

    def test_validate_preserves_name_pattern(self):
        """Test that name pattern is preserved."""
        result = validate_filter_arguments(name_pattern="test-*")
        assert result == {"name_pattern": "test-*"}

    def test_validate_preserves_empty_name_pattern(self):
        """Test that empty string name pattern is preserved."""
        result = validate_filter_arguments(name_pattern="")
        assert result == {"name_pattern": ""}

    def test_validate_converts_tags_tuple_to_list(self):
        """Test that tags tuple is converted to list."""
        result = validate_filter_arguments(tags=("tag1", "tag2"))
        assert result == {"tags": ["tag1", "tag2"]}

    def test_validate_ignores_empty_tags_tuple(self):
        """Test that empty tags tuple is ignored."""
        result = validate_filter_arguments(tags=())
        assert result == {}

    def test_validate_preserves_time_filters(self):
        """Test that time filters are preserved."""
        result = validate_filter_arguments(
            started_after="2025-01-01",
            started_before="2025-12-31",
            ended_after="2025-06-01",
            ended_before="2025-06-30",
        )
        assert result == {
            "started_after": "2025-01-01",
            "started_before": "2025-12-31",
            "ended_after": "2025-06-01",
            "ended_before": "2025-06-30",
        }

    def test_validate_preserves_archived_flag(self):
        """Test that archived flag is preserved."""
        result = validate_filter_arguments(archived=True)
        assert result == {"archived": True}

        result = validate_filter_arguments(archived=False)
        assert result == {"archived": False}

    def test_validate_ignores_none_archived(self):
        """Test that None archived value is ignored."""
        result = validate_filter_arguments(archived=None)
        assert result == {}

    def test_validate_passes_through_additional_kwargs(self):
        """Test that additional kwargs are passed through."""
        result = validate_filter_arguments(
            script_pattern="train.py", limit=10, custom_field="value"
        )
        assert result["script_pattern"] == "train.py"
        assert result["limit"] == 10
        assert result["custom_field"] == "value"

    @patch("click.echo")
    def test_validate_warns_on_script_pattern_with_path_separator(self, mock_echo):
        """Test warning when script_pattern contains path separators."""
        result = validate_filter_arguments(script_pattern="path/to/script.py")

        # Should still include the pattern
        assert result["script_pattern"] == "path/to/script.py"

        # Should print warning
        mock_echo.assert_called_once()
        warning_text = mock_echo.call_args[0][0]
        assert "Warning" in warning_text
        assert "path separators" in warning_text

    def test_validate_combined_arguments(self):
        """Test validation with multiple arguments."""
        result = validate_filter_arguments(
            ids=("exp1", "exp2"),
            status="completed",
            name_pattern="test-*",
            tags=("ml", "training"),
            started_after="2025-01-01",
            archived=True,
        )

        assert result == {
            "ids": ["exp1", "exp2"],
            "status": "completed",
            "name_pattern": "test-*",
            "tags": ["ml", "training"],
            "started_after": "2025-01-01",
            "archived": True,
        }


class TestRequireFiltersOrConfirmation:
    """Test require_filters_or_confirmation function."""

    def test_require_filters_passes_with_filters(self):
        """Test that function passes when filters are provided."""
        filter_args = {"status": "completed", "name_pattern": "test-*"}
        # Should not raise
        require_filters_or_confirmation(filter_args, "delete", force=False)

    def test_require_filters_passes_with_force_flag(self):
        """Test that function passes with force flag even without filters."""
        filter_args = {}
        # Should not raise
        require_filters_or_confirmation(filter_args, "delete", force=True)

    @patch("click.confirm", return_value=True)
    def test_require_filters_passes_with_confirmation(self, mock_confirm):
        """Test that function passes when user confirms."""
        filter_args = {}
        # Should not raise
        require_filters_or_confirmation(filter_args, "delete", force=False)
        mock_confirm.assert_called_once()

    @patch("click.confirm", return_value=False)
    def test_require_filters_raises_without_confirmation(self, mock_confirm):
        """Test that function raises when user declines confirmation."""
        filter_args = {}
        with pytest.raises(click.ClickException) as exc_info:
            require_filters_or_confirmation(filter_args, "delete", force=False)

        assert "cancelled" in str(exc_info.value).lower()

    def test_require_filters_ignores_non_meaningful_filters(self):
        """Test that non-meaningful filters don't bypass confirmation."""
        # limit, sort_by, etc. should not count as meaningful filters
        filter_args = {"limit": 10, "sort_by": "created_at"}

        with patch("click.confirm", return_value=True):
            require_filters_or_confirmation(filter_args, "delete", force=False)


class TestFormatFilterSummary:
    """Test format_filter_summary function."""

    def test_format_empty_filters(self):
        """Test formatting empty filters."""
        result = format_filter_summary({})
        assert result == "No filters applied"

    def test_format_with_ids(self):
        """Test formatting with IDs."""
        result = format_filter_summary({"ids": ["exp1", "exp2"]})
        assert "IDs: exp1, exp2" in result

    def test_format_with_many_ids(self):
        """Test formatting with more than 3 IDs."""
        result = format_filter_summary(
            {"ids": ["exp1", "exp2", "exp3", "exp4", "exp5"]}
        )
        assert "IDs: exp1, exp2, exp3 and 2 more" in result

    def test_format_with_status(self):
        """Test formatting with status."""
        result = format_filter_summary({"status": "completed"})
        assert "Status: completed" in result

    def test_format_with_status_list(self):
        """Test formatting with status list."""
        result = format_filter_summary({"status": ["completed", "failed"]})
        assert "Status: completed, failed" in result

    def test_format_with_name_pattern(self):
        """Test formatting with name pattern."""
        result = format_filter_summary({"name_pattern": "test-*"})
        assert "Name pattern: 'test-*'" in result

    def test_format_with_script_pattern(self):
        """Test formatting with script pattern."""
        result = format_filter_summary({"script_pattern": "train.py"})
        assert "Script pattern: 'train.py'" in result

    def test_format_with_tags(self):
        """Test formatting with tags."""
        result = format_filter_summary({"tags": ["ml", "training"]})
        assert "Tags: ml, training" in result

    def test_format_with_time_filters(self):
        """Test formatting with time filters."""
        result = format_filter_summary(
            {
                "started_after": "2025-01-01",
                "started_before": "2025-12-31",
                "ended_after": "2025-06-01",
                "ended_before": "2025-06-30",
            }
        )
        assert "Started after: 2025-01-01" in result
        assert "Started before: 2025-12-31" in result
        assert "Ended after: 2025-06-01" in result
        assert "Ended before: 2025-06-30" in result

    def test_format_with_archived_true(self):
        """Test formatting with archived=True."""
        result = format_filter_summary({"archived": True})
        assert "archived only" in result

    def test_format_with_archived_false(self):
        """Test formatting with archived=False."""
        result = format_filter_summary({"archived": False})
        assert "non-archived only" in result

    def test_format_with_multiple_filters(self):
        """Test formatting with multiple filters."""
        result = format_filter_summary(
            {
                "status": "completed",
                "name_pattern": "test-*",
                "tags": ["ml"],
                "archived": False,
            }
        )
        assert "Status: completed" in result
        assert "Name pattern: 'test-*'" in result
        assert "Tags: ml" in result
        assert "non-archived only" in result
        assert "Filters:" in result


class TestParseCliTimeFilters:
    """Test parse_cli_time_filters function."""

    @patch("yanex.cli.error_handling.CLIErrorHandler.parse_time_filters")
    def test_parse_time_filters_delegates_to_error_handler(self, mock_parse):
        """Test that function delegates to CLIErrorHandler."""
        mock_parse.return_value = ("a", "b", "c", "d")

        result = parse_cli_time_filters(
            "2025-01-01", "2025-12-31", "2025-06-01", "2025-06-30"
        )

        assert result == ("a", "b", "c", "d")
        mock_parse.assert_called_once_with(
            "2025-01-01", "2025-12-31", "2025-06-01", "2025-06-30"
        )

    @patch("yanex.cli.error_handling.CLIErrorHandler.parse_time_filters")
    def test_parse_time_filters_converts_value_error_to_bad_parameter(self, mock_parse):
        """Test that ValueError is converted to BadParameter."""
        mock_parse.side_effect = ValueError("Invalid date format")

        with pytest.raises(click.BadParameter) as exc_info:
            parse_cli_time_filters("invalid", None, None, None)

        assert "Invalid date format" in str(exc_info.value)

    @patch("yanex.cli.error_handling.CLIErrorHandler.parse_time_filters")
    def test_parse_time_filters_with_none_values(self, mock_parse):
        """Test parsing with None values."""
        mock_parse.return_value = (None, None, None, None)

        result = parse_cli_time_filters(None, None, None, None)

        assert result == (None, None, None, None)
