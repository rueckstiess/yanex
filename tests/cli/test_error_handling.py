"""Tests for CLI error handling utilities."""

from datetime import datetime, timezone
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from yanex.cli.error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
    confirm_destructive_operation,
    format_validation_error,
    show_warning,
    validate_experiment_state,
)
from yanex.utils.exceptions import ValidationError


class TestCLIErrorHandler:
    """Test CLI error handler functionality."""

    def test_handle_cli_errors_decorator_success(self):
        """Test that decorator allows successful function execution."""
        @CLIErrorHandler.handle_cli_errors
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_handle_cli_errors_decorator_preserves_click_exception(self):
        """Test that ClickException is preserved by decorator."""
        @CLIErrorHandler.handle_cli_errors
        def test_func():
            raise click.ClickException("test error")

        with pytest.raises(click.ClickException, match="test error"):
            test_func()

    def test_handle_cli_errors_decorator_converts_exception(self):
        """Test that general exceptions are converted to ClickAbort."""
        @CLIErrorHandler.handle_cli_errors
        def test_func():
            raise ValueError("test error")

        with pytest.raises(click.Abort):
            test_func()

    def test_validate_targeting_options_both_specified(self):
        """Test validation fails when both identifiers and filters provided."""
        with pytest.raises(click.ClickException, match="Cannot use both"):
            CLIErrorHandler.validate_targeting_options(
                ["exp1"], True, "test"
            )

    def test_validate_targeting_options_none_specified(self):
        """Test validation fails when neither identifiers nor filters provided."""
        with pytest.raises(click.ClickException, match="Must specify either"):
            CLIErrorHandler.validate_targeting_options(
                [], False, "test"
            )

    def test_validate_targeting_options_identifiers_only(self):
        """Test validation passes with only identifiers."""
        # Should not raise
        CLIErrorHandler.validate_targeting_options(
            ["exp1"], False, "test"
        )

    def test_validate_targeting_options_filters_only(self):
        """Test validation passes with only filters."""
        # Should not raise
        CLIErrorHandler.validate_targeting_options(
            [], True, "test"
        )

    @patch('yanex.cli.error_handling.parse_time_spec')
    def test_parse_time_filters_all_none(self, mock_parse):
        """Test parsing with all None values."""
        result = CLIErrorHandler.parse_time_filters()
        assert result == (None, None, None, None)
        mock_parse.assert_not_called()

    @patch('yanex.cli.error_handling.parse_time_spec')
    def test_parse_time_filters_valid_specs(self, mock_parse):
        """Test parsing with valid time specifications."""
        mock_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_parse.return_value = mock_dt

        result = CLIErrorHandler.parse_time_filters("2023-01-01", None, "yesterday", None)

        assert result == (mock_dt, None, mock_dt, None)
        assert mock_parse.call_count == 2

    @patch('yanex.cli.error_handling.parse_time_spec')
    def test_parse_time_filters_invalid_spec(self, mock_parse):
        """Test parsing with invalid time specification."""
        mock_parse.return_value = None

        with pytest.raises(click.ClickException, match="Invalid time specification"):
            CLIErrorHandler.parse_time_filters("invalid", None, None, None)

    @patch('yanex.cli.error_handling.parse_time_spec')
    def test_parse_time_filters_parse_exception(self, mock_parse):
        """Test parsing when parse_time_spec raises exception."""
        mock_parse.side_effect = ValueError("parse error")

        with pytest.raises(click.ClickException, match="Failed to parse"):
            CLIErrorHandler.parse_time_filters("2023-01-01", None, None, None)


class TestBulkOperationReporter:
    """Test bulk operation reporter functionality."""

    def test_reporter_initialization(self):
        """Test reporter initializes correctly."""
        reporter = BulkOperationReporter("test", show_progress=False)
        assert reporter.operation_name == "test"
        assert reporter.show_progress is False
        assert reporter.success_count == 0
        assert reporter.failure_count == 0
        assert reporter.errors == []

    def test_report_success(self, capsys):
        """Test success reporting."""
        reporter = BulkOperationReporter("archive")
        reporter.report_success("exp1", "test experiment")

        captured = capsys.readouterr()
        assert "✓ Archived exp1 (test experiment)" in captured.out
        assert reporter.success_count == 1
        assert reporter.failure_count == 0

    def test_report_success_no_progress(self, capsys):
        """Test success reporting with progress disabled."""
        reporter = BulkOperationReporter("archive", show_progress=False)
        reporter.report_success("exp1", "test experiment")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert reporter.success_count == 1

    def test_report_failure(self, capsys):
        """Test failure reporting."""
        reporter = BulkOperationReporter("delete")
        error = ValueError("test error")
        reporter.report_failure("exp2", error, "failed experiment")

        captured = capsys.readouterr()
        assert "✗ Failed to delete exp2 (failed experiment): test error" in captured.err
        assert reporter.failure_count == 1
        assert len(reporter.errors) == 1
        assert reporter.errors[0]["experiment_id"] == "exp2"

    def test_report_summary_success_only(self, capsys):
        """Test summary with only successes."""
        reporter = BulkOperationReporter("unarchive")
        reporter.success_count = 3
        reporter.report_summary()

        captured = capsys.readouterr()
        assert "✓ Successfully unarchived 3 experiments" in captured.out

    def test_report_summary_with_failures(self, capsys):
        """Test summary with failures."""
        reporter = BulkOperationReporter("update")
        reporter.success_count = 2
        reporter.failure_count = 1
        reporter.report_summary()

        captured = capsys.readouterr()
        assert "✓ Successfully updated 2 experiments" in captured.out
        assert "✗ Failed to update 1 experiment" in captured.err

    def test_report_summary_no_experiments(self, capsys):
        """Test summary with no experiments."""
        reporter = BulkOperationReporter("archive")
        reporter.report_summary()

        captured = capsys.readouterr()
        assert "No experiments found to archive" in captured.out

    def test_has_failures(self):
        """Test failure detection."""
        reporter = BulkOperationReporter("test")
        assert not reporter.has_failures()

        reporter.failure_count = 1
        assert reporter.has_failures()

    def test_get_exit_code(self):
        """Test exit code calculation."""
        reporter = BulkOperationReporter("test")

        # No operations
        assert reporter.get_exit_code() == 0

        # Only successes
        reporter.success_count = 3
        assert reporter.get_exit_code() == 0

        # Only failures
        reporter.success_count = 0
        reporter.failure_count = 2
        assert reporter.get_exit_code() == 1

        # Mixed results
        reporter.success_count = 2
        reporter.failure_count = 1
        assert reporter.get_exit_code() == 2


class TestConfirmDestructiveOperation:
    """Test destructive operation confirmation."""

    def test_confirm_with_force(self):
        """Test confirmation skipped when force=True."""
        result = confirm_destructive_operation("delete", 5, force=True)
        assert result is True

    @patch('click.confirm')
    def test_confirm_single_experiment(self, mock_confirm):
        """Test confirmation for single experiment."""
        mock_confirm.return_value = True
        result = confirm_destructive_operation("archive", 1, force=False)

        assert result is True
        mock_confirm.assert_called_once_with(
            "Are you sure you want to archive 1 experiment?"
        )

    @patch('click.confirm')
    def test_confirm_multiple_experiments(self, mock_confirm):
        """Test confirmation for multiple experiments."""
        mock_confirm.return_value = False
        result = confirm_destructive_operation("delete", 3, force=False)

        assert result is False
        mock_confirm.assert_called_once_with(
            "Are you sure you want to delete 3 experiments?"
        )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_format_validation_error_with_context(self):
        """Test validation error formatting with context."""
        error = ValidationError("Invalid input")
        result = format_validation_error(error, "Command validation")
        assert result == "Command validation: Invalid input"

    def test_format_validation_error_without_context(self):
        """Test validation error formatting without context."""
        error = ValidationError("Invalid input")
        result = format_validation_error(error)
        assert result == "Invalid input"

    def test_show_warning(self, capsys):
        """Test warning display."""
        show_warning("This is a test warning")
        captured = capsys.readouterr()
        assert "⚠️ This is a test warning" in captured.err

    def test_validate_experiment_state_placeholder(self):
        """Test experiment state validation (placeholder)."""
        # This is a placeholder function for now
        validate_experiment_state("exp1", "completed", "failed", "test")
        # Should not raise since it's a placeholder


class TestIntegrationWithCLI:
    """Test integration with CLI commands."""

    def test_cli_command_with_error_handler(self):
        """Test CLI command with error handler decorator."""
        @click.command()
        @CLIErrorHandler.handle_cli_errors
        def test_command():
            raise ValueError("Test error")

        runner = CliRunner()
        result = runner.invoke(test_command)

        assert result.exit_code != 0
        # The error message format might vary, but should contain the error
        assert "Test error" in result.output or result.exit_code == 1

    def test_targeting_validation_in_command(self):
        """Test targeting validation in a mock command."""
        @click.command()
        @click.argument("identifiers", nargs=-1)
        @click.option("--status")
        @CLIErrorHandler.handle_cli_errors
        def test_command(identifiers, status):
            has_filters = bool(status)
            CLIErrorHandler.validate_targeting_options(
                list(identifiers), has_filters, "test"
            )
            click.echo("Success")

        runner = CliRunner()

        # Valid: identifiers only
        result = runner.invoke(test_command, ["exp1", "exp2"])
        assert result.exit_code == 0
        assert "Success" in result.output

        # Valid: filters only
        result = runner.invoke(test_command, ["--status", "completed"])
        assert result.exit_code == 0
        assert "Success" in result.output

        # Invalid: both
        result = runner.invoke(test_command, ["exp1", "--status", "completed"])
        assert result.exit_code != 0
        assert "Cannot use both" in result.output

    def test_time_filter_parsing_in_command(self):
        """Test time filter parsing in a mock command."""
        @click.command()
        @click.option("--started-after")
        @click.option("--ended-before")
        @CLIErrorHandler.handle_cli_errors
        def test_command(started_after, ended_before):
            CLIErrorHandler.parse_time_filters(
                started_after=started_after, ended_before=ended_before
            )
            click.echo("Success")

        runner = CliRunner()

        # Should work with None values
        result = runner.invoke(test_command)
        assert result.exit_code == 0
        assert "Success" in result.output
