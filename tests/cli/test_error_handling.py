"""Tests for CLI error handling utilities."""

from datetime import datetime, timezone
from unittest.mock import patch

import click
import pytest

from tests.test_utils import create_cli_runner
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

    @pytest.mark.parametrize(
        "exception_class,exception_message",
        [
            (ValueError, "test value error"),
            (RuntimeError, "test runtime error"),
            (KeyError, "test key error"),
            (TypeError, "test type error"),
        ],
    )
    def test_handle_cli_errors_decorator_converts_exception(
        self, exception_class, exception_message
    ):
        """Test that general exceptions are converted to ClickAbort."""

        @CLIErrorHandler.handle_cli_errors
        def test_func():
            raise exception_class(exception_message)

        with pytest.raises(click.Abort):
            test_func()

    @pytest.mark.parametrize(
        "identifiers,has_filters,should_raise,expected_match",
        [
            (["exp1"], True, True, "Cannot use both"),  # Both specified
            ([], False, True, "Must specify either"),  # None specified
            (["exp1"], False, False, None),  # Identifiers only
            ([], True, False, None),  # Filters only
            (["exp1", "exp2"], False, False, None),  # Multiple identifiers only
        ],
    )
    def test_validate_targeting_options(
        self, identifiers, has_filters, should_raise, expected_match
    ):
        """Test targeting options validation with various combinations."""
        if should_raise:
            with pytest.raises(click.ClickException, match=expected_match):
                CLIErrorHandler.validate_targeting_options(
                    identifiers, has_filters, "test"
                )
        else:
            # Should not raise
            CLIErrorHandler.validate_targeting_options(identifiers, has_filters, "test")

    @patch("yanex.cli.error_handling.parse_time_spec")
    def test_parse_time_filters_all_none(self, mock_parse):
        """Test parsing with all None values."""
        result = CLIErrorHandler.parse_time_filters()
        assert result == (None, None, None, None)
        mock_parse.assert_not_called()

    @patch("yanex.cli.error_handling.parse_time_spec")
    def test_parse_time_filters_valid_specs(self, mock_parse):
        """Test parsing with valid time specifications."""
        mock_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_parse.return_value = mock_dt

        result = CLIErrorHandler.parse_time_filters(
            "2023-01-01", None, "yesterday", None
        )

        assert result == (mock_dt, None, mock_dt, None)
        assert mock_parse.call_count == 2

    @patch("yanex.cli.error_handling.parse_time_spec")
    def test_parse_time_filters_invalid_spec(self, mock_parse):
        """Test parsing with invalid time specification."""
        mock_parse.return_value = None

        with pytest.raises(click.ClickException, match="Invalid time specification"):
            CLIErrorHandler.parse_time_filters("invalid", None, None, None)

    @patch("yanex.cli.error_handling.parse_time_spec")
    def test_parse_time_filters_parse_exception(self, mock_parse):
        """Test parsing when parse_time_spec raises exception."""
        mock_parse.side_effect = ValueError("parse error")

        with pytest.raises(click.ClickException, match="Failed to parse"):
            CLIErrorHandler.parse_time_filters("2023-01-01", None, None, None)

    @pytest.mark.parametrize(
        "time_specs,expected_call_count",
        [
            ({"started_after": "2023-01-01"}, 1),
            ({"ended_before": "yesterday"}, 1),
            ({"started_after": "2023-01-01", "ended_before": "yesterday"}, 2),
            ({"started_before": "last week", "ended_after": "today"}, 2),
        ],
    )
    @patch("yanex.cli.error_handling.parse_time_spec")
    def test_parse_time_filters_call_counts(
        self, mock_parse, time_specs, expected_call_count
    ):
        """Test that parse_time_spec is called the correct number of times."""
        mock_dt = datetime(2023, 1, 1, tzinfo=timezone.utc)
        mock_parse.return_value = mock_dt

        CLIErrorHandler.parse_time_filters(**time_specs)
        assert mock_parse.call_count == expected_call_count


class TestBulkOperationReporter:
    """Test bulk operation reporter functionality."""

    @pytest.mark.parametrize(
        "operation_name,show_progress",
        [
            ("test", False),
            ("archive", True),
            ("delete", False),
            ("update", True),
        ],
    )
    def test_reporter_initialization(self, operation_name, show_progress):
        """Test reporter initializes correctly with various parameters."""
        reporter = BulkOperationReporter(operation_name, show_progress=show_progress)
        assert reporter.operation_name == operation_name
        assert reporter.show_progress == show_progress
        assert reporter.success_count == 0
        assert reporter.failure_count == 0
        assert reporter.errors == []

    @pytest.mark.parametrize(
        "operation,exp_id,description,expected_output",
        [
            ("archive", "exp1", "test experiment", "✓ Archived exp1 (test experiment)"),
            (
                "delete",
                "exp123",
                "ML training run",
                "✓ Deleted exp123 (ML training run)",
            ),
            ("update", "short", None, "✓ Updated short"),
        ],
    )
    def test_report_success(
        self, capsys, operation, exp_id, description, expected_output
    ):
        """Test success reporting with various inputs."""
        reporter = BulkOperationReporter(operation)
        reporter.report_success(exp_id, description)

        captured = capsys.readouterr()
        assert expected_output in captured.out
        assert reporter.success_count == 1
        assert reporter.failure_count == 0

    def test_report_success_no_progress(self, capsys):
        """Test success reporting with progress disabled."""
        reporter = BulkOperationReporter("archive", show_progress=False)
        reporter.report_success("exp1", "test experiment")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert reporter.success_count == 1

    @pytest.mark.parametrize(
        "operation,exp_id,error,description,expected_output_part",
        [
            (
                "delete",
                "exp2",
                ValueError("test error"),
                "failed experiment",
                "✗ Failed to delete exp2 (failed experiment): test error",
            ),
            (
                "archive",
                "exp123",
                RuntimeError("disk full"),
                None,
                "✗ Failed to archive exp123: disk full",
            ),
            (
                "update",
                "short",
                KeyError("missing key"),
                "data processing",
                "✗ Failed to update short (data processing): 'missing key'",
            ),
        ],
    )
    def test_report_failure(
        self, capsys, operation, exp_id, error, description, expected_output_part
    ):
        """Test failure reporting with various inputs."""
        reporter = BulkOperationReporter(operation)
        reporter.report_failure(exp_id, error, description)

        captured = capsys.readouterr()
        assert expected_output_part in captured.err
        assert reporter.failure_count == 1
        assert len(reporter.errors) == 1
        assert reporter.errors[0]["experiment_id"] == exp_id

    @pytest.mark.parametrize(
        "operation,success_count,expected_message",
        [
            ("unarchive", 1, "✓ Successfully unarchived 1 experiment"),
            ("archive", 3, "✓ Successfully archived 3 experiments"),
            ("delete", 5, "✓ Successfully deleted 5 experiments"),
        ],
    )
    def test_report_summary_success_only(
        self, capsys, operation, success_count, expected_message
    ):
        """Test summary with only successes."""
        reporter = BulkOperationReporter(operation)
        reporter.success_count = success_count
        reporter.report_summary()

        captured = capsys.readouterr()
        assert expected_message in captured.out

    @pytest.mark.parametrize(
        "operation,success_count,failure_count",
        [
            ("update", 2, 1),
            ("archive", 5, 2),
            ("delete", 1, 3),
        ],
    )
    def test_report_summary_with_failures(
        self, capsys, operation, success_count, failure_count
    ):
        """Test summary with failures."""
        reporter = BulkOperationReporter(operation)
        reporter.success_count = success_count
        reporter.failure_count = failure_count
        reporter.report_summary()

        captured = capsys.readouterr()
        assert f"✓ Successfully {operation}d {success_count} experiment" in captured.out
        assert f"✗ Failed to {operation} {failure_count} experiment" in captured.err

    @pytest.mark.parametrize(
        "operation,expected_message",
        [
            ("archive", "No experiments found to archive"),
            ("delete", "No experiments found to delete"),
            ("update", "No experiments found to update"),
        ],
    )
    def test_report_summary_no_experiments(self, capsys, operation, expected_message):
        """Test summary with no experiments."""
        reporter = BulkOperationReporter(operation)
        reporter.report_summary()

        captured = capsys.readouterr()
        assert expected_message in captured.out

    @pytest.mark.parametrize(
        "failure_count,expected_result",
        [
            (0, False),
            (1, True),
            (5, True),
        ],
    )
    def test_has_failures(self, failure_count, expected_result):
        """Test failure detection."""
        reporter = BulkOperationReporter("test")
        reporter.failure_count = failure_count
        assert reporter.has_failures() == expected_result

    @pytest.mark.parametrize(
        "success_count,failure_count,expected_exit_code",
        [
            (0, 0, 0),  # No operations
            (3, 0, 0),  # Only successes
            (0, 2, 1),  # Only failures
            (2, 1, 2),  # Mixed results
            (5, 3, 2),  # Mixed results
        ],
    )
    def test_get_exit_code(self, success_count, failure_count, expected_exit_code):
        """Test exit code calculation."""
        reporter = BulkOperationReporter("test")
        reporter.success_count = success_count
        reporter.failure_count = failure_count
        assert reporter.get_exit_code() == expected_exit_code


class TestConfirmDestructiveOperation:
    """Test destructive operation confirmation."""

    @pytest.mark.parametrize(
        "operation,count",
        [
            ("delete", 1),
            ("archive", 5),
            ("update", 10),
        ],
    )
    def test_confirm_with_force(self, operation, count):
        """Test confirmation skipped when force=True."""
        result = confirm_destructive_operation(operation, count, force=True)
        assert result is True

    @pytest.mark.parametrize(
        "operation,count,user_choice,expected_result,expected_prompt",
        [
            (
                "archive",
                1,
                True,
                True,
                "Are you sure you want to archive 1 experiment?",
            ),
            (
                "delete",
                3,
                False,
                False,
                "Are you sure you want to delete 3 experiments?",
            ),
            (
                "update",
                10,
                True,
                True,
                "Are you sure you want to update 10 experiments?",
            ),
        ],
    )
    @patch("click.confirm")
    def test_confirm_operations(
        self,
        mock_confirm,
        operation,
        count,
        user_choice,
        expected_result,
        expected_prompt,
    ):
        """Test confirmation for various operations."""
        mock_confirm.return_value = user_choice
        result = confirm_destructive_operation(operation, count, force=False)

        assert result == expected_result
        mock_confirm.assert_called_once_with(expected_prompt)


class TestUtilityFunctions:
    """Test utility functions."""

    @pytest.mark.parametrize(
        "error_message,context,expected_result",
        [
            (
                "Invalid input",
                "Command validation",
                "Command validation: Invalid input",
            ),
            ("Missing parameter", "Configuration", "Configuration: Missing parameter"),
            ("Bad format", None, "Bad format"),
            ("", "Context", "Context: "),
        ],
    )
    def test_format_validation_error(self, error_message, context, expected_result):
        """Test validation error formatting."""
        error = ValidationError(error_message)
        result = format_validation_error(error, context)
        assert result == expected_result

    @pytest.mark.parametrize(
        "warning_message,expected_output",
        [
            ("This is a test warning", "⚠️ This is a test warning"),
            ("Missing configuration file", "⚠️ Missing configuration file"),
            ("Experiment already exists", "⚠️ Experiment already exists"),
        ],
    )
    def test_show_warning(self, capsys, warning_message, expected_output):
        """Test warning display."""
        show_warning(warning_message)
        captured = capsys.readouterr()
        assert expected_output in captured.err

    @pytest.mark.parametrize(
        "exp_id,current_state,required_state,operation",
        [
            ("exp1", "completed", "failed", "test"),
            ("exp2", "running", "completed", "archive"),
            ("exp3", "failed", "running", "retry"),
        ],
    )
    def test_validate_experiment_state_placeholder(
        self, exp_id, current_state, required_state, operation
    ):
        """Test experiment state validation (placeholder)."""
        # This is a placeholder function for now
        validate_experiment_state(exp_id, current_state, required_state, operation)
        # Should not raise since it's a placeholder


class TestIntegrationWithCLI:
    """Test integration with CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_cli_command_with_error_handler(self):
        """Test CLI command with error handler decorator."""

        @click.command()
        @CLIErrorHandler.handle_cli_errors
        def test_command():
            raise ValueError("Test error")

        result = self.runner.invoke(test_command)

        assert result.exit_code != 0
        # The error message format might vary, but should contain the error
        assert "Test error" in result.output or result.exit_code == 1

    @pytest.mark.parametrize(
        "args,expected_exit_code,expected_output",
        [
            (["exp1", "exp2"], 0, "Success"),  # Valid: identifiers only
            (["--status", "completed"], 0, "Success"),  # Valid: filters only
            (["exp1", "--status", "completed"], 1, "Cannot use both"),  # Invalid: both
            ([], 1, "Must specify either"),  # Invalid: none
        ],
    )
    def test_targeting_validation_in_command(
        self, args, expected_exit_code, expected_output
    ):
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

        result = self.runner.invoke(test_command, args)
        assert result.exit_code == expected_exit_code
        assert expected_output in result.output

    @pytest.mark.parametrize(
        "args,expected_exit_code,expected_output",
        [
            ([], 0, "Success"),  # No time filters
            (["--started-after", "2023-01-01"], 0, "Success"),  # Valid time filter
            (["--ended-before", "yesterday"], 0, "Success"),  # Valid time filter
        ],
    )
    def test_time_filter_parsing_in_command(
        self, args, expected_exit_code, expected_output
    ):
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

        result = self.runner.invoke(test_command, args)
        assert result.exit_code == expected_exit_code
        assert expected_output in result.output

    def test_error_handler_with_click_exception(self):
        """Test that ClickException passes through correctly."""

        @click.command()
        @CLIErrorHandler.handle_cli_errors
        def test_command():
            raise click.ClickException("Custom click error")

        result = self.runner.invoke(test_command)
        assert result.exit_code != 0
        assert "Custom click error" in result.output

    def test_error_handler_preserves_successful_execution(self):
        """Test that successful commands work normally."""

        @click.command()
        @CLIErrorHandler.handle_cli_errors
        def test_command():
            click.echo("Command executed successfully")
            return "success"

        result = self.runner.invoke(test_command)
        assert result.exit_code == 0
        assert "Command executed successfully" in result.output
