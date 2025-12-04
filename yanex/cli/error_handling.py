"""Centralized error handling utilities for CLI commands."""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import click

from ..cli.filters.time_utils import parse_time_spec
from ..utils.exceptions import ValidationError
from .formatters import (
    FAILURE_SYMBOL as _FAILURE_SYMBOL,
)
from .formatters import (
    SUCCESS_SYMBOL as _SUCCESS_SYMBOL,
)
from .formatters import (
    WARNING_SYMBOL as _WARNING_SYMBOL,
)
from .formatters import (
    OutputFormat,
    OutputMode,
    format_action_result,
    format_action_result_markdown,
    is_machine_format,
    output_action_result_csv,
    output_json,
)

F = TypeVar("F", bound=Callable[..., Any])


class CLIErrorHandler:
    """Centralized error handling for CLI commands."""

    # Standard error message prefixes
    ERROR_PREFIX = "Error:"
    # Use imported symbols from theme for consistency
    WARNING_PREFIX = _WARNING_SYMBOL
    SUCCESS_SYMBOL = _SUCCESS_SYMBOL
    FAILURE_SYMBOL = _FAILURE_SYMBOL

    @staticmethod
    def handle_cli_errors(func: F) -> F:
        """
        Decorator for consistent CLI error handling.

        Catches all exceptions and formats them consistently, ensuring proper exit codes.
        Preserves click.ClickException for proper CLI error handling.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except click.ClickException:
                # Re-raise ClickException to preserve click's error handling
                raise
            except Exception as e:
                # Format and display error, then exit with code 1
                click.echo(f"{CLIErrorHandler.ERROR_PREFIX} {e}", err=True)
                raise click.Abort() from e

        return wrapper

    @staticmethod
    def validate_targeting_options(
        experiment_identifiers: list[str],
        has_filters: bool,
        operation_name: str = "operation",
    ) -> None:
        """
        Validate mutual exclusivity between experiment identifiers and filters.

        Args:
            experiment_identifiers: List of experiment IDs/patterns
            has_filters: Whether any filter options are provided
            operation_name: Name of the operation for error messages

        Raises:
            click.ClickException: If validation fails
        """
        has_identifiers = len(experiment_identifiers) > 0

        if has_identifiers and has_filters:
            raise click.ClickException(
                f"Cannot use both experiment identifiers and filter options for {operation_name}. "
                "Choose one approach."
            )

        if not has_identifiers and not has_filters:
            raise click.ClickException(
                f"Must specify either experiment identifiers or filter options for {operation_name}."
            )

    @staticmethod
    def parse_time_filters(
        started_after: str | None = None,
        started_before: str | None = None,
        ended_after: str | None = None,
        ended_before: str | None = None,
    ) -> tuple:
        """
        Parse time filter specifications with consistent error handling.

        Args:
            started_after: Started after time specification
            started_before: Started before time specification
            ended_after: Ended after time specification
            ended_before: Ended before time specification

        Returns:
            Tuple of parsed datetime objects (or None if not provided)

        Raises:
            click.ClickException: If any time specification is invalid
        """
        results = []
        time_specs = [
            ("started-after", started_after),
            ("started-before", started_before),
            ("ended-after", ended_after),
            ("ended-before", ended_before),
        ]

        for spec_name, spec_value in time_specs:
            if spec_value is None:
                results.append(None)
                continue

            try:
                parsed_time = parse_time_spec(spec_value)
                if parsed_time is None:
                    raise click.ClickException(
                        f"Invalid time specification for --{spec_name}: '{spec_value}'"
                    )
                results.append(parsed_time)
            except Exception as e:
                raise click.ClickException(
                    f"Failed to parse --{spec_name} '{spec_value}': {e}"
                ) from e

        return tuple(results)


class BulkOperationReporter:
    """Handles consistent progress reporting for bulk operations.

    Supports multiple output modes: console (default), JSON, CSV, and markdown.
    For machine-readable modes (JSON/CSV), progress messages are suppressed
    and only the final summary is output to stdout.
    """

    def __init__(
        self,
        operation_name: str,
        output_mode: OutputMode | None = None,
        show_progress: bool = True,
        output_format: OutputFormat | None = None,
    ):
        """
        Initialize bulk operation reporter.

        Args:
            operation_name: Name of the operation (e.g., "archive", "delete")
            output_mode: Legacy output mode (deprecated, use output_format)
            show_progress: Whether to show individual operation progress
            output_format: Output format (preferred over output_mode)
        """
        self.operation_name = operation_name

        # Support both OutputFormat (new) and OutputMode (legacy)
        if output_format is not None:
            self.output_format = output_format
            # Map OutputFormat to OutputMode for backward compatibility
            mode_map = {
                OutputFormat.DEFAULT: OutputMode.CONSOLE,
                OutputFormat.JSON: OutputMode.JSON,
                OutputFormat.CSV: OutputMode.CSV,
                OutputFormat.MARKDOWN: OutputMode.MARKDOWN,
                OutputFormat.SWEEP: OutputMode.CSV,  # SWEEP is similar to CSV
            }
            self.output_mode = mode_map.get(output_format, OutputMode.CONSOLE)
        elif output_mode is not None:
            self.output_mode = output_mode
            # Map OutputMode to OutputFormat
            format_map = {
                OutputMode.CONSOLE: OutputFormat.DEFAULT,
                OutputMode.JSON: OutputFormat.JSON,
                OutputMode.CSV: OutputFormat.CSV,
                OutputMode.MARKDOWN: OutputFormat.MARKDOWN,
            }
            self.output_format = format_map.get(output_mode, OutputFormat.DEFAULT)
        else:
            self.output_mode = OutputMode.CONSOLE
            self.output_format = OutputFormat.DEFAULT

        # Suppress progress for machine-readable output formats
        self.show_progress = show_progress and not is_machine_format(self.output_format)
        self.success_count = 0
        self.failure_count = 0
        self.successful_ids: list[str] = []
        self.errors: list[dict[str, str]] = []

    def report_success(
        self, experiment_id: str, experiment_name: str | None = None
    ) -> None:
        """Report successful operation on an experiment."""
        self.success_count += 1
        self.successful_ids.append(experiment_id)

        if self.show_progress:
            name_part = f" ({experiment_name})" if experiment_name else ""
            click.echo(
                f"  {CLIErrorHandler.SUCCESS_SYMBOL} {self.operation_name.title()}d {experiment_id}{name_part}"
            )

    def report_failure(
        self, experiment_id: str, error: Exception, experiment_name: str | None = None
    ) -> None:
        """Report failed operation on an experiment."""
        self.failure_count += 1

        # Include both 'id' and 'experiment_id' for backward compatibility
        self.errors.append(
            {"id": experiment_id, "experiment_id": experiment_id, "error": str(error)}
        )

        if self.show_progress:
            name_part = f" ({experiment_name})" if experiment_name else ""
            error_msg = f"  {CLIErrorHandler.FAILURE_SYMBOL} Failed to {self.operation_name} {experiment_id}{name_part}: {error}"
            click.echo(error_msg, err=True)

    def report_summary(self) -> None:
        """Report final summary of the bulk operation based on output mode."""
        total = self.success_count + self.failure_count

        if self.output_mode == OutputMode.JSON:
            self._report_summary_json()
        elif self.output_mode == OutputMode.CSV:
            self._report_summary_csv()
        elif self.output_mode == OutputMode.MARKDOWN:
            self._report_summary_markdown()
        else:
            self._report_summary_console(total)

    def _report_summary_json(self) -> None:
        """Output summary as JSON."""
        result = format_action_result(
            operation=self.operation_name,
            success=self.failure_count == 0,
            experiments=self.successful_ids,
            failed=self.errors,
        )
        output_json(result)

    def _report_summary_csv(self) -> None:
        """Output summary as CSV."""
        output_action_result_csv(
            operation=self.operation_name,
            experiments=self.successful_ids,
            failed=self.errors,
        )

    def _report_summary_markdown(self) -> None:
        """Output summary as markdown."""
        markdown = format_action_result_markdown(
            operation=self.operation_name,
            success_count=self.success_count,
            failed_count=self.failure_count,
            experiments=self.successful_ids,
            failed=self.errors,
        )
        click.echo(markdown)

    def _report_summary_console(self, total: int) -> None:
        """Output summary in console format."""
        if total == 0:
            click.echo(f"No experiments found to {self.operation_name}.")
            return

        success_msg = f"{CLIErrorHandler.SUCCESS_SYMBOL} Successfully {self.operation_name}d {self.success_count} experiment"
        if self.success_count != 1:
            success_msg += "s"

        if self.failure_count > 0:
            failure_msg = f"{CLIErrorHandler.FAILURE_SYMBOL} Failed to {self.operation_name} {self.failure_count} experiment"
            if self.failure_count != 1:
                failure_msg += "s"
            click.echo(f"\n{success_msg}")
            click.echo(failure_msg, err=True)
        else:
            click.echo(f"\n{success_msg}")

    def has_failures(self) -> bool:
        """Check if any operations failed."""
        return self.failure_count > 0

    def get_exit_code(self) -> int:
        """Get appropriate exit code based on operation results."""
        if self.failure_count > 0:
            return 1 if self.success_count == 0 else 2  # 2 for partial success
        return 0


def validate_experiment_state(
    experiment_id: str,
    required_status: str | None = None,
    forbidden_status: str | None = None,
    operation_name: str = "operation",
) -> None:
    """
    Validate experiment state for operations.

    Args:
        experiment_id: Experiment identifier
        required_status: Status required for the operation
        forbidden_status: Status that prevents the operation
        operation_name: Name of the operation for error messages

    Raises:
        ValidationError: If experiment state is invalid for the operation
    """
    # This would integrate with the experiment manager to check status
    # For now, it's a placeholder for the validation framework
    pass


def format_validation_error(error: ValidationError, context: str = "") -> str:
    """
    Format validation errors consistently.

    Args:
        error: Validation error to format
        context: Additional context for the error

    Returns:
        Formatted error message
    """
    base_msg = str(error)
    if context:
        return f"{context}: {base_msg}"
    return base_msg


def show_warning(message: str) -> None:
    """Display a standardized warning message."""
    click.echo(f"{_WARNING_SYMBOL} {message}", err=True)


def confirm_destructive_operation(
    operation_name: str, experiment_count: int, force: bool = False
) -> bool:
    """
    Confirm destructive operations with consistent messaging.

    Args:
        operation_name: Name of the destructive operation
        experiment_count: Number of experiments affected
        force: Whether to skip confirmation

    Returns:
        True if confirmed, False otherwise
    """
    if force:
        return True

    exp_text = "experiment" if experiment_count == 1 else "experiments"
    message = (
        f"Are you sure you want to {operation_name} {experiment_count} {exp_text}?"
    )

    return click.confirm(message)
