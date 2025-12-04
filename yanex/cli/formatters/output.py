"""Output mode handling for yanex CLI commands.

This module provides infrastructure for supporting multiple output formats
(console, JSON, CSV, markdown) across CLI commands.
"""

from enum import Enum

import click


class OutputMode(Enum):
    """Output format modes for CLI commands."""

    CONSOLE = "console"  # Rich console output (default)
    JSON = "json"  # Machine-readable JSON
    CSV = "csv"  # Machine-readable CSV
    MARKDOWN = "markdown"  # GitHub-flavored markdown


def output_mode_options(func):
    """Decorator to add --json, --csv, and --markdown options to a command.

    Usage:
        @click.command()
        @output_mode_options
        def my_command(json_output, csv_output, markdown_output, ...):
            mode = get_output_mode(ctx)
            ...
    """
    func = click.option(
        "--json",
        "-j",
        "json_output",
        is_flag=True,
        help="Output as JSON (machine-readable, pipeable)",
    )(func)
    func = click.option(
        "--csv",
        "csv_output",
        is_flag=True,
        help="Output as CSV (machine-readable, pipeable)",
    )(func)
    func = click.option(
        "--markdown",
        "-m",
        "markdown_output",
        is_flag=True,
        help="Output as GitHub-flavored markdown",
    )(func)
    return func


def get_output_mode(
    json_output: bool = False,
    csv_output: bool = False,
    markdown_output: bool = False,
) -> OutputMode:
    """Determine output mode from CLI flags.

    Args:
        json_output: Whether --json flag was provided
        csv_output: Whether --csv flag was provided
        markdown_output: Whether --markdown flag was provided

    Returns:
        OutputMode based on flags (priority: JSON > CSV > Markdown > Console)
    """
    if json_output:
        return OutputMode.JSON
    if csv_output:
        return OutputMode.CSV
    if markdown_output:
        return OutputMode.MARKDOWN
    return OutputMode.CONSOLE


def is_machine_output(mode: OutputMode) -> bool:
    """Check if output mode requires clean stdout for piping.

    Machine-readable modes (JSON, CSV) should only output valid data to stdout.
    Any informational messages should go to stderr.

    Args:
        mode: The current output mode

    Returns:
        True for JSON and CSV modes, False otherwise
    """
    return mode in (OutputMode.JSON, OutputMode.CSV)


def echo_info(message: str, mode: OutputMode) -> None:
    """Print informational message, respecting output mode.

    For machine-readable modes (JSON/CSV), messages go to stderr to keep
    stdout clean for piping. For other modes, messages go to stdout.

    Args:
        message: The message to print
        mode: The current output mode
    """
    if is_machine_output(mode):
        click.echo(message, err=True)
    else:
        click.echo(message)


def echo_error(message: str) -> None:
    """Print error message to stderr.

    Args:
        message: The error message to print
    """
    click.echo(message, err=True)


def validate_output_mode_flags(
    json_output: bool, csv_output: bool, markdown_output: bool
) -> None:
    """Validate that only one output mode flag is specified.

    Args:
        json_output: Whether --json flag was provided
        csv_output: Whether --csv flag was provided
        markdown_output: Whether --markdown flag was provided

    Raises:
        click.ClickException: If multiple output mode flags are specified
    """
    count = sum([json_output, csv_output, markdown_output])
    if count > 1:
        raise click.ClickException(
            "Cannot specify multiple output formats. "
            "Choose one of --json, --csv, or --markdown."
        )
