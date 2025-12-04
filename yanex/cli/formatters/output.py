"""Output mode handling for yanex CLI commands.

This module provides infrastructure for supporting multiple output formats
(console, JSON, CSV, markdown) across CLI commands.
"""

from collections.abc import Callable
from enum import Enum

import click


class OutputMode(Enum):
    """Output format modes for CLI commands.

    DEPRECATED: Use OutputFormat instead. This class is kept for backwards
    compatibility during the transition period.
    """

    CONSOLE = "console"  # Rich console output (default)
    JSON = "json"  # Machine-readable JSON
    CSV = "csv"  # Machine-readable CSV
    MARKDOWN = "markdown"  # GitHub-flavored markdown


class OutputFormat(str, Enum):
    """Output format options for CLI commands.

    This is the new unified output format enum that replaces OutputMode.
    """

    DEFAULT = "default"  # Human-readable console output
    JSON = "json"  # Machine-readable JSON
    CSV = "csv"  # True CSV with ID,value columns
    MARKDOWN = "markdown"  # GitHub-flavored markdown table
    SWEEP = "sweep"  # Comma-separated values only (for bash substitution)


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


def format_options(include_sweep: bool = False) -> Callable:
    """Decorator adding --format option with optional legacy flags.

    This is the new unified approach replacing output_mode_options.

    Args:
        include_sweep: Whether to include 'sweep' format option (for yanex get)

    Usage:
        @click.command()
        @format_options()
        def my_command(output_format, json_flag, csv_flag, markdown_flag, ...):
            fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
            ...

        # For yanex get (includes sweep format):
        @click.command()
        @format_options(include_sweep=True)
        def get_field(...):
            ...
    """
    choices = ["default", "json", "csv", "markdown"]
    if include_sweep:
        choices.append("sweep")

    def decorator(func: Callable) -> Callable:
        # Add --format option
        func = click.option(
            "--format",
            "-F",
            "output_format",
            type=click.Choice(choices, case_sensitive=False),
            default=None,
            help="Output format (default: default)",
        )(func)
        # Keep legacy flags as hidden aliases for backwards compatibility
        func = click.option(
            "--json",
            "-j",
            "json_flag",
            is_flag=True,
            hidden=True,
            help="Output as JSON (alias for --format json)",
        )(func)
        func = click.option(
            "--csv",
            "csv_flag",
            is_flag=True,
            hidden=True,
            help="Output as CSV (alias for --format csv)",
        )(func)
        func = click.option(
            "--markdown",
            "-m",
            "markdown_flag",
            is_flag=True,
            hidden=True,
            help="Output as markdown (alias for --format markdown)",
        )(func)
        return func

    return decorator


def resolve_output_format(
    output_format: str | None,
    json_flag: bool = False,
    csv_flag: bool = False,
    markdown_flag: bool = False,
    csv_means_sweep: bool = False,
) -> OutputFormat:
    """Resolve output format from --format option or legacy flags.

    Args:
        output_format: Value from --format option (or None if not provided)
        json_flag: Whether --json flag was provided
        csv_flag: Whether --csv flag was provided
        markdown_flag: Whether --markdown flag was provided
        csv_means_sweep: If True, legacy --csv maps to SWEEP (for backwards compat)

    Returns:
        Resolved OutputFormat

    Raises:
        click.ClickException: If multiple output formats are specified
    """
    # Count how many options are specified
    flag_count = sum([json_flag, csv_flag, markdown_flag])
    format_specified = output_format is not None

    if format_specified and flag_count > 0:
        raise click.ClickException(
            "Cannot use --format with legacy flags (--json, --csv, --markdown). "
            "Use --format only."
        )

    if flag_count > 1:
        raise click.ClickException(
            "Cannot specify multiple output formats. "
            "Choose one of --json, --csv, or --markdown."
        )

    # Resolve from --format option
    if output_format:
        return OutputFormat(output_format.lower())

    # Resolve from legacy flags
    if json_flag:
        return OutputFormat.JSON
    if csv_flag:
        # For yanex get, legacy --csv maps to SWEEP for bash substitution
        return OutputFormat.SWEEP if csv_means_sweep else OutputFormat.CSV
    if markdown_flag:
        return OutputFormat.MARKDOWN

    return OutputFormat.DEFAULT


def is_machine_format(fmt: OutputFormat) -> bool:
    """Check if output format requires clean stdout for piping.

    Machine-readable formats (JSON, CSV, SWEEP) should only output valid data
    to stdout. Any informational messages should go to stderr.

    Args:
        fmt: The current output format

    Returns:
        True for JSON, CSV, and SWEEP formats, False otherwise
    """
    return fmt in (OutputFormat.JSON, OutputFormat.CSV, OutputFormat.SWEEP)


def echo_format_info(message: str, fmt: OutputFormat) -> None:
    """Print informational message, respecting output format.

    For machine-readable formats (JSON/CSV/SWEEP), messages go to stderr to keep
    stdout clean for piping. For other formats, messages go to stdout.

    Args:
        message: The message to print
        fmt: The current output format
    """
    if is_machine_format(fmt):
        click.echo(message, err=True)
    else:
        click.echo(message)
