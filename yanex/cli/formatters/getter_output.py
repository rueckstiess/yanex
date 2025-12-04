"""Getter output formatting for yanex get command.

This module provides a unified abstraction for formatting getter output
across different getter types (scalar, list, complex, multiline) and
output formats (default, json, csv, markdown, sweep).

To add a new getter:
1. Add to GETTER_TYPES registry if it's a known field
2. Add resolve logic in get.py's resolve_field_value()

The getter automatically inherits correct behavior for all output formats.
"""

import json
from enum import Enum
from typing import Any

import click

from .output import OutputFormat


class GetterType(Enum):
    """Type of value returned by a getter."""

    SCALAR = "scalar"  # Single value (string, number, bool)
    LIST = "list"  # List of values (tags, param names, metric names)
    COMPLEX = "complex"  # Complex value (dict, nested structure)
    MULTILINE = "multiline"  # Multi-line text (stdout, stderr)


# Registry mapping field names to their getter types
GETTER_TYPES: dict[str, GetterType] = {
    # Scalar getters
    "id": GetterType.SCALAR,
    "name": GetterType.SCALAR,
    "status": GetterType.SCALAR,
    "description": GetterType.SCALAR,
    "script_path": GetterType.SCALAR,
    "created_at": GetterType.SCALAR,
    "started_at": GetterType.SCALAR,
    "completed_at": GetterType.SCALAR,
    "failed_at": GetterType.SCALAR,
    "cancelled_at": GetterType.SCALAR,
    "error_message": GetterType.SCALAR,
    "cancellation_reason": GetterType.SCALAR,
    "cli-command": GetterType.SCALAR,
    "run-command": GetterType.SCALAR,
    "experiment-dir": GetterType.SCALAR,
    "artifacts-dir": GetterType.SCALAR,
    "git.branch": GetterType.SCALAR,
    "git.commit_hash": GetterType.SCALAR,
    "git.dirty": GetterType.SCALAR,
    "git.remote_url": GetterType.SCALAR,
    # params.<key> and metrics.<key> are SCALAR (handled by prefix matching)
    # List getters
    "tags": GetterType.LIST,
    "params": GetterType.LIST,  # List of param names
    "metrics": GetterType.LIST,  # List of metric names
    # Complex getters
    "dependencies": GetterType.COMPLEX,
    # params.<nested_dict> returns COMPLEX (detected at runtime)
    # Multi-line getters
    "stdout": GetterType.MULTILINE,
    "stderr": GetterType.MULTILINE,
    "artifacts": GetterType.MULTILINE,
    # Lineage getters (graph visualization)
    "upstream": GetterType.MULTILINE,
    "downstream": GetterType.MULTILINE,
    "lineage": GetterType.MULTILINE,
}


def get_getter_type(field: str, value: Any = None) -> GetterType:
    """Determine getter type for a field.

    Args:
        field: Field name (e.g., "status", "params.lr", "stdout")
        value: Optional value to help determine type for dynamic fields

    Returns:
        GetterType for the field
    """
    # Check exact match first
    if field in GETTER_TYPES:
        return GETTER_TYPES[field]

    # Handle prefixed fields
    if field.startswith("params.") or field.startswith("metrics."):
        # If value is provided, use it to determine type
        if value is not None:
            if isinstance(value, dict):
                return GetterType.COMPLEX
            elif isinstance(value, list):
                return GetterType.LIST
        return GetterType.SCALAR

    if field.startswith("git.") or field.startswith("environment."):
        return GetterType.SCALAR

    # Default to scalar for unknown fields
    return GetterType.SCALAR


class GetterOutput:
    """Handles output formatting for all getter types.

    This class provides unified output handling for the yanex get command,
    supporting all getter types (SCALAR, LIST, COMPLEX, MULTILINE) and
    all output formats (DEFAULT, JSON, CSV, MARKDOWN, SWEEP).

    Usage:
        results = [(exp_id1, value1), (exp_id2, value2), ...]
        getter_type = get_getter_type(field, results[0][1])
        output = GetterOutput(field, output_format)
        output.output(results, getter_type)
    """

    def __init__(self, field: str, fmt: OutputFormat):
        """Initialize getter output handler.

        Args:
            field: Field name being retrieved (e.g., "status", "params.lr")
            fmt: Output format to use
        """
        self.field = field
        self.format = fmt

    def output(self, results: list[tuple[str, Any]], getter_type: GetterType) -> None:
        """Output results based on getter type and format.

        Args:
            results: List of (exp_id, value) tuples
            getter_type: Type of the getter
        """
        if getter_type == GetterType.MULTILINE:
            self._output_multiline(results)
        else:
            # SCALAR, LIST, and COMPLEX all use the same output logic
            # (difference is in value formatting)
            self._output_values(results, getter_type)

    def _output_values(
        self, results: list[tuple[str, Any]], getter_type: GetterType
    ) -> None:
        """Output scalar/list/complex values."""
        is_single = len(results) == 1

        if self.format == OutputFormat.JSON:
            self._output_json(results, is_single)
        elif self.format == OutputFormat.CSV:
            self._output_csv(results, getter_type)
        elif self.format == OutputFormat.MARKDOWN:
            self._output_markdown(results, getter_type)
        elif self.format == OutputFormat.SWEEP:
            self._output_sweep(results, getter_type)
        else:
            self._output_default(results)

    def _output_json(self, results: list[tuple[str, Any]], is_single: bool) -> None:
        """JSON output - consistent structure for single and multi."""
        if is_single:
            exp_id, value = results[0]
            click.echo(json.dumps({"id": exp_id, "value": value}))
        else:
            output = [{"id": exp_id, "value": value} for exp_id, value in results]
            click.echo(json.dumps(output))

    def _output_csv(
        self, results: list[tuple[str, Any]], getter_type: GetterType
    ) -> None:
        """CSV output with ID column."""
        click.echo(f"ID,{self.field}")
        for exp_id, value in results:
            csv_value = self._format_csv_value(value, getter_type)
            click.echo(f"{exp_id},{csv_value}")

    def _output_markdown(
        self, results: list[tuple[str, Any]], getter_type: GetterType
    ) -> None:
        """Markdown table output."""
        click.echo(f"| ID | {self.field} |")
        click.echo("| --- | --- |")
        for exp_id, value in results:
            display_value = self._format_display_value(value, getter_type)
            # Escape pipe characters in markdown
            display_value = display_value.replace("|", "\\|")
            click.echo(f"| {exp_id} | {display_value} |")

    def _output_sweep(
        self, results: list[tuple[str, Any]], getter_type: GetterType
    ) -> None:
        """Sweep output (comma-separated values only, no trailing newline)."""
        values = [self._format_csv_value(value, getter_type) for _, value in results]
        click.echo(",".join(values), nl=False)

    def _output_default(self, results: list[tuple[str, Any]]) -> None:
        """Default console output."""
        is_single = len(results) == 1

        for exp_id, value in results:
            display_value = self._format_display_value(value, GetterType.SCALAR)

            # Special case: id getter just outputs the value
            if self.field == "id":
                click.echo(display_value)
            elif is_single:
                # Single experiment: just output the value
                click.echo(display_value)
            else:
                # Multiple experiments: prefix with ID
                click.echo(f"{exp_id}: {display_value}")

    def _output_multiline(self, results: list[tuple[str, Any]]) -> None:
        """Output multi-line content with Rich Rule headers (for multiple experiments)."""
        # Validate format compatibility
        if self.format == OutputFormat.SWEEP:
            raise click.ClickException(
                f"--format sweep not supported for '{self.field}'"
            )
        if self.format == OutputFormat.CSV:
            raise click.ClickException(f"--format csv not supported for '{self.field}'")
        if self.format == OutputFormat.MARKDOWN:
            raise click.ClickException(
                f"--format markdown not supported for '{self.field}'"
            )

        if self.format == OutputFormat.JSON:
            # JSON output for multiline: use field name as key
            output = [{"id": exp_id, self.field: value} for exp_id, value in results]
            click.echo(json.dumps(output))
            return

        # Default: console output
        is_single = len(results) == 1

        if is_single:
            # Single experiment: just output the value (no header)
            _, value = results[0]
            if value:
                click.echo(value)
        else:
            # Multiple experiments: Rich console with styled headers
            from rich.console import Console
            from rich.rule import Rule

            console = Console()
            for i, (exp_id, value) in enumerate(results):
                if i > 0:
                    console.print()
                console.print(Rule(f"Experiment {exp_id}", style="dim", align="center"))
                if value:
                    console.print(value)

    @staticmethod
    def _format_csv_value(value: Any, getter_type: GetterType) -> str:
        """Format value for CSV output.

        Args:
            value: Value to format
            getter_type: Type of the getter

        Returns:
            CSV-safe string representation
        """
        if value is None:
            return ""
        if getter_type == GetterType.LIST:
            if isinstance(value, list):
                # Quote if contains comma
                joined = ",".join(str(v) for v in value)
                return f'"{joined}"' if "," in joined else joined
        if getter_type == GetterType.COMPLEX:
            return f'"{json.dumps(value)}"'
        # Escape quotes and wrap in quotes if value contains comma
        str_value = str(value)
        if "," in str_value or '"' in str_value:
            str_value = str_value.replace('"', '""')
            return f'"{str_value}"'
        return str_value

    @staticmethod
    def _format_display_value(value: Any, getter_type: GetterType) -> str:
        """Format value for human display.

        Args:
            value: Value to format
            getter_type: Type of the getter

        Returns:
            Human-readable string representation
        """
        if value is None:
            return ""
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            # Format as key=value pairs for dependencies-style dicts
            if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
                return " ".join(f"{k}={v}" for k, v in sorted(value.items()))
            return json.dumps(value)
        return str(value)
