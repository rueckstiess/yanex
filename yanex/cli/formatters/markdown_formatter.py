"""GitHub-flavored markdown formatting utilities for yanex CLI.

This module provides utilities for generating markdown tables and
formatted output suitable for documentation and GitHub.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


def _format_cell(value: Any) -> str:
    """Format a cell value for markdown output.

    Args:
        value: Any value to format

    Returns:
        String representation suitable for markdown cells
    """
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        if not value:
            return "-"
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        if not value:
            return "-"
        pairs = [f"{k}={v}" for k, v in value.items()]
        return ", ".join(pairs)

    # Escape pipe characters in markdown
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def format_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Generate a GitHub-flavored markdown table.

    Args:
        rows: List of row dictionaries
        columns: Column keys in desired order (defaults to keys from first row)
        headers: Optional mapping of column keys to display headers

    Returns:
        Markdown table string

    Example:
        | ID | Name | Status |
        |----|------|--------|
        | abc123 | exp-1 | completed |
    """
    if not rows:
        return "_No data_"

    # Determine columns from first row if not specified
    if columns is None:
        columns = list(rows[0].keys())

    # Build header row
    if headers:
        header_cells = [headers.get(col, col) for col in columns]
    else:
        header_cells = [col.replace("_", " ").title() for col in columns]

    lines = []

    # Header row
    lines.append("| " + " | ".join(header_cells) + " |")

    # Separator row (use minimum 3 dashes)
    separators = ["-" * max(3, len(h)) for h in header_cells]
    lines.append("| " + " | ".join(separators) + " |")

    # Data rows
    for row in rows:
        cells = [_format_cell(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def output_markdown_table(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    headers: dict[str, str] | None = None,
    title: str | None = None,
    file: TextIO = sys.stdout,
) -> None:
    """Output a markdown table.

    Args:
        rows: List of row dictionaries
        columns: Column keys in desired order
        headers: Optional column key to header mapping
        title: Optional title to display above table
        file: Output file (defaults to stdout)
    """
    if title:
        print(f"## {title}\n", file=file)

    table = format_markdown_table(rows, columns, headers)
    print(table, file=file)


def format_action_result_markdown(
    operation: str,
    success_count: int,
    failed_count: int,
    experiments: list[str],
    failed: list[dict[str, str]] | None = None,
) -> str:
    """Format action command result as markdown.

    Returns markdown summary suitable for GitHub issues/PRs.

    Args:
        operation: Operation name (archive, delete, etc.)
        success_count: Number of successful operations
        failed_count: Number of failed operations
        experiments: List of successfully processed experiment IDs
        failed: List of {"id": ..., "error": ...} dicts for failures

    Returns:
        Markdown formatted string
    """
    lines = []

    # Title with status emoji
    status_emoji = "✓" if failed_count == 0 else "⚠️"
    lines.append(f"### {status_emoji} {operation.title()} Results\n")

    # Summary counts
    lines.append(f"- **Successful**: {success_count}")
    lines.append(f"- **Failed**: {failed_count}")
    lines.append("")

    # List successful experiments (limit to first 10)
    if experiments:
        lines.append("**Processed experiments:**")
        for exp_id in experiments[:10]:
            lines.append(f"- `{exp_id}`")
        if len(experiments) > 10:
            lines.append(f"- ... and {len(experiments) - 10} more")
        lines.append("")

    # List failed experiments
    if failed:
        lines.append("**Failed experiments:**")
        for item in failed:
            exp_id = item.get("id", "unknown")
            error = item.get("error", "unknown error")
            lines.append(f"- `{exp_id}`: {error}")

    return "\n".join(lines)


def output_action_result_markdown(
    operation: str,
    success_count: int,
    failed_count: int,
    experiments: list[str],
    failed: list[dict[str, str]] | None = None,
    file: TextIO = sys.stdout,
) -> None:
    """Output action command result as markdown.

    Args:
        operation: Operation name (archive, delete, etc.)
        success_count: Number of successful operations
        failed_count: Number of failed operations
        experiments: List of successfully processed experiment IDs
        failed: List of {"id": ..., "error": ...} dicts for failures
        file: Output file (defaults to stdout)
    """
    markdown = format_action_result_markdown(
        operation, success_count, failed_count, experiments, failed
    )
    print(markdown, file=file)
