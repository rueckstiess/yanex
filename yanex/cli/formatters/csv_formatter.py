"""CSV formatting utilities for yanex CLI.

This module provides utilities for outputting data as CSV,
with proper handling of special characters and quoting.
"""

import csv
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, TextIO


def _serialize_csv_value(value: Any) -> str:
    """Serialize a value for CSV output.

    Args:
        value: Any value to serialize

    Returns:
        String representation suitable for CSV
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        # Join list items with comma (will be quoted by CSV writer if needed)
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        # For dicts, use JSON-like format
        pairs = [f"{k}={v}" for k, v in value.items()]
        return ";".join(pairs)
    return str(value)


def output_csv(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    headers: dict[str, str] | None = None,
    file: TextIO = sys.stdout,
) -> None:
    """Output data as CSV.

    Outputs directly to stdout (or specified file) for clean piping.

    Args:
        rows: List of row dictionaries
        columns: Column keys in desired order (defaults to keys from first row)
        headers: Optional mapping of column keys to display headers
        file: Output file (defaults to stdout)
    """
    if not rows:
        return

    # Determine columns from first row if not specified
    if columns is None:
        columns = list(rows[0].keys())

    # Determine header names
    if headers:
        header_row = [headers.get(col, col) for col in columns]
    else:
        header_row = columns

    # Create CSV writer
    writer = csv.writer(file)

    # Write header
    writer.writerow(header_row)

    # Write data rows
    for row in rows:
        csv_row = [_serialize_csv_value(row.get(col)) for col in columns]
        writer.writerow(csv_row)


def format_csv(
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
    headers: dict[str, str] | None = None,
) -> str:
    """Format data as CSV string.

    Args:
        rows: List of row dictionaries
        columns: Column keys in desired order
        headers: Optional column key to header mapping

    Returns:
        CSV string
    """
    output = StringIO()
    output_csv(rows, columns, headers, file=output)
    return output.getvalue()


def format_action_result_csv(
    operation: str,
    experiments: list[str],
    failed: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Format action command result as CSV rows.

    Args:
        operation: Operation name (archive, delete, etc.)
        experiments: List of successfully processed experiment IDs
        failed: List of {"id": ..., "error": ...} dicts for failures

    Returns:
        List of dicts suitable for output_csv, with columns: id, status, error
    """
    rows: list[dict[str, Any]] = []

    # Add successful experiments
    for exp_id in experiments:
        rows.append({"id": exp_id, "status": "success", "error": ""})

    # Add failed experiments
    if failed:
        for item in failed:
            rows.append(
                {
                    "id": item.get("id", ""),
                    "status": "failed",
                    "error": item.get("error", ""),
                }
            )

    return rows


def output_action_result_csv(
    operation: str,
    experiments: list[str],
    failed: list[dict[str, str]] | None = None,
    file: TextIO = sys.stdout,
) -> None:
    """Output action command result as CSV.

    Args:
        operation: Operation name (archive, delete, etc.)
        experiments: List of successfully processed experiment IDs
        failed: List of {"id": ..., "error": ...} dicts for failures
        file: Output file (defaults to stdout)
    """
    rows = format_action_result_csv(operation, experiments, failed)
    output_csv(rows, columns=["id", "status", "error"], file=file)
