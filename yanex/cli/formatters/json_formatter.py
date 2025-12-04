"""JSON formatting utilities for yanex CLI.

This module provides utilities for outputting data as JSON,
with proper handling of datetime and Path serialization.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


class YanexJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling datetime and Path objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def output_json(data: Any, file: TextIO = sys.stdout, compact: bool = False) -> None:
    """Output data as formatted JSON.

    Outputs directly to stdout (or specified file) for clean piping.

    Args:
        data: Data to serialize (dict, list, or primitive)
        file: Output file (defaults to stdout)
        compact: If True, output without indentation
    """
    indent = None if compact else 2
    json_str = json.dumps(data, cls=YanexJSONEncoder, indent=indent)
    print(json_str, file=file)


def format_json(data: Any, compact: bool = False) -> str:
    """Format data as JSON string.

    Args:
        data: Data to serialize
        compact: If True, output without indentation

    Returns:
        JSON string
    """
    indent = None if compact else 2
    return json.dumps(data, cls=YanexJSONEncoder, indent=indent)


def format_action_result(
    operation: str,
    success: bool,
    experiments: list[str],
    failed: list[dict[str, str]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Format action command result for JSON output.

    Args:
        operation: Operation name (archive, delete, etc.)
        success: Overall success status
        experiments: List of successfully processed experiment IDs
        failed: List of {"id": ..., "error": ...} dicts for failures
        message: Optional message

    Returns:
        Structured result dict

    Example output:
        {
            "success": true,
            "operation": "archive",
            "experiments": ["abc123", "def456"],
            "failed": [],
            "count": 2
        }
    """
    result: dict[str, Any] = {
        "success": success,
        "operation": operation,
        "experiments": experiments,
        "failed": failed or [],
        "count": len(experiments),
    }
    if message:
        result["message"] = message
    return result
