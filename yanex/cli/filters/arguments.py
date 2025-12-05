"""
Standardized CLI argument decorators for unified filtering.

This module provides consistent Click options for all multi-experiment commands
using the new unified filtering system.
"""

from collections.abc import Callable
from typing import Any

import click

from ...core.constants import EXPERIMENT_STATUSES


def experiment_filter_options(
    include_ids: bool = True,
    include_archived: bool = True,
    include_limit: bool = True,
    default_limit: int | None = None,
) -> Callable:
    """
    Decorator factory for adding standard filter options to CLI commands.

    Args:
        include_ids: Whether to include --ids option
        include_archived: Whether to include --archived option
        include_limit: Whether to include --limit option
        default_limit: Default limit value (None means no default)

    Returns:
        Decorator function that adds the specified options
    """

    def decorator(func: Callable) -> Callable:
        # Apply options in reverse order (Click applies them bottom-up)

        # Time filtering options
        func = click.option(
            "--ended-before",
            help="Filter experiments ended before date/time (e.g., '2025-01-01', 'yesterday', '1 week ago')",
        )(func)

        func = click.option(
            "--ended-after", help="Filter experiments ended after date/time"
        )(func)

        func = click.option(
            "--started-before", help="Filter experiments started before date/time"
        )(func)

        func = click.option(
            "--started-after",
            help="Filter experiments started after date/time (e.g., '2025-01-01', 'yesterday', '1 week ago')",
        )(func)

        # Core filtering options
        func = click.option(
            "--tag",
            "-t",
            "tags",
            multiple=True,
            help="Filter experiments that have ALL specified tags (repeatable)",
        )(func)

        func = click.option(
            "--name",
            "-n",
            "name_pattern",
            help="Filter by name using glob patterns (e.g., '*training*'). Use empty string '' to match unnamed experiments.",
        )(func)

        func = click.option(
            "--status",
            "-s",
            type=click.Choice(EXPERIMENT_STATUSES, case_sensitive=False),
            help="Filter by experiment status",
        )(func)

        func = click.option(
            "--script",
            "-c",
            "script_pattern",
            help="Filter by script name using glob patterns (case insensitive, e.g., 'train.py', '*prep*'). Extensions are optional.",
        )(func)

        # Conditional options
        if include_ids:
            func = click.option(
                "--ids",
                "-i",
                help="Filter by experiment ID(s). Accepts a single ID or comma-separated list (e.g., 'abc123' or 'a1,b2,c3')",
            )(func)

        if include_archived:
            func = click.option(
                "--archived",
                "-a",
                is_flag=True,
                help="Include archived experiments (default: only regular experiments)",
            )(func)

        if include_limit:
            help_text = "Maximum number of experiments to process"
            if default_limit:
                help_text += f" (default: {default_limit})"

            func = click.option(
                "--limit", "-l", type=int, default=default_limit, help=help_text
            )(func)

        return func

    return decorator


def validate_filter_arguments(
    ids: str | None = None,
    status: str | None = None,
    name_pattern: str | None = None,
    tags: tuple | None = None,
    started_after: str | None = None,
    started_before: str | None = None,
    ended_after: str | None = None,
    ended_before: str | None = None,
    archived: bool | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Validate and normalize filter arguments from CLI.

    Args:
        ids: Comma-separated experiment IDs string (e.g., 'a1,b2,c3')
        status: Status string from Click
        name_pattern: Name pattern string
        tags: Tuple of tags from Click
        started_after: Start time filter string
        started_before: Start time filter string
        ended_after: End time filter string
        ended_before: End time filter string
        archived: Archived flag
        **kwargs: Additional arguments to pass through

    Returns:
        Dictionary of normalized filter arguments for ExperimentFilter

    Raises:
        click.BadParameter: If validation fails
    """
    normalized = {}

    # Parse comma-separated IDs into a list
    if ids:
        id_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]
        if id_list:
            normalized["ids"] = id_list

    if status:
        normalized["status"] = status

    if name_pattern is not None:
        normalized["name_pattern"] = name_pattern

    if tags and len(tags) > 0:
        normalized["tags"] = list(tags)

    # Time filters
    for time_field in [
        "started_after",
        "started_before",
        "ended_after",
        "ended_before",
    ]:
        value = locals().get(time_field)
        if value:
            normalized[time_field] = value

    # Boolean filters
    if archived is not None:
        normalized["archived"] = archived

    # Pass through any additional keyword arguments
    for key, value in kwargs.items():
        if value is not None:
            # Validate script_pattern doesn't contain directory separators
            if key == "script_pattern" and isinstance(value, str):
                if "/" in value or "\\" in value:
                    click.echo(
                        "Warning: Script patterns should not include path separators. "
                        "Use filename patterns only (e.g., 'train.py', '*prep*').",
                        err=True,
                    )
            normalized[key] = value

    return normalized


def require_filters_or_confirmation(
    filter_args: dict[str, Any], operation_name: str, force: bool = False
) -> None:
    """
    Require either specific filters or user confirmation for bulk operations.

    This prevents accidental bulk operations on all experiments.

    Args:
        filter_args: Normalized filter arguments
        operation_name: Name of operation (for error messages)
        force: Whether to skip confirmation

    Raises:
        click.ClickException: If no filters provided and no confirmation
    """
    # Check if any meaningful filters are provided
    meaningful_filters = {
        k: v
        for k, v in filter_args.items()
        if k not in ["limit", "sort_by", "sort_desc", "include_all"] and v is not None
    }

    if not meaningful_filters:
        if force:
            return  # Force flag allows operation on all experiments

        # No filters provided - require confirmation
        if not click.confirm(
            f"Are you sure you want to {operation_name} all experiments?"
        ):
            raise click.ClickException(
                f"{operation_name.capitalize()} operation cancelled."
            )


def format_filter_summary(filter_args: dict[str, Any]) -> str:
    """
    Create a human-readable summary of applied filters.

    Args:
        filter_args: Filter arguments dictionary

    Returns:
        Formatted string describing the filters
    """
    if not filter_args:
        return "No filters applied"

    parts = []

    if "ids" in filter_args:
        ids_str = ", ".join(filter_args["ids"][:3])
        if len(filter_args["ids"]) > 3:
            ids_str += f" and {len(filter_args['ids']) - 3} more"
        parts.append(f"IDs: {ids_str}")

    if "status" in filter_args:
        status_val = filter_args["status"]
        if isinstance(status_val, list):
            parts.append(f"Status: {', '.join(status_val)}")
        else:
            parts.append(f"Status: {status_val}")

    if "name_pattern" in filter_args:
        parts.append(f"Name pattern: '{filter_args['name_pattern']}'")

    if "script_pattern" in filter_args:
        parts.append(f"Script pattern: '{filter_args['script_pattern']}'")

    if "tags" in filter_args:
        parts.append(f"Tags: {', '.join(filter_args['tags'])}")

    if "started_after" in filter_args:
        parts.append(f"Started after: {filter_args['started_after']}")

    if "started_before" in filter_args:
        parts.append(f"Started before: {filter_args['started_before']}")

    if "ended_after" in filter_args:
        parts.append(f"Ended after: {filter_args['ended_after']}")

    if "ended_before" in filter_args:
        parts.append(f"Ended before: {filter_args['ended_before']}")

    if "archived" in filter_args:
        archived_text = (
            "archived only" if filter_args["archived"] else "non-archived only"
        )
        parts.append(f"Archive status: {archived_text}")

    return "Filters: " + "; ".join(parts) if parts else "No filters applied"


def parse_cli_time_filters(
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
) -> tuple[Any, Any, Any, Any]:
    """
    Parse CLI time filter strings into datetime objects.

    Args:
        started_after: Start time filter string
        started_before: Start time filter string
        ended_after: End time filter string
        ended_before: End time filter string

    Returns:
        Tuple of parsed datetime objects (or None)

    Raises:
        click.BadParameter: If parsing fails
    """
    from ..error_handling import CLIErrorHandler

    try:
        return CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    except ValueError as e:
        raise click.BadParameter(str(e))
