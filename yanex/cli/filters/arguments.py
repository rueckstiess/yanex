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
            "tags",
            multiple=True,
            help="Filter experiments that have ALL specified tags (repeatable)",
        )(func)

        func = click.option(
            "--name",
            "name_pattern",
            help="Filter by name using glob patterns (e.g., '*training*'). Use empty string '' to match unnamed experiments.",
        )(func)

        func = click.option(
            "--status",
            multiple=True,
            type=click.Choice(EXPERIMENT_STATUSES, case_sensitive=False),
            help="Filter by experiment status (repeatable for OR logic)",
        )(func)

        # Conditional options
        if include_ids:
            func = click.option(
                "--ids",
                multiple=True,
                help="Filter by specific experiment IDs (repeatable for OR logic)",
            )(func)

        if include_archived:
            func = click.option(
                "--archived",
                is_flag=True,
                help="Include archived experiments (default: only regular experiments)",
            )(func)

        if include_limit:
            help_text = "Maximum number of experiments to process"
            if default_limit:
                help_text += f" (default: {default_limit})"

            func = click.option(
                "--limit", type=int, default=default_limit, help=help_text
            )(func)

        return func

    return decorator


def validate_filter_arguments(
    ids: tuple | None = None,
    status: tuple | None = None,
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
        ids: Tuple of experiment IDs from Click
        status: Tuple of statuses from Click
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

    # Convert Click tuples to lists and handle empty cases
    if ids and len(ids) > 0:
        normalized["ids"] = list(ids)

    if status and len(status) > 0:
        normalized["status"] = list(status)

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
        parts.append(f"Status: {', '.join(filter_args['status'])}")

    if "name_pattern" in filter_args:
        parts.append(f"Name pattern: '{filter_args['name_pattern']}'")

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
