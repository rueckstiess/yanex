"""
Confirmation utilities for bulk operations on experiments.
"""

from typing import Any, Dict, List

import click

from ..filters import ExperimentFilter
from ..formatters.console import ExperimentTableFormatter


def confirm_experiment_operation(
    experiments: List[Dict[str, Any]],
    operation: str,
    force: bool = False,
    operation_verb: str = None,
    default_yes: bool = False,
) -> bool:
    """
    Show experiments and confirm operation.

    Args:
        experiments: List of experiment dictionaries to show
        operation: Operation name (e.g., "archive", "delete")
        force: Skip confirmation if True
        operation_verb: Past tense verb for confirmation (e.g., "archived", "deleted")
        default_yes: If True, default to Yes instead of No

    Returns:
        True if user confirms, False otherwise
    """
    if not experiments:
        click.echo(f"No experiments found to {operation}.")
        return False

    if operation_verb is None:
        operation_verb = f"{operation}d"  # Default: "archive" -> "archived"

    # Display table of experiments
    formatter = ExperimentTableFormatter()

    click.echo(
        f"The following {len(experiments)} experiment(s) will be {operation_verb}:"
    )
    click.echo()

    formatter.print_experiments_table(experiments)
    click.echo()

    # Skip confirmation if force flag is set
    if force:
        return True

    # Get confirmation from user
    if len(experiments) == 1:
        message = f"{operation.capitalize()} this experiment?"
    else:
        message = f"{operation.capitalize()} these {len(experiments)} experiments?"

    return click.confirm(message, default=default_yes)


def find_experiments_by_identifiers(
    filter_obj: ExperimentFilter,
    identifiers: List[str],
    include_archived: bool = False,
    archived_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find experiments by list of identifiers (IDs or names).

    Args:
        filter_obj: ExperimentFilter instance
        identifiers: List of experiment IDs or names
        include_archived: Whether to search archived experiments
        archived_only: If True, search ONLY archived experiments (for unarchive command)

    Returns:
        List of found experiments

    Raises:
        click.ClickException: If any identifier is not found or ambiguous
    """
    from .show import find_experiment  # Import here to avoid circular import

    all_experiments = []

    for identifier in identifiers:
        if archived_only:
            # For unarchive: search all experiments but only consider archived ones
            result = find_experiment(filter_obj, identifier, include_archived=True)

            if result is None:
                raise click.ClickException(
                    f"No archived experiment found with ID or name '{identifier}'"
                )

            if isinstance(result, list):
                # Filter to only archived experiments
                archived_results = [exp for exp in result if exp.get("archived", False)]

                if not archived_results:
                    raise click.ClickException(
                        f"No archived experiment found with name '{identifier}'"
                    )
                elif len(archived_results) == 1:
                    result = archived_results[0]
                else:
                    # Multiple archived experiments with same name
                    click.echo(
                        f"Multiple archived experiments found with name '{identifier}':"
                    )
                    click.echo()

                    formatter = ExperimentTableFormatter()
                    formatter.print_experiments_table(archived_results)
                    click.echo()
                    click.echo(
                        "Please use specific experiment IDs instead of names for bulk operations."
                    )
                    raise click.ClickException(
                        f"Ambiguous archived experiment name: '{identifier}'"
                    )
            else:
                # Single result - check if it's archived
                if not result.get("archived", False):
                    raise click.ClickException(
                        f"Experiment '{identifier}' is not archived"
                    )
        else:
            # Normal mode: search as specified
            result = find_experiment(filter_obj, identifier, include_archived)

            if result is None:
                location = "archived" if include_archived else "regular"
                raise click.ClickException(
                    f"No {location} experiment found with ID or name '{identifier}'"
                )

            if isinstance(result, list):
                # Multiple experiments with same name
                click.echo(f"Multiple experiments found with name '{identifier}':")
                click.echo()

                formatter = ExperimentTableFormatter()
                formatter.print_experiments_table(result)
                click.echo()
                click.echo(
                    "Please use specific experiment IDs instead of names for bulk operations."
                )
                raise click.ClickException(f"Ambiguous experiment name: '{identifier}'")

        # Single experiment found
        all_experiments.append(result)

    return all_experiments


def find_experiments_by_filters(
    filter_obj: ExperimentFilter,
    status: str = None,
    name_pattern: str = None,
    tags: List[str] = None,
    started_after=None,
    started_before=None,
    ended_after=None,
    ended_before=None,
    include_archived: bool = False,
) -> List[Dict[str, Any]]:
    """
    Find experiments using filter criteria.

    Args:
        filter_obj: ExperimentFilter instance
        status: Filter by status
        name_pattern: Filter by name pattern
        tags: Filter by tags
        started_after: Filter by start time
        started_before: Filter by start time
        ended_after: Filter by end time
        ended_before: Filter by end time
        include_archived: Whether to search archived experiments

    Returns:
        List of found experiments
    """
    return filter_obj.filter_experiments(
        status=status,
        name_pattern=name_pattern,
        tags=tags,
        started_after=started_after,
        started_before=started_before,
        ended_after=ended_after,
        ended_before=ended_before,
        include_all=True,  # Get all matching experiments for bulk operations
        include_archived=include_archived,
    )
