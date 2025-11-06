"""
Confirmation utilities for bulk operations on experiments.
"""

from typing import Any

import click

from ..filters import ExperimentFilter
from ..formatters.console import ExperimentTableFormatter


def confirm_experiment_operation(
    experiments: list[dict[str, Any]],
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


def find_experiment(
    filter_obj: ExperimentFilter, identifier: str, include_archived: bool = False
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """
    Find experiment by ID, ID prefix, or name.

    Args:
        filter_obj: ExperimentFilter instance
        identifier: Experiment ID or name
        include_archived: Whether to search archived experiments

    Returns:
        - Single experiment dict if found by ID or unique name
        - List of experiments if multiple names match
        - None if not found
    """
    # First, try to find by exact ID match
    try:
        all_experiments = filter_obj._load_all_experiments(include_archived)

        # Try ID prefix match
        id_prefix_matches: list[dict[str, Any]] = []
        if identifier:
            for exp in all_experiments:
                exp_id = exp.get("id", "")
                if isinstance(exp_id, str) and exp_id.startswith(identifier):
                    id_prefix_matches.append(exp)

        if len(id_prefix_matches) == 1:
            return id_prefix_matches[0]
        elif len(id_prefix_matches) > 1:
            return id_prefix_matches

        # Try name match
        name_matches = []
        for exp in all_experiments:
            exp_name = exp.get("name")
            if exp_name and exp_name == identifier:
                name_matches.append(exp)

        # Return based on name matches
        if len(name_matches) == 1:
            return name_matches[0]
        elif len(name_matches) > 1:
            return name_matches

        # No matches found
        return None

    except Exception:
        return None


def find_experiments_by_identifiers(
    filter_obj: ExperimentFilter,
    identifiers: list[str],
    archived: bool | None = False,
) -> list[dict[str, Any]]:
    """
    Find experiments by list of identifiers (IDs or names).

    Args:
        filter_obj: ExperimentFilter instance
        identifiers: List of experiment IDs or names
        archived: True for archived only, False for non-archived only, None for both

    Returns:
        List of found experiments

    Raises:
        click.ClickException: If any identifier is not found or ambiguous
    """
    all_experiments = []

    for identifier in identifiers:
        # Try to find by ID first (8-character hex)
        if len(identifier) == 8:
            try:
                # Try exact ID match using the unified filter
                results = filter_obj.filter_experiments(
                    ids=[identifier], archived=archived, include_all=True
                )
                if results:
                    all_experiments.append(results[0])
                    continue
            except Exception:
                pass

        # Try ID prefix match (if shorter than 8 characters)
        if len(identifier) < 8:
            try:
                # Get all experiments and filter by ID prefix
                all_exps = filter_obj.filter_experiments(
                    archived=archived, include_all=True
                )
                prefix_matches = [
                    exp for exp in all_exps if exp.get("id", "").startswith(identifier)
                ]

                if len(prefix_matches) == 1:
                    all_experiments.append(prefix_matches[0])
                    continue
                elif len(prefix_matches) > 1:
                    raise click.ClickException(
                        f"Ambiguous experiment ID prefix '{identifier}' matches multiple experiments"
                    )
            except Exception:
                pass

        # Try name match
        try:
            results = filter_obj.filter_experiments(
                name=identifier, archived=archived, include_all=True
            )

            if len(results) == 1:
                all_experiments.append(results[0])
                continue
            elif len(results) > 1:
                # Multiple experiments with same name
                click.echo(f"Multiple experiments found with name '{identifier}':")
                click.echo()

                from ..formatters.console import ExperimentTableFormatter

                formatter = ExperimentTableFormatter()
                formatter.print_experiments_table(results)
                click.echo()
                click.echo(
                    "Please use specific experiment IDs instead of names for bulk operations."
                )
                raise click.ClickException(f"Ambiguous experiment name: '{identifier}'")
        except Exception:
            pass

        # If we get here, nothing was found
        if archived is True:
            location = "archived"
        elif archived is False:
            location = "regular"
        else:
            location = ""

        raise click.ClickException(
            f"No {location} experiment found with ID or name '{identifier}'"
        )

    return all_experiments


def find_experiments_by_filters(
    filter_obj: ExperimentFilter,
    status: str = None,
    name: str = None,
    tags: list[str] = None,
    script_pattern: str = None,
    started_after=None,
    started_before=None,
    ended_after=None,
    ended_before=None,
    archived: bool = False,
) -> list[dict[str, Any]]:
    """
    Find experiments using filter criteria.

    Args:
        filter_obj: ExperimentFilter instance
        status: Filter by status
        name: Filter by name pattern
        tags: Filter by tags
        script_pattern: Filter by script name pattern
        started_after: Filter by start time
        started_before: Filter by start time
        ended_after: Filter by end time
        ended_before: Filter by end time
        archived: Whether to search archived experiments

    Returns:
        List of found experiments
    """
    return filter_obj.filter_experiments(
        status=status,
        name=name,
        tags=tags,
        script_pattern=script_pattern,
        started_after=started_after,
        started_before=started_before,
        ended_after=ended_after,
        ended_before=ended_before,
        include_all=True,  # Get all matching experiments for bulk operations
        archived=archived,
    )
