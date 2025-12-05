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

    Uses the core resolve_experiment_id() utility for consistent ID resolution.

    Args:
        filter_obj: ExperimentFilter instance
        identifier: Experiment ID or name
        include_archived: Whether to search archived experiments

    Returns:
        - Single experiment dict if found by ID or unique name
        - List of experiments if multiple names match
        - None if not found
    """
    from ...utils.exceptions import AmbiguousIDError, ExperimentNotFoundError
    from ...utils.id_resolution import resolve_experiment_id

    try:
        # Use the core utility for ID resolution
        manager = filter_obj.manager
        experiment_id = resolve_experiment_id(
            identifier, manager, include_archived=include_archived
        )

        # Load and return the experiment metadata
        metadata = manager.storage.load_metadata(
            experiment_id, include_archived=include_archived
        )
        return metadata

    except AmbiguousIDError:
        # Multiple matches - load all matching experiments for user to choose
        all_experiments = filter_obj._load_all_experiments(include_archived)

        # Return all experiments that match the identifier (by ID prefix or name)
        matches = []
        for exp in all_experiments:
            exp_id = exp.get("id", "")
            exp_name = exp.get("name")

            # Check ID prefix match
            if isinstance(exp_id, str) and exp_id.startswith(identifier):
                matches.append(exp)
            # Check exact name match
            elif exp_name and exp_name == identifier:
                if exp not in matches:  # Avoid duplicates
                    matches.append(exp)

        return matches if matches else None

    except ExperimentNotFoundError:
        # No match found
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

    Uses the core resolve_experiment_id() utility for consistent ID resolution.

    Args:
        filter_obj: ExperimentFilter instance
        identifiers: List of experiment IDs or names
        archived: True for archived only, False for non-archived only, None for both

    Returns:
        List of found experiments

    Raises:
        click.ClickException: If any identifier is not found or ambiguous
    """
    from ...utils.exceptions import AmbiguousIDError, ExperimentNotFoundError
    from ...utils.id_resolution import resolve_experiment_id

    all_experiments = []
    manager = filter_obj.manager

    # Determine include_archived flag for resolve_experiment_id
    # archived=True: only archived, archived=False: only regular, archived=None: both
    include_archived = archived is not False

    for identifier in identifiers:
        try:
            # Use core utility to resolve ID
            experiment_id = resolve_experiment_id(
                identifier, manager, include_archived=include_archived
            )

            # Load metadata
            metadata = manager.storage.load_metadata(
                experiment_id, include_archived=include_archived
            )

            # If archived filter is strict (True or False, not None), verify match
            if archived is True:
                # Must be archived
                if not manager.storage.archived_experiment_exists(experiment_id):
                    raise click.ClickException(
                        f"Experiment '{identifier}' exists but is not archived"
                    )
            elif archived is False:
                # Must NOT be archived
                if manager.storage.archived_experiment_exists(experiment_id):
                    raise click.ClickException(
                        f"Experiment '{identifier}' is archived (use --archived flag)"
                    )

            all_experiments.append(metadata)

        except AmbiguousIDError as e:
            # Multiple matches - show table and raise clear error
            # For name matches, we want to show the table
            if len(e.matches) > 1:
                # Load full metadata for all matches to show in table
                match_experiments = []
                for match_id in e.matches:
                    # Extract just the ID if match contains extra info
                    exp_id = match_id.split()[0] if " " in match_id else match_id
                    try:
                        meta = manager.storage.load_metadata(
                            exp_id, include_archived=include_archived
                        )
                        match_experiments.append(meta)
                    except Exception:
                        continue

                if match_experiments:
                    click.echo(f"Multiple experiments found matching '{identifier}':")
                    click.echo()

                    from ..formatters.console import ExperimentTableFormatter

                    formatter = ExperimentTableFormatter()
                    formatter.print_experiments_table(match_experiments)
                    click.echo()
                    click.echo(
                        "Please use a more specific ID or full experiment ID for bulk operations."
                    )

            raise click.ClickException(
                f"Ambiguous identifier '{identifier}': {str(e).split(':', 1)[1].strip()}"
            )

        except ExperimentNotFoundError:
            # Determine location string for error message
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
    ids: list[str] = None,
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
        ids: Filter by specific experiment IDs
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
        ids=ids,
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
