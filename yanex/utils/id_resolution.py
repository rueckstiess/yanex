"""Utilities for resolving experiment IDs (including short IDs)."""

from typing import TYPE_CHECKING

from .exceptions import AmbiguousIDError, ExperimentNotFoundError

if TYPE_CHECKING:
    from ..core.manager import ExperimentManager


def resolve_experiment_id(
    identifier: str,
    manager: "ExperimentManager",
    include_archived: bool = True,
) -> str:
    """Resolve experiment identifier to full 8-character ID.

    Supports:
    - Full 8-character experiment IDs
    - Short ID prefixes (minimum 4 characters recommended)
    - Exact experiment names

    Args:
        identifier: Experiment ID (full or partial) or name.
        manager: ExperimentManager instance.
        include_archived: Whether to search archived experiments.

    Returns:
        Full 8-character experiment ID.

    Raises:
        ExperimentNotFoundError: If no matching experiment found.
        AmbiguousIDError: If identifier matches multiple experiments.

    Examples:
        >>> resolve_experiment_id("abc12345", manager)
        "abc12345"  # Full ID returns as-is

        >>> resolve_experiment_id("abc1", manager)
        "abc12345"  # Short ID resolves to full ID

        >>> resolve_experiment_id("my-experiment", manager)
        "abc12345"  # Name resolves to ID
    """
    # Get all experiment IDs
    all_experiment_ids = manager.storage.list_experiments(
        include_archived=include_archived
    )

    # Try ID prefix match
    id_matches = [
        exp_id for exp_id in all_experiment_ids if exp_id.startswith(identifier)
    ]

    if len(id_matches) == 1:
        return id_matches[0]
    elif len(id_matches) > 1:
        # Multiple ID matches - load metadata for helpful error message
        match_details = []
        for match_id in id_matches[:5]:  # Show max 5
            try:
                metadata = manager.storage.load_metadata(
                    match_id, include_archived=include_archived
                )
                name = metadata.get("name") or "unnamed"
                status = metadata.get("status", "unknown")
                match_details.append(f"{match_id} ({name}, {status})")
            except Exception:
                match_details.append(match_id)

        if len(id_matches) > 5:
            match_details.append(f"... and {len(id_matches) - 5} more")

        raise AmbiguousIDError(identifier, match_details)

    # Try exact name match
    name_matches = []
    for exp_id in all_experiment_ids:
        try:
            metadata = manager.storage.load_metadata(
                exp_id, include_archived=include_archived
            )
            exp_name = metadata.get("name")
            if exp_name and exp_name == identifier:
                name_matches.append(exp_id)
        except Exception:
            continue

    if len(name_matches) == 1:
        return name_matches[0]
    elif len(name_matches) > 1:
        raise AmbiguousIDError(identifier, name_matches)

    # No matches found
    raise ExperimentNotFoundError(identifier)
