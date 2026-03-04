"""Filter utilities for remote experiment metadata.

Reuses ExperimentFilter's filtering logic to apply the same filters
to remote metadata dicts as we do to local experiments.
"""

from datetime import datetime
from typing import Any

from ..core.filtering import ExperimentFilter


def filter_experiment_dicts(
    experiments: list[dict[str, Any]],
    ids: list[str] | None = None,
    status: str | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    script_pattern: str | None = None,
    started_after: datetime | None = None,
    started_before: datetime | None = None,
    ended_after: datetime | None = None,
    ended_before: datetime | None = None,
    project: str | None = None,
) -> list[dict[str, Any]]:
    """Apply standard yanex filters to a list of experiment metadata dicts.

    This reuses ExperimentFilter's filtering logic without requiring
    local experiment storage.

    Args:
        experiments: List of experiment metadata dicts
        ids: Filter by experiment IDs (OR logic)
        status: Filter by status
        name: Glob pattern for name matching
        tags: Required tags (AND logic)
        script_pattern: Glob pattern for script name
        started_after: Filter by start time (inclusive)
        started_before: Filter by start time (exclusive)
        ended_after: Filter by end time (inclusive)
        ended_before: Filter by end time (exclusive)
        project: Filter by project name

    Returns:
        Filtered list of experiment metadata dicts
    """
    ef = ExperimentFilter()
    normalized = ef._normalize_filter_inputs(
        ids=ids,
        status=[status] if isinstance(status, str) else status,
        name_pattern=name,
        tags=tags,
        script_pattern=script_pattern,
        started_after=started_after,
        started_before=started_before,
        ended_after=ended_after,
        ended_before=ended_before,
        archived=None,
        project=project,
    )
    return ef._apply_all_filters(experiments, normalized)
