"""
Core experiment filtering functionality.
"""

import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any

from ...core.constants import EXPERIMENT_STATUSES_SET
from ...core.manager import ExperimentManager
from ...utils.datetime_utils import parse_iso_timestamp


class ExperimentFilter:
    """
    Reusable experiment filtering system for CLI commands.

    Supports filtering by status, name patterns, tags, and time ranges.
    Can be used by list, delete, archive, and other commands.
    """

    def __init__(self, manager: ExperimentManager | None = None):
        """
        Initialize experiment filter.

        Args:
            manager: ExperimentManager instance (creates default if None)
        """
        self.manager = manager or ExperimentManager()

    def filter_experiments(
        self,
        status: str | None = None,
        name_pattern: str | None = None,
        tags: list[str] | None = None,
        started_after: datetime | None = None,
        started_before: datetime | None = None,
        ended_after: datetime | None = None,
        ended_before: datetime | None = None,
        limit: int | None = None,
        include_all: bool = False,
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Filter experiments based on multiple criteria.

        Args:
            status: Filter by experiment status (created/running/completed/failed/cancelled/staged)
            name_pattern: Filter by name using glob patterns (e.g., "*tuning*")
            tags: List of tags - experiments must have ALL specified tags
            started_after: Filter experiments started after this time
            started_before: Filter experiments started before this time
            ended_after: Filter experiments ended after this time
            ended_before: Filter experiments ended before this time
            limit: Maximum number of results to return (for pagination)
            include_all: If True, ignore default limit and return all matching experiments
            include_archived: If True, include archived experiments in results

        Returns:
            List of experiment metadata dictionaries matching all criteria

        Raises:
            ValueError: If invalid status or other parameters provided
        """
        # Validate status if provided
        if status is not None:
            if status not in EXPERIMENT_STATUSES_SET:
                raise ValueError(
                    f"Invalid status '{status}'. Valid options: {', '.join(sorted(EXPERIMENT_STATUSES_SET))}"
                )

        # Get all experiments
        all_experiments = self._load_all_experiments(include_archived)

        # Apply filters
        filtered = all_experiments

        if status is not None:
            filtered = [exp for exp in filtered if exp.get("status") == status]

        if name_pattern is not None:
            filtered = [
                exp for exp in filtered if self._matches_name_pattern(exp, name_pattern)
            ]

        if tags:
            filtered = [exp for exp in filtered if self._has_all_tags(exp, tags)]

        if started_after is not None:
            filtered = [
                exp for exp in filtered if self._started_after(exp, started_after)
            ]

        if started_before is not None:
            filtered = [
                exp for exp in filtered if self._started_before(exp, started_before)
            ]

        if ended_after is not None:
            filtered = [exp for exp in filtered if self._ended_after(exp, ended_after)]

        if ended_before is not None:
            filtered = [
                exp for exp in filtered if self._ended_before(exp, ended_before)
            ]

        # Sort by creation time (newest first)
        filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Apply limit
        if not include_all and limit is not None:
            filtered = filtered[:limit]
        elif not include_all and limit is None:
            # Default limit of 10 if not explicitly requesting all
            filtered = filtered[:10]

        return filtered

    def _load_all_experiments(
        self, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """
        Load metadata for all experiments.

        Args:
            include_archived: Whether to include archived experiments

        Returns:
            List of experiment metadata dictionaries
        """
        experiments = []
        experiments_dir = self.manager.storage.experiments_dir

        if not experiments_dir.exists():
            return experiments

        # Helper function to load experiments from a directory
        def load_from_directory(directory: Path, is_archived: bool = False):
            for exp_dir in directory.iterdir():
                if not exp_dir.is_dir():
                    continue

                # Skip the archived directory when loading regular experiments
                if not is_archived and exp_dir.name == "archived":
                    continue

                experiment_id = exp_dir.name

                # Validate experiment ID format (8 characters)
                if len(experiment_id) != 8:
                    continue

                try:
                    # Load metadata for this experiment
                    metadata = self.manager.storage.load_metadata(
                        experiment_id, include_archived=is_archived
                    )

                    # Add the experiment ID to metadata for convenience
                    metadata["id"] = experiment_id

                    # Add archived flag to distinguish archived experiments
                    metadata["archived"] = is_archived

                    experiments.append(metadata)

                except Exception:
                    # Skip experiments with corrupted or missing metadata
                    continue

        # Load regular experiments
        load_from_directory(experiments_dir, is_archived=False)

        # Load archived experiments if requested
        if include_archived:
            archived_dir = experiments_dir / "archived"
            if archived_dir.exists():
                load_from_directory(archived_dir, is_archived=True)

        return experiments

    def _matches_name_pattern(self, experiment: dict[str, Any], pattern: str) -> bool:
        """Check if experiment name matches glob pattern."""
        name = experiment.get("name", "")
        original_name = name

        # Special case: empty pattern should match empty names
        if not pattern:
            return not original_name

        if not name:
            # Handle unnamed experiments - convert to searchable form
            name = "[unnamed]"
        return fnmatch.fnmatch(name.lower(), pattern.lower())

    def _has_all_tags(
        self, experiment: dict[str, Any], required_tags: list[str]
    ) -> bool:
        """Check if experiment has all required tags."""
        exp_tags = set(experiment.get("tags", []))
        required_tags_set = set(required_tags)
        return required_tags_set.issubset(exp_tags)

    def _started_after(self, experiment: dict[str, Any], after_time: datetime) -> bool:
        """Check if experiment started after the specified time."""
        started_at = experiment.get("started_at")
        if not started_at:
            return False

        exp_start = parse_iso_timestamp(started_at)
        if exp_start is None:
            return False

        return exp_start >= after_time

    def _started_before(
        self, experiment: dict[str, Any], before_time: datetime
    ) -> bool:
        """Check if experiment started before the specified time."""
        started_at = experiment.get("started_at")
        if not started_at:
            return False

        exp_start = parse_iso_timestamp(started_at)
        if exp_start is None:
            return False

        return exp_start < before_time

    def _ended_after(self, experiment: dict[str, Any], after_time: datetime) -> bool:
        """Check if experiment ended after the specified time."""
        ended_at = experiment.get("ended_at")
        if not ended_at:
            return False

        exp_end = parse_iso_timestamp(ended_at)
        if exp_end is None:
            return False

        return exp_end >= after_time

    def _ended_before(self, experiment: dict[str, Any], before_time: datetime) -> bool:
        """Check if experiment ended before the specified time."""
        ended_at = experiment.get("ended_at")
        if not ended_at:
            return False

        exp_end = parse_iso_timestamp(ended_at)
        if exp_end is None:
            return False

        return exp_end < before_time
