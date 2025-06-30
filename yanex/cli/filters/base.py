"""
Core experiment filtering functionality.
"""

import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.constants import EXPERIMENT_STATUSES_SET
from ...core.manager import ExperimentManager


class ExperimentFilter:
    """
    Reusable experiment filtering system for CLI commands.

    Supports filtering by status, name patterns, tags, and time ranges.
    Can be used by list, delete, archive, and other commands.
    """

    def __init__(self, manager: Optional[ExperimentManager] = None):
        """
        Initialize experiment filter.

        Args:
            manager: ExperimentManager instance (creates default if None)
        """
        self.manager = manager or ExperimentManager()

    def filter_experiments(
        self,
        status: Optional[str] = None,
        name_pattern: Optional[str] = None,
        tags: Optional[List[str]] = None,
        started_after: Optional[datetime] = None,
        started_before: Optional[datetime] = None,
        ended_after: Optional[datetime] = None,
        ended_before: Optional[datetime] = None,
        limit: Optional[int] = None,
        include_all: bool = False,
        include_archived: bool = False,
    ) -> List[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
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

    def _matches_name_pattern(self, experiment: Dict[str, Any], pattern: str) -> bool:
        """Check if experiment name matches glob pattern."""
        name = experiment.get("name", "")
        if not name:
            # Handle unnamed experiments
            name = "[unnamed]"
        return fnmatch.fnmatch(name.lower(), pattern.lower())

    def _has_all_tags(
        self, experiment: Dict[str, Any], required_tags: List[str]
    ) -> bool:
        """Check if experiment has all required tags."""
        exp_tags = set(experiment.get("tags", []))
        required_tags_set = set(required_tags)
        return required_tags_set.issubset(exp_tags)

    def _started_after(self, experiment: Dict[str, Any], after_time: datetime) -> bool:
        """Check if experiment started after the specified time."""
        started_at = experiment.get("started_at")
        if not started_at:
            return False

        try:
            # Parse the ISO format timestamp with proper timezone handling
            if started_at.endswith("Z"):
                exp_start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elif "+" in started_at:
                exp_start = datetime.fromisoformat(started_at)
            else:
                # No timezone info, assume UTC
                from datetime import timezone

                exp_start = datetime.fromisoformat(started_at).replace(
                    tzinfo=timezone.utc
                )
            return exp_start >= after_time
        except Exception:
            return False

    def _started_before(
        self, experiment: Dict[str, Any], before_time: datetime
    ) -> bool:
        """Check if experiment started before the specified time."""
        started_at = experiment.get("started_at")
        if not started_at:
            return False

        try:
            # Parse the ISO format timestamp with proper timezone handling
            if started_at.endswith("Z"):
                exp_start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            elif "+" in started_at:
                exp_start = datetime.fromisoformat(started_at)
            else:
                # No timezone info, assume UTC
                from datetime import timezone

                exp_start = datetime.fromisoformat(started_at).replace(
                    tzinfo=timezone.utc
                )
            return exp_start < before_time
        except Exception:
            return False

    def _ended_after(self, experiment: Dict[str, Any], after_time: datetime) -> bool:
        """Check if experiment ended after the specified time."""
        ended_at = experiment.get("ended_at")
        if not ended_at:
            return False

        try:
            # Parse the ISO format timestamp with proper timezone handling
            if ended_at.endswith("Z"):
                exp_end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            elif "+" in ended_at:
                exp_end = datetime.fromisoformat(ended_at)
            else:
                # No timezone info, assume UTC
                from datetime import timezone

                exp_end = datetime.fromisoformat(ended_at).replace(tzinfo=timezone.utc)
            return exp_end >= after_time
        except Exception:
            return False

    def _ended_before(self, experiment: Dict[str, Any], before_time: datetime) -> bool:
        """Check if experiment ended before the specified time."""
        ended_at = experiment.get("ended_at")
        if not ended_at:
            return False

        try:
            # Parse the ISO format timestamp with proper timezone handling
            if ended_at.endswith("Z"):
                exp_end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            elif "+" in ended_at:
                exp_end = datetime.fromisoformat(ended_at)
            else:
                # No timezone info, assume UTC
                from datetime import timezone

                exp_end = datetime.fromisoformat(ended_at).replace(tzinfo=timezone.utc)
            return exp_end < before_time
        except Exception:
            return False
