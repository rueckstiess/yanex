"""
Unified experiment filtering system for both CLI and Python API.

This module provides a single, consistent filtering interface that supports
flexible AND/OR combinations of filter criteria.
"""

import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.datetime_utils import parse_iso_timestamp
from .constants import EXPERIMENT_STATUSES_SET
from .manager import ExperimentManager


class ExperimentFilter:
    """
    Unified experiment filtering system supporting flexible filter combinations.

    This class implements the new filtering logic where:
    - Different filter types are combined with AND logic
    - Within list-based filters, OR logic is used (ids, status)
    - Tags use AND logic (must have ALL specified tags)
    - IDs are treated as another filter type, not mutually exclusive
    """

    def __init__(self, manager: ExperimentManager | None = None):
        """
        Initialize unified experiment filter.

        Args:
            manager: ExperimentManager instance (creates default if None)
        """
        self.manager = manager or ExperimentManager()

    def filter_experiments(
        self,
        ids: list[str] | None = None,
        status: str | list[str] | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        script_pattern: str | None = None,
        started_after: str | datetime | None = None,
        started_before: str | datetime | None = None,
        ended_after: str | datetime | None = None,
        ended_before: str | datetime | None = None,
        archived: bool | None = None,
        limit: int | None = None,
        sort_by: str = "created_at",
        sort_desc: bool = True,
        include_all: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Filter experiments based on multiple criteria with unified logic.

        Args:
            ids: List of experiment IDs to match (OR logic within list)
            status: Single status or list of statuses to match (OR logic within list)
            name: Glob pattern for name matching
            tags: List of tags - experiments must have ALL specified tags (AND logic)
            script_pattern: Glob pattern for script name matching (with or without .py extension)
            started_after: Filter experiments started after this time
            started_before: Filter experiments started before this time
            ended_after: Filter experiments ended after this time
            ended_before: Filter experiments ended before this time
            archived: True for archived only, False for non-archived only, None for both
            limit: Maximum number of results to return
            sort_by: Field to sort by (created_at, started_at, name, status)
            sort_desc: Sort in descending order
            include_all: If True, ignore default limit and return all matching experiments

        Returns:
            List of experiment metadata dictionaries matching ALL specified criteria

        Raises:
            ValueError: If invalid status or other parameters provided
        """
        # Validate and normalize inputs
        normalized_filters = self._normalize_filter_inputs(
            ids=ids,
            status=status,
            name_pattern=name,
            tags=tags,
            script_pattern=script_pattern,
            started_after=started_after,
            started_before=started_before,
            ended_after=ended_after,
            ended_before=ended_before,
            archived=archived,
        )

        # Load all experiments (including archived based on filter)
        include_archived = archived is not False  # Include if True or None
        all_experiments = self._load_all_experiments(include_archived)

        # Apply all filters with AND logic between filter types
        filtered = self._apply_all_filters(all_experiments, normalized_filters)

        # Sort results
        filtered = self._sort_experiments(filtered, sort_by, sort_desc)

        # Apply limit
        if not include_all and limit is not None:
            filtered = filtered[:limit]
        elif not include_all and limit is None:
            # Default limit of 10 if not explicitly requesting all
            filtered = filtered[:10]

        return filtered

    def _normalize_filter_inputs(
        self,
        ids: list[str] | None,
        status: str | list[str] | None,
        name_pattern: str | None,
        tags: list[str] | None,
        script_pattern: str | None,
        started_after: str | datetime | None,
        started_before: str | datetime | None,
        ended_after: str | datetime | None,
        ended_before: str | datetime | None,
        archived: bool | None,
    ) -> dict[str, Any]:
        """Validate and normalize all filter inputs."""
        normalized = {}

        # Normalize IDs
        if ids is not None:
            if not isinstance(ids, list):
                raise ValueError("ids must be a list of strings")
            if not all(isinstance(id_val, str) for id_val in ids):
                raise ValueError("All ids must be strings")
            if ids:  # Only add non-empty lists
                normalized["ids"] = ids

        # Normalize status (convert single string to list for consistent processing)
        if status is not None:
            if isinstance(status, str):
                status_list = [status]
            elif isinstance(status, list):
                status_list = status
            else:
                raise ValueError("status must be a string or list of strings")

            # Validate all statuses
            for s in status_list:
                if s not in EXPERIMENT_STATUSES_SET:
                    raise ValueError(
                        f"Invalid status '{s}'. Valid options: {', '.join(sorted(EXPERIMENT_STATUSES_SET))}"
                    )
            normalized["status"] = status_list

        # Normalize name pattern
        if name_pattern is not None:
            if not isinstance(name_pattern, str):
                raise ValueError("name_pattern must be a string")
            normalized["name_pattern"] = name_pattern

        # Normalize script pattern
        if script_pattern is not None:
            if not isinstance(script_pattern, str):
                raise ValueError("script_pattern must be a string")
            normalized["script_pattern"] = script_pattern

        # Normalize tags
        if tags is not None:
            if not isinstance(tags, list):
                raise ValueError("tags must be a list of strings")
            if not all(isinstance(tag, str) for tag in tags):
                raise ValueError("All tags must be strings")
            if tags:  # Only add non-empty lists
                normalized["tags"] = tags

        # Normalize datetime filters
        for dt_field in [
            "started_after",
            "started_before",
            "ended_after",
            "ended_before",
        ]:
            dt_value = locals()[dt_field]
            if dt_value is not None:
                if isinstance(dt_value, str):
                    # Parse string to datetime
                    from ..cli.filters.time_utils import parse_time_spec

                    try:
                        normalized[dt_field] = parse_time_spec(dt_value)
                    except ValueError as e:
                        raise ValueError(f"Invalid {dt_field} format: {e}")
                elif isinstance(dt_value, datetime):
                    normalized[dt_field] = dt_value
                else:
                    raise ValueError(f"{dt_field} must be a string or datetime object")

        # Normalize archived flag
        if archived is not None:
            if not isinstance(archived, bool):
                raise ValueError("archived must be a boolean")
            normalized["archived"] = archived

        return normalized

    def _apply_all_filters(
        self, experiments: list[dict[str, Any]], filters: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Apply all filters with AND logic between different filter types."""
        filtered = experiments

        # Apply ID filter (OR logic within list)
        if "ids" in filters:
            filtered = [exp for exp in filtered if exp["id"] in filters["ids"]]

        # Apply status filter (OR logic within list)
        if "status" in filters:
            filtered = [
                exp for exp in filtered if exp.get("status") in filters["status"]
            ]

        # Apply name pattern filter
        if "name_pattern" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._matches_name_pattern(exp, filters["name_pattern"])
            ]

        # Apply script pattern filter
        if "script_pattern" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._matches_script_pattern(exp, filters["script_pattern"])
            ]

        # Apply tags filter (AND logic - must have ALL tags)
        if "tags" in filters:
            filtered = [
                exp for exp in filtered if self._has_all_tags(exp, filters["tags"])
            ]

        # Apply time filters
        if "started_after" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._started_after(exp, filters["started_after"])
            ]

        if "started_before" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._started_before(exp, filters["started_before"])
            ]

        if "ended_after" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._ended_after(exp, filters["ended_after"])
            ]

        if "ended_before" in filters:
            filtered = [
                exp
                for exp in filtered
                if self._ended_before(exp, filters["ended_before"])
            ]

        # Apply archived filter
        if "archived" in filters:
            archived_value = filters["archived"]
            filtered = [
                exp for exp in filtered if exp.get("archived", False) == archived_value
            ]

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

    def _matches_script_pattern(self, experiment: dict[str, Any], pattern: str) -> bool:
        """
        Check if script name matches glob pattern.

        Matches against both the full filename (e.g., 'train.py') and stem (e.g., 'train').
        This allows users to filter with or without the .py extension.
        Returns False if no script_path in experiment metadata.

        Args:
            experiment: Experiment metadata dictionary
            pattern: Glob pattern to match against

        Returns:
            True if script matches pattern, False otherwise
        """
        script_path = experiment.get("script_path", "")
        if not script_path:
            return False

        # Create Path object once for performance
        script_path_obj = Path(script_path)
        script_name = script_path_obj.name  # Full filename: "train.py"
        script_stem = script_path_obj.stem  # Stem only: "train"

        # Match against both full name and stem for flexibility
        pattern_lower = pattern.lower()
        return fnmatch.fnmatch(script_name.lower(), pattern_lower) or fnmatch.fnmatch(
            script_stem.lower(), pattern_lower
        )

    def _has_all_tags(
        self, experiment: dict[str, Any], required_tags: list[str]
    ) -> bool:
        """Check if experiment has all required tags (AND logic)."""
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

    def _sort_experiments(
        self, experiments: list[dict[str, Any]], sort_by: str, sort_desc: bool
    ) -> list[dict[str, Any]]:
        """Sort experiments by specified field."""

        def get_sort_key(exp: dict[str, Any]) -> Any:
            if sort_by == "created_at":
                return exp.get("created_at", "")
            elif sort_by == "started_at":
                return exp.get("started_at", "")
            elif sort_by == "name":
                name = exp.get("name") or ""
                return name.lower()
            elif sort_by == "status":
                return exp.get("status", "")
            else:
                # Default to created_at
                return exp.get("created_at", "")

        return sorted(experiments, key=get_sort_key, reverse=sort_desc)

    def get_experiment_count(self, **filters) -> int:
        """Get count of experiments matching filters without loading full data."""
        experiments = self.filter_experiments(include_all=True, **filters)
        return len(experiments)

    def experiment_exists(
        self, experiment_id: str, include_archived: bool = True
    ) -> bool:
        """Check if an experiment exists without loading full metadata."""
        try:
            self.manager.storage.load_metadata(experiment_id, include_archived)
            return True
        except Exception:
            return False
