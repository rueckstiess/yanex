"""
API routes for yanex web UI.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..cli.filters import ExperimentFilter
from ..core.manager import ExperimentManager

router = APIRouter()

# Initialize experiment manager and filter
manager = ExperimentManager()
experiment_filter = ExperimentFilter()


@router.get("/experiments")
async def list_experiments(
    limit: int | None = Query(
        None, description="Maximum number of experiments to return"
    ),
    status: str | None = Query(None, description="Filter by experiment status"),
    name_pattern: str | None = Query(
        None, description="Filter by name using glob patterns"
    ),
    tags: str | None = Query(
        None, description="Comma-separated list of tags to filter by"
    ),
    started_after: str | None = Query(
        None, description="Filter experiments started after this date"
    ),
    started_before: str | None = Query(
        None, description="Filter experiments started before this date"
    ),
    ended_after: str | None = Query(
        None, description="Filter experiments ended after this date"
    ),
    ended_before: str | None = Query(
        None, description="Filter experiments ended before this date"
    ),
    sort_order: str = Query("newest", description="Sort order: 'newest' or 'oldest'"),
    archived: bool = Query(False, description="Include archived experiments"),
) -> dict[str, Any]:
    """List experiments with filtering options."""
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Parse time filters
        started_after_dt = None
        started_before_dt = None
        ended_after_dt = None
        ended_before_dt = None

        if started_after:
            from ..cli.error_handling import CLIErrorHandler

            started_after_dt, _, _, _ = CLIErrorHandler.parse_time_filters(
                started_after, None, None, None
            )

        if started_before:
            from ..cli.error_handling import CLIErrorHandler

            _, started_before_dt, _, _ = CLIErrorHandler.parse_time_filters(
                None, started_before, None, None
            )

        if ended_after:
            from ..cli.error_handling import CLIErrorHandler

            _, _, ended_after_dt, _ = CLIErrorHandler.parse_time_filters(
                None, None, ended_after, None
            )

        if ended_before:
            from ..cli.error_handling import CLIErrorHandler

            _, _, _, ended_before_dt = CLIErrorHandler.parse_time_filters(
                None, None, None, ended_before
            )

        # Get experiments
        experiments = experiment_filter.filter_experiments(
            status=status,
            name_pattern=name_pattern,
            tags=tag_list,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            limit=limit,
            include_all=limit is None,
            include_archived=archived,
        )

        # Filter by archived status
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        # Apply sort order
        if sort_order == "oldest":
            experiments.sort(key=lambda x: x.get("created_at", ""), reverse=False)
        else:  # newest (default)
            experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "experiments": experiments,
            "total": len(experiments),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
