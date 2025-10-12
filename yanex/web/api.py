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


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, archived: bool = False) -> dict[str, Any]:
    """Get detailed information about a specific experiment."""
    try:
        # Find the experiment
        experiments = experiment_filter.filter_experiments(
            include_all=True,
            include_archived=archived,
        )

        # Filter by archived status
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        # Find matching experiment
        experiment = None
        for exp in experiments:
            if exp["id"] == experiment_id or exp["id"].startswith(experiment_id):
                experiment = exp
                break

        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        # Get additional details
        try:
            config = manager.storage.load_config(
                experiment_id, include_archived=archived
            )
        except Exception:
            config = {}

        try:
            results = manager.storage.load_results(
                experiment_id, include_archived=archived
            )
        except Exception:
            results = []

        try:
            metadata = manager.storage.load_metadata(
                experiment_id, include_archived=archived
            )
        except Exception:
            metadata = {}

        # Get artifacts
        try:
            exp_dir = manager.storage.get_experiment_dir(
                experiment_id, include_archived=archived
            )
            artifacts_dir = exp_dir / "artifacts"
            artifacts = []
            if artifacts_dir.exists():
                for artifact_path in artifacts_dir.iterdir():
                    if artifact_path.is_file():
                        artifacts.append(
                            {
                                "name": artifact_path.name,
                                "size": artifact_path.stat().st_size,
                                "modified": artifact_path.stat().st_mtime,
                            }
                        )
        except Exception:
            artifacts = []

        return {
            "experiment": experiment,
            "config": config,
            "results": results,
            "metadata": metadata,
            "artifacts": artifacts,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/artifacts/{artifact_name}")
async def download_artifact(
    experiment_id: str, artifact_name: str, archived: bool = False
):
    """Download an artifact file."""
    try:
        exp_dir = manager.storage.get_experiment_dir(
            experiment_id, include_archived=archived
        )
        artifact_path = exp_dir / "artifacts" / artifact_name

        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        from fastapi.responses import FileResponse

        return FileResponse(
            path=str(artifact_path),
            filename=artifact_name,
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> dict[str, Any]:
    """Get system status and statistics."""
    try:
        # Get basic statistics
        all_experiments = experiment_filter.filter_experiments(
            include_all=True, include_archived=False
        )
        archived_experiments = experiment_filter.filter_experiments(
            include_all=True, include_archived=True
        )
        archived_experiments = [
            exp for exp in archived_experiments if exp.get("archived", False)
        ]

        # Count by status
        status_counts = {}
        for exp in all_experiments:
            status = exp.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_experiments": len(all_experiments),
            "archived_experiments": len(archived_experiments),
            "status_counts": status_counts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
