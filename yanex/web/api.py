"""
API routes for yanex web UI.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from ..cli.filters import ExperimentFilter
from ..core.dependency_graph import DependencyGraph
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
    sort_by: str = Query(
        "created_at", description="Sort by field: 'name', 'status', 'created_at'"
    ),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    page: int = Query(1, description="Page number for pagination"),
    archived: bool = Query(False, description="Include archived experiments"),
    project: str | None = Query(None, description="Filter by project name"),
) -> dict[str, Any]:
    """List experiments with filtering options."""
    try:
        # Normalize FastAPI Query defaults when function is called directly
        tags = tags if isinstance(tags, str) else None
        status = status if isinstance(status, str) else None
        name_pattern = name_pattern if isinstance(name_pattern, str) else None
        started_after = started_after if isinstance(started_after, str) else None
        started_before = started_before if isinstance(started_before, str) else None
        ended_after = ended_after if isinstance(ended_after, str) else None
        ended_before = ended_before if isinstance(ended_before, str) else None
        sort_by = sort_by if isinstance(sort_by, str) else "created_at"
        sort_order = sort_order if isinstance(sort_order, str) else "desc"
        project = project if isinstance(project, str) else None

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

        # Get ALL experiments first (no limit applied yet)
        # Note: archived parameter changed - True=only archived, False=only non-archived, None=both
        all_experiments = experiment_filter.filter_experiments(
            status=status,
            name=name_pattern,  # Changed from name_pattern to name
            tags=tag_list,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=archived,  # Changed from include_archived - now handles filtering internally
            project=project,
            limit=None,  # Get all experiments
            include_all=True,
        )
        # Note: Manual archived filtering removed - now handled by filter_experiments()

        # Apply sorting to ALL experiments
        # Handle legacy sort_order values
        if sort_order in ["newest", "oldest"]:
            # Legacy format - convert to new format
            reverse = sort_order == "newest"
            sort_by = "created_at"
        else:
            # New format
            reverse = sort_order == "desc"

        if sort_by == "name":
            all_experiments.sort(
                key=lambda x: (x.get("name") or "").lower(), reverse=reverse
            )
        elif sort_by == "status":
            all_experiments.sort(key=lambda x: x.get("status", ""), reverse=reverse)
        elif sort_by == "created_at":
            all_experiments.sort(key=lambda x: x.get("created_at", ""), reverse=reverse)
        else:
            # Default to created_at desc
            all_experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        # Calculate pagination
        total_experiments = len(all_experiments)
        total_pages = 1
        start_index = 0
        end_index = total_experiments

        if limit and limit > 0:
            total_pages = (total_experiments + limit - 1) // limit  # Ceiling division
            start_index = (page - 1) * limit
            end_index = min(start_index + limit, total_experiments)

        # Apply pagination
        experiments = all_experiments[start_index:end_index]

        return {
            "experiments": experiments,
            "total": total_experiments,
            "page": page,
            "total_pages": total_pages,
            "limit": limit or total_experiments,  # If no limit, use total count
            "has_next": page < total_pages,
            "has_prev": page > 1,
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
            archived=archived,  # Changed from include_archived
        )
        # Note: Manual archived filtering removed - now handled by filter_experiments()

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
            include_all=True,
            archived=False,  # Changed from include_archived
        )
        archived_experiments = experiment_filter.filter_experiments(
            include_all=True,
            archived=True,  # Changed from include_archived
        )
        # Note: Manual archived filtering removed - now handled by filter_experiments()

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


@router.get("/graph")
async def get_dependency_graph(
    limit: int | None = Query(
        None, description="Maximum number of experiments to consider"
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
    sort_by: str = Query(
        "created_at", description="Sort by field: 'name', 'status', 'created_at'"
    ),
    sort_order: str = Query(
        "desc", description="Sort order: 'asc'/'desc' or 'newest'/'oldest'"
    ),
    project: str | None = Query(None, description="Filter by project name"),
) -> dict[str, Any]:
    """Get the full dependency graph using the reverse graph.

    Returns all experiments that participate in at least one dependency
    relationship. Edges point from dependency to dependent (data-flow direction).
    """
    try:
        # Normalize FastAPI Query defaults when function is called directly
        tags = tags if isinstance(tags, str) else None
        status = status if isinstance(status, str) else None
        name_pattern = name_pattern if isinstance(name_pattern, str) else None
        started_after = started_after if isinstance(started_after, str) else None
        started_before = started_before if isinstance(started_before, str) else None
        ended_after = ended_after if isinstance(ended_after, str) else None
        ended_before = ended_before if isinstance(ended_before, str) else None
        sort_by = sort_by if isinstance(sort_by, str) else "created_at"
        sort_order = sort_order if isinstance(sort_order, str) else "desc"
        project = project if isinstance(project, str) else None

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

        # Apply same filtering semantics as experiment list endpoint
        filtered_experiments = experiment_filter.filter_experiments(
            status=status,
            name=name_pattern,
            tags=tag_list,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=False,
            project=project,
            limit=None,
            include_all=True,
        )

        # Apply sorting semantics
        # "none" preserves ExperimentFilter's natural ordering
        if sort_order in ["newest", "oldest"]:
            reverse = sort_order == "newest"
            sort_by = "created_at"
            if sort_by == "name":
                filtered_experiments.sort(
                    key=lambda x: (x.get("name") or "").lower(), reverse=reverse
                )
            elif sort_by == "status":
                filtered_experiments.sort(
                    key=lambda x: x.get("status", ""), reverse=reverse
                )
            else:
                filtered_experiments.sort(
                    key=lambda x: x.get("created_at", ""), reverse=reverse
                )
        elif sort_order in ["asc", "desc"]:
            reverse = sort_order == "desc"
            if sort_by == "name":
                filtered_experiments.sort(
                    key=lambda x: (x.get("name") or "").lower(), reverse=reverse
                )
            elif sort_by == "status":
                filtered_experiments.sort(
                    key=lambda x: x.get("status", ""), reverse=reverse
                )
            else:
                filtered_experiments.sort(
                    key=lambda x: x.get("created_at", ""), reverse=reverse
                )

        if limit and limit > 0:
            filtered_experiments = filtered_experiments[:limit]

        allowed_ids = {exp["id"] for exp in filtered_experiments}

        dep_graph = DependencyGraph(manager.storage)
        graph = dep_graph._reverse

        # Keep only edges where both endpoints match filters
        filtered_edges = []
        node_ids: set[str] = set()
        for u, v, data in graph.edges(data=True):
            if u in allowed_ids and v in allowed_ids:
                filtered_edges.append(
                    {
                        "source": u,
                        "target": v,
                        "slot": data.get("slot", ""),
                    }
                )
                node_ids.add(u)
                node_ids.add(v)

        # Also include filtered experiments that are part of the dependency graph,
        # even if no same-filter edge remains (e.g. failed child of completed parent)
        for exp_id in allowed_ids:
            if exp_id in graph:
                node_ids.add(exp_id)

        nodes = []
        for node_id in node_ids:
            # Load config (parameters) for tooltip display
            params: dict[str, Any] = {}
            try:
                params = manager.storage.load_config(
                    node_id, include_archived=True
                )
            except Exception:
                pass

            # Load metrics for tooltip display
            metrics: dict[str, Any] = {}
            try:
                results = manager.storage.load_results(
                    node_id, include_archived=True
                )
                if results:
                    for result in results:
                        for k, v in result.items():
                            if k not in ("step", "timestamp"):
                                metrics[k] = v
            except Exception:
                pass

            meta = dep_graph.get_metadata(node_id)
            nodes.append(
                {
                    "id": node_id,
                    "name": meta.get("name", ""),
                    "status": meta.get("status", "unknown"),
                    "script": meta.get("script", ""),
                    "params": params,
                    "metrics": metrics,
                }
            )

        return {"nodes": nodes, "edges": filtered_edges}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
