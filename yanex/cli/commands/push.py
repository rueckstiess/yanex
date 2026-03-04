"""Push experiments to a remote target."""

import click

from ...sync.target import parse_target
from ...sync.transport import sync_experiments_push
from ..error_handling import CLIErrorHandler
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
)


@click.command("push")
@click.argument("target")
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=False)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--progress", is_flag=True, help="Show transfer progress")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def push_experiments(
    ctx,
    target: str,
    ids: str | None,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    script_pattern: str | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    archived: bool,
    yes: bool,
    progress: bool,
    project: str | None,
    global_scope: bool,
):
    """
    Push experiments to a remote target.

    TARGET is an SSH host or S3 URL:

    \b
      SSH:  host, host:path, user@host:path
      S3:   s3://bucket/prefix

    SSH targets use ~/.ssh/config aliases. Default remote path
    is ~/.yanex/experiments/.

    By default, all experiments are included (global scope).
    Use filter options to push specific experiments.

    Examples:

    \b
        yanex push sky-dev                         # Push all to SSH host
        yanex push sky-dev -n "train*"             # Push matching name pattern
        yanex push sky-dev -s completed -t sweep   # Push completed sweeps
        yanex push user@gpu-box:~/experiments      # Custom remote path
        yanex push s3://my-bucket/experiments      # Push to S3
        yanex push sky-dev --ids a1b2,c3d4         # Push specific experiments
    """
    # Parse target
    sync_target = parse_target(target)

    # Parse filter arguments
    ids_list = None
    if ids:
        ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

    # Parse time filters
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Global scope by default: only apply project filter if explicitly provided
    resolved_project = project  # None if not provided = no project filter

    # Find local experiments matching filters
    filter_obj = ExperimentFilter()
    experiments = find_experiments_by_filters(
        filter_obj,
        ids=ids_list,
        status=status,
        name=name_pattern,
        tags=list(tags) if tags else None,
        script_pattern=script_pattern,
        started_after=started_after_dt,
        started_before=started_before_dt,
        ended_after=ended_after_dt,
        ended_before=ended_before_dt,
        archived=archived,
        project=resolved_project,
    )

    if not experiments:
        click.echo("No experiments found to push.")
        return

    # Confirm with user
    if not confirm_experiment_operation(
        experiments, "push", force=yes, operation_verb="pushed"
    ):
        click.echo("Push cancelled.")
        return

    # Execute sync
    experiment_ids = [exp["id"] for exp in experiments]
    local_dir = filter_obj.manager.storage.experiments_dir

    click.echo(
        f"Pushing {len(experiment_ids)} experiment(s) to {sync_target.display_name}..."
    )
    result = sync_experiments_push(
        experiment_ids, sync_target, local_dir, progress=progress
    )

    # Report results
    if result.all_succeeded:
        click.echo(f"Successfully pushed {result.success_count} experiment(s).")
    else:
        click.echo(
            f"Push completed: {result.success_count} succeeded, {result.failed_count} failed."
        )
        for error in result.errors:
            click.echo(f"  Error: {error}", err=True)
        ctx.exit(1)
