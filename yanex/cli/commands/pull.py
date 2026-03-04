"""Pull experiments from a remote target."""

import click

from ...core.manager import ExperimentManager
from ...sync.filter_utils import filter_experiment_dicts
from ...sync.remote_metadata import fetch_remote_metadata
from ...sync.target import parse_target
from ...sync.transport import sync_experiments_pull
from ..error_handling import CLIErrorHandler
from ..filters.arguments import experiment_filter_options
from .confirm import confirm_experiment_operation


@click.command("pull")
@click.argument("target")
@experiment_filter_options(
    include_ids=True, include_archived=False, include_limit=False
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--progress", is_flag=True, help="Show transfer progress")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def pull_experiments(
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
    yes: bool,
    progress: bool,
    project: str | None,
    global_scope: bool,
):
    """
    Pull experiments from a remote target.

    TARGET is an SSH host or S3 URL:

    \b
      SSH:  host, host:path, user@host:path
      S3:   s3://bucket/prefix

    SSH targets use ~/.ssh/config aliases. Default remote path
    is ~/.yanex/experiments/.

    Reads remote experiment metadata to apply filters, then
    transfers matching experiments. By default, all remote
    experiments are included (global scope).

    Examples:

    \b
        yanex pull sky-dev                         # Pull all from SSH host
        yanex pull sky-dev -n "train*"             # Pull matching name pattern
        yanex pull sky-dev -s completed            # Pull completed experiments
        yanex pull user@gpu-box:~/experiments      # Custom remote path
        yanex pull s3://my-bucket/experiments      # Pull from S3
        yanex pull sky-dev --ids a1b2,c3d4         # Pull specific experiments
    """
    # Parse target
    sync_target = parse_target(target)

    # Fetch remote metadata
    click.echo(f"Fetching experiment metadata from {sync_target.display_name}...")
    remote_experiments = fetch_remote_metadata(sync_target)

    if not remote_experiments:
        click.echo("No experiments found on remote.")
        return

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

    # Apply filters to remote metadata
    has_filters = any(
        [
            ids_list,
            status,
            name_pattern is not None,
            tags,
            script_pattern,
            started_after,
            started_before,
            ended_after,
            ended_before,
            resolved_project,
        ]
    )

    if has_filters:
        experiments = filter_experiment_dicts(
            remote_experiments,
            ids=ids_list,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            project=resolved_project,
        )
    else:
        experiments = remote_experiments

    if not experiments:
        click.echo("No experiments on remote match the specified filters.")
        return

    # Confirm with user
    if not confirm_experiment_operation(
        experiments, "pull", force=yes, operation_verb="pulled"
    ):
        click.echo("Pull cancelled.")
        return

    # Determine local experiments directory
    manager = ExperimentManager()
    local_dir = manager.storage.experiments_dir

    # Execute sync
    experiment_ids = [exp["id"] for exp in experiments]

    click.echo(
        f"Pulling {len(experiment_ids)} experiment(s) from {sync_target.display_name}..."
    )
    result = sync_experiments_pull(
        experiment_ids, sync_target, local_dir, progress=progress
    )

    # Report results
    if result.all_succeeded:
        click.echo(f"Successfully pulled {result.success_count} experiment(s).")
    else:
        click.echo(
            f"Pull completed: {result.success_count} succeeded, {result.failed_count} failed."
        )
        for error in result.errors:
            click.echo(f"  Error: {error}", err=True)
        ctx.exit(1)
