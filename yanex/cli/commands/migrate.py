"""
Migrate experiments to latest storage schema version.
"""

import click

from ...core.migrations import (
    CURRENT_VERSION,
    get_pending_migrations,
    get_storage_version,
    migrate_experiment,
    needs_migration,
)
from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from ..formatters import (
    OutputFormat,
    echo_format_info,
    format_options,
    is_machine_format,
    resolve_output_format,
)
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("migrate")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=False)
@click.option(
    "--all",
    "migrate_all",
    is_flag=True,
    help="Migrate all experiments (regular and archived)",
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be migrated without making changes",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def migrate_experiments(
    ctx,
    experiment_identifiers: tuple,
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
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
    migrate_all: bool,
    force: bool,
    dry_run: bool,
):
    """
    Migrate experiments to latest storage schema version.

    This command applies pending migrations to update experiment storage format.
    Migrations are idempotent - running them multiple times is safe.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    Use --all to migrate all experiments (regular and archived).

    Supports multiple output formats:

    \\b
      --format json      Output result as JSON (for scripting/AI processing)
      --format csv       Output result as CSV (for data analysis)
      --format markdown  Output result as markdown

    Examples:

    \\b
        # Preview all migrations (dry run)
        yanex migrate --all --dry-run

        # Migrate all experiments
        yanex migrate --all

        # Migrate specific experiments
        yanex migrate exp1 exp2

        # Migrate by filters
        yanex migrate -s completed --ended-before "1 month ago"

        # Force (skip confirmation)
        yanex migrate --all --force

        # Output result as JSON
        yanex migrate --all --format json
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
    filter_obj = ExperimentFilter()

    # Parse comma-separated IDs into a list
    ids_list = None
    if ids:
        ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

    # Validate targeting options
    # Note: name_pattern="" is a valid filter for unnamed experiments
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
        ]
    )

    # Must specify --all, identifiers, or filters
    if not migrate_all and not experiment_identifiers and not has_filters:
        raise click.ClickException(
            "Must specify --all, experiment identifiers, or filter options.\n"
            "Use --all to migrate all experiments, or specify identifiers/filters."
        )

    # Can't combine --all with identifiers
    if migrate_all and experiment_identifiers:
        raise click.ClickException("Cannot combine --all with experiment identifiers.")

    # Can't combine --all with filters
    if migrate_all and has_filters:
        raise click.ClickException("Cannot combine --all with filter options.")

    # Validate targeting for identifiers vs filters (but not if --all)
    if not migrate_all:
        CLIErrorHandler.validate_targeting_options(
            list(experiment_identifiers), has_filters, "migrate"
        )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to migrate
    if migrate_all:
        # Get all experiments (regular + archived)
        experiments = find_experiments_by_filters(
            filter_obj,
            status=None,
            name=None,
            tags=None,
            script_pattern=None,
            started_after=None,
            started_before=None,
            ended_after=None,
            ended_before=None,
            archived=False,  # Get regular experiments
        )
        # Also get archived experiments
        archived_experiments = find_experiments_by_filters(
            filter_obj,
            status=None,
            name=None,
            tags=None,
            script_pattern=None,
            started_after=None,
            started_before=None,
            ended_after=None,
            ended_before=None,
            archived=True,  # Get archived experiments
        )
        # Filter archived list to only actually archived ones
        archived_experiments = [
            exp for exp in archived_experiments if exp.get("archived", False)
        ]
        experiments = experiments + archived_experiments
    elif experiment_identifiers:
        # Migrate specific experiments by ID/name
        # Use archived=None to search both regular and archived experiments
        # when specific IDs are provided (user knows what they want)
        experiments = find_experiments_by_identifiers(
            filter_obj, list(experiment_identifiers), archived=None
        )
    else:
        # Migrate experiments by filter criteria
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
        )
        # Filter based on archived flag
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        echo_format_info("No experiments found.", fmt)
        return

    # Check which experiments need migration
    needs_update = []
    up_to_date = []

    for exp in experiments:
        if needs_migration(exp):
            needs_update.append(exp)
        else:
            up_to_date.append(exp)

    # Report status (only in console mode)
    if not is_machine_format(fmt):
        click.echo(
            f"Checking {len(experiments)} experiment(s) for pending migrations..."
        )
        click.echo()

    if dry_run:
        # Detailed dry-run output
        _show_dry_run_details(filter_obj, needs_update, up_to_date, fmt)
        return

    if not needs_update:
        echo_format_info(
            f"All {len(experiments)} experiment(s) are up to date (v{CURRENT_VERSION}).",
            fmt,
        )
        return

    # Show confirmation for migrations (only in console mode)
    if not is_machine_format(fmt):
        click.echo(f"{len(needs_update)} experiment(s) need migration:")
        for exp in needs_update:
            version = get_storage_version(exp)
            version_str = f"v{version}" if version is not None else "legacy"
            exp_name = exp.get("name") or "[unnamed]"
            click.echo(
                f"  {exp['id']} ({exp_name}) - {version_str} -> v{CURRENT_VERSION}"
            )
        click.echo()

        if len(up_to_date) > 0:
            click.echo(f"{len(up_to_date)} experiment(s) already up to date.")
            click.echo()

    # For machine-readable output, skip confirmation
    effective_force = force or is_machine_format(fmt)

    # Get confirmation
    if not effective_force:
        if not confirm_experiment_operation(
            needs_update, "migrate", effective_force, "migrated", default_yes=True
        ):
            echo_format_info("Migration cancelled.", fmt)
            return

    # Migrate experiments
    echo_format_info(f"Migrating {len(needs_update)} experiment(s)...", fmt)
    reporter = BulkOperationReporter("migrate", output_format=fmt)

    for exp in needs_update:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            # Get experiment directory (include_archived=True to search both locations)
            exp_dir = (
                filter_obj.manager.storage.directory_manager.get_experiment_directory(
                    experiment_id, include_archived=True
                )
            )
            # Apply migrations
            results = migrate_experiment(exp_dir, dry_run=False)
            # Check if any migration was applied
            if any(r.applied for r in results):
                reporter.report_success(experiment_id, exp_name)
            else:
                # Already up to date (shouldn't happen but be safe)
                reporter.report_success(experiment_id, f"{exp_name} (no changes)")
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())


def _show_dry_run_details(
    filter_obj: ExperimentFilter,
    needs_update: list[dict],
    up_to_date: list[dict],
    fmt: OutputFormat,
) -> None:
    """Show detailed dry-run output for migrations."""
    from ..formatters import format_json, format_markdown_table

    # For machine-readable output, format as JSON/CSV/markdown
    if fmt == OutputFormat.JSON:
        data = {
            "dry_run": True,
            "needs_migration": [
                {
                    "id": exp["id"],
                    "name": exp.get("name") or "[unnamed]",
                    "current_version": get_storage_version(exp),
                    "target_version": CURRENT_VERSION,
                }
                for exp in needs_update
            ],
            "up_to_date": [
                {"id": exp["id"], "name": exp.get("name") or "[unnamed]"}
                for exp in up_to_date
            ],
            "summary": {
                "needs_migration_count": len(needs_update),
                "up_to_date_count": len(up_to_date),
            },
        }
        click.echo(format_json(data))
        return

    if fmt == OutputFormat.CSV:
        from ..formatters import format_csv

        rows = []
        for exp in needs_update:
            rows.append(
                {
                    "id": exp["id"],
                    "name": exp.get("name") or "[unnamed]",
                    "status": "needs_migration",
                    "current_version": get_storage_version(exp),
                    "target_version": CURRENT_VERSION,
                }
            )
        for exp in up_to_date:
            rows.append(
                {
                    "id": exp["id"],
                    "name": exp.get("name") or "[unnamed]",
                    "status": "up_to_date",
                    "current_version": CURRENT_VERSION,
                    "target_version": CURRENT_VERSION,
                }
            )
        if rows:
            click.echo(
                format_csv(
                    rows,
                    columns=[
                        "id",
                        "name",
                        "status",
                        "current_version",
                        "target_version",
                    ],
                ),
                nl=False,
            )
        return

    if fmt == OutputFormat.MARKDOWN:
        rows = []
        for exp in needs_update:
            rows.append(
                {
                    "ID": exp["id"],
                    "Name": exp.get("name") or "[unnamed]",
                    "Status": "needs_migration",
                    "Current": str(get_storage_version(exp)),
                    "Target": str(CURRENT_VERSION),
                }
            )
        for exp in up_to_date:
            rows.append(
                {
                    "ID": exp["id"],
                    "Name": exp.get("name") or "[unnamed]",
                    "Status": "up_to_date",
                    "Current": str(CURRENT_VERSION),
                    "Target": str(CURRENT_VERSION),
                }
            )
        if rows:
            click.echo(
                format_markdown_table(
                    rows, columns=["ID", "Name", "Status", "Current", "Target"]
                )
            )
        click.echo()
        click.echo(
            f"**Summary:** {len(needs_update)} need migration, {len(up_to_date)} up to date"
        )
        return

    # Console output (default)
    # Show experiments that need migration with details
    for exp in needs_update:
        exp_name = exp.get("name") or "[unnamed]"
        version = get_storage_version(exp)

        click.echo(f"{exp['id']} ({exp_name}) - needs migration:")

        # Get pending migrations and show their descriptions
        pending = get_pending_migrations(version)
        for migration in pending:
            from_str = (
                f"v{migration.from_version}"
                if migration.from_version is not None
                else "legacy"
            )
            click.echo(
                f"  {from_str} -> v{migration.to_version}: {migration.description}"
            )

            # Run migration in dry-run mode to get specific changes
            try:
                exp_dir = filter_obj.manager.storage.directory_manager.get_experiment_directory(
                    exp["id"], include_archived=True
                )
                result = migration.migrate_fn(exp_dir, dry_run=True)
                for change in result.changes:
                    click.echo(f"    - {change}")
            except Exception as e:
                click.echo(f"    - Error checking changes: {e}")

        click.echo()

    # Show up-to-date experiments (brief)
    if up_to_date:
        click.echo(
            f"{len(up_to_date)} experiment(s) already up to date (v{CURRENT_VERSION}):"
        )
        for exp in up_to_date[:5]:  # Show first 5
            exp_name = exp.get("name") or "[unnamed]"
            click.echo(f"  {exp['id']} ({exp_name})")
        if len(up_to_date) > 5:
            click.echo(f"  ... and {len(up_to_date) - 5} more")
        click.echo()

    # Summary
    click.echo("Summary:")
    click.echo(f"  - {len(needs_update)} experiment(s) need migration")
    click.echo(f"  - {len(up_to_date)} experiment(s) up to date")
    click.echo()
    click.echo("Dry run completed. No changes were made.")
    click.echo("Run without --dry-run to apply migrations.")
