"""
Delete experiments permanently.
"""

import click

from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


def _collect_dependents_recursive(
    storage, experiment_ids: set[str], include_archived: bool = True
) -> set[str]:
    """Recursively collect all experiments that depend on the given experiments.

    Args:
        storage: Storage instance
        experiment_ids: Set of experiment IDs to find dependents for
        include_archived: Whether to include archived experiments

    Returns:
        Set of all dependent experiment IDs (transitively)
    """
    all_dependents = set()
    to_process = set(experiment_ids)

    while to_process:
        current_id = to_process.pop()

        # Load dependencies to get depended_by list
        deps = storage.load_dependencies(current_id, include_archived=include_archived)
        if not deps or "depended_by" not in deps:
            continue

        # Get direct dependents
        for dep_entry in deps.get("depended_by", []):
            dependent_id = dep_entry["experiment_id"]

            # If we haven't seen this dependent yet, add it to process
            if dependent_id not in all_dependents:
                all_dependents.add(dependent_id)
                to_process.add(dependent_id)

    return all_dependents


@click.command("delete")
@click.argument("experiment_identifiers", nargs=-1)
@experiment_filter_options(
    include_ids=False, include_archived=True, include_limit=False
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--cascade",
    is_flag=True,
    help="Delete experiments that depend on the selected experiments (cascade down)",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def delete_experiments(
    ctx,
    experiment_identifiers: tuple,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    script_pattern: str | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    archived: bool,
    depends_on: str | None,
    depends_on_script: str | None,
    root: bool,
    leaf: bool,
    force: bool,
    cascade: bool,
):
    """
    Permanently delete experiments.

    ⚠️  WARNING: This operation cannot be undone!

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    By default, deletion fails if other experiments depend on the selected ones.
    Use --cascade to delete dependent experiments too.

    Examples:
    \\b
        yanex delete exp1 exp2               # Delete specific experiments
        yanex delete -s failed               # Delete all failed experiments
        yanex delete -a --ended-before "6 months ago"
        yanex delete -n "*test*"             # Delete experiments with "test" in name
        yanex delete -t temp                 # Delete experiments with "temp" tag
        yanex delete exp123 --cascade        # Delete exp123 and all experiments depending on it
        yanex delete -s failed --cascade     # Delete failed + all their dependents
    """
    filter_obj = ExperimentFilter()

    # Validate mutually exclusive targeting
    has_filters = any(
        [
            status,
            name_pattern,
            tags,
            script_pattern,
            started_after,
            started_before,
            ended_after,
            ended_before,
            depends_on,
            depends_on_script,
            root,
            leaf,
        ]
    )

    CLIErrorHandler.validate_targeting_options(
        list(experiment_identifiers), has_filters, "delete"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to delete
    if experiment_identifiers:
        # Delete specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj, list(experiment_identifiers), archived=archived
        )
    else:
        # Delete experiments by filter criteria
        experiments = find_experiments_by_filters(
            filter_obj,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=archived,
            depends_on=depends_on,
            depends_on_script=depends_on_script,
            root=root,
            leaf=leaf,
        )

    # Filter experiments based on archived flag
    if archived:
        experiments = [exp for exp in experiments if exp.get("archived", False)]
    else:
        experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        location = "archived" if archived else "regular"
        click.echo(f"No {location} experiments found to delete.")
        return

    # Check for dependents and handle cascade
    original_exp_ids = {exp["id"] for exp in experiments}
    dependent_ids = _collect_dependents_recursive(
        filter_obj.manager.storage, original_exp_ids, include_archived=True
    )

    if dependent_ids and not cascade:
        # Error: experiments have dependents but cascade not enabled
        click.echo(
            f"Error: Cannot delete {len(experiments)} experiment(s) because "
            f"{len(dependent_ids)} other experiment(s) depend on them.",
            err=True,
        )
        click.echo("\nExperiments with dependents:", err=True)
        for exp in experiments:
            exp_id = exp["id"]
            exp_name = exp.get("name", "[unnamed]")
            # Count direct dependents
            deps = filter_obj.manager.storage.load_dependencies(
                exp_id, include_archived=True
            )
            direct_count = len(deps.get("depended_by", [])) if deps else 0
            click.echo(
                f"  {exp_id} ({exp_name}) - {direct_count} direct dependent(s)",
                err=True,
            )
        click.echo(
            "\nUse --cascade to delete these experiments and all their dependents.",
            err=True,
        )
        ctx.exit(1)

    # If cascade enabled and there are dependents, add them to the list
    if cascade and dependent_ids:
        click.echo(
            f"Cascade delete enabled: found {len(dependent_ids)} dependent experiment(s)"
        )
        click.echo()

        # Load full experiment metadata for dependents
        dependent_experiments = []
        for dep_id in dependent_ids:
            try:
                dep_metadata = filter_obj.manager.storage.load_metadata(
                    dep_id, include_archived=True
                )
                if dep_metadata:
                    dependent_experiments.append(dep_metadata)
            except Exception:
                # If we can't load metadata, create minimal entry
                dependent_experiments.append({"id": dep_id, "name": "[unavailable]"})

        # Show what will be deleted
        click.echo(f"Original selection: {len(experiments)} experiment(s)")
        click.echo(f"Dependent experiments: {len(dependent_experiments)} experiment(s)")
        click.echo(f"Total to delete: {len(experiments) + len(dependent_experiments)}")
        click.echo()

        # Add dependents to experiments list
        experiments.extend(dependent_experiments)

    # Show experiments and get confirmation (always required for deletion)
    operation_verb = "permanently deleted"
    if not confirm_experiment_operation(experiments, "delete", force, operation_verb):
        click.echo("Delete operation cancelled.")
        return

    # Additional warning for bulk deletions
    if len(experiments) > 1 and not force:
        click.echo()
        click.echo("⚠️  You are about to permanently delete multiple experiments.")
        click.echo("   This action cannot be undone!")
        if not click.confirm("Are you absolutely sure?", default=False):
            click.echo("Delete operation cancelled.")
            return

    # Delete experiments using centralized reporter
    click.echo(f"Deleting {len(experiments)} experiment(s)...")
    reporter = BulkOperationReporter("delete")

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            # Clean up bidirectional dependency tracking before deletion
            # Remove this experiment from its dependencies' depended_by lists
            deps = filter_obj.manager.storage.load_dependencies(
                experiment_id, include_archived=True
            )
            if deps and "resolved_dependencies" in deps:
                for _slot_name, dep_id in deps["resolved_dependencies"].items():
                    try:
                        filter_obj.manager.storage.remove_dependent(
                            dep_id, experiment_id, include_archived=True
                        )
                    except Exception:
                        # Best effort - continue even if this fails
                        pass

            # Delete the experiment
            if exp.get("archived", False):
                # Delete from archived directory
                filter_obj.manager.storage.delete_archived_experiment(experiment_id)
            else:
                # Delete from regular directory
                filter_obj.manager.storage.delete_experiment(experiment_id)

            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
