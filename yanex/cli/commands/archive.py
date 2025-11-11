"""
Archive experiments - move them to archived directory.
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


def _collect_dependencies_recursive(
    storage, experiment_ids: set[str], include_archived: bool = False
) -> set[str]:
    """Recursively collect all experiments that the given experiments depend on.

    Args:
        storage: Storage instance
        experiment_ids: Set of experiment IDs to find dependencies for
        include_archived: Whether to include archived experiments

    Returns:
        Set of all dependency experiment IDs (transitively)
    """
    all_dependencies = set()
    to_process = set(experiment_ids)

    while to_process:
        current_id = to_process.pop()

        # Load dependencies
        deps = storage.load_dependencies(current_id, include_archived=include_archived)
        if not deps or "resolved_dependencies" not in deps:
            continue

        # Get direct dependencies
        for dep_id in deps.get("resolved_dependencies", {}).values():
            # If we haven't seen this dependency yet, add it to process
            if dep_id not in all_dependencies:
                all_dependencies.add(dep_id)
                to_process.add(dep_id)

    return all_dependencies


def _collect_dependents_recursive(
    storage, experiment_ids: set[str], include_archived: bool = False
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


@click.command("archive")
@click.argument("experiment_identifiers", nargs=-1)
@experiment_filter_options(
    include_ids=False, include_archived=False, include_limit=False
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--with-dependencies",
    is_flag=True,
    help="Archive experiments that the selected ones depend on (upstream)",
)
@click.option(
    "--with-dependents",
    is_flag=True,
    help="Archive experiments that depend on the selected ones (downstream)",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def archive_experiments(
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
    depends_on: str | None,
    depends_on_script: str | None,
    root: bool,
    leaf: bool,
    force: bool,
    with_dependencies: bool,
    with_dependents: bool,
):
    """
    Archive experiments by moving them to archived directory.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Use --with-dependencies to archive upstream dependencies too.
    Use --with-dependents to archive downstream dependents too.

    Examples:
    \\b
        yanex archive exp1 exp2              # Archive specific experiments
        yanex archive -s failed              # Archive all failed experiments
        yanex archive -s completed --ended-before "1 month ago"
        yanex archive -n "*training*"        # Archive experiments with "training" in name
        yanex archive -t experiment-v1       # Archive experiments with specific tag
        yanex archive exp123 --with-dependencies  # Archive exp123 + what it depends on
        yanex archive exp123 --with-dependents    # Archive exp123 + what depends on it
        yanex archive -s completed --with-dependents  # Archive completed + dependents
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
        list(experiment_identifiers), has_filters, "archive"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to archive
    if experiment_identifiers:
        # Archive specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj,
            list(experiment_identifiers),
            archived=False,  # Can't archive already archived experiments
        )
    else:
        # Archive experiments by filter criteria
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
            archived=False,
            depends_on=depends_on,
            depends_on_script=depends_on_script,
            root=root,
            leaf=leaf,
        )

    # Filter out already archived experiments (extra safety)
    experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        click.echo("No experiments found to archive.")
        return

    # Collect related experiments if requested
    original_exp_ids = {exp["id"] for exp in experiments}
    related_ids = set()

    if with_dependencies:
        dependency_ids = _collect_dependencies_recursive(
            filter_obj.manager.storage, original_exp_ids, include_archived=False
        )
        related_ids.update(dependency_ids)

    if with_dependents:
        dependent_ids = _collect_dependents_recursive(
            filter_obj.manager.storage, original_exp_ids, include_archived=False
        )
        related_ids.update(dependent_ids)

    # Load related experiments
    if related_ids:
        related_experiments = []
        for related_id in related_ids:
            try:
                metadata = filter_obj.manager.storage.load_metadata(
                    related_id, include_archived=False
                )
                if metadata and not metadata.get("archived", False):
                    related_experiments.append(metadata)
            except Exception:
                # If we can't load, create minimal entry
                related_experiments.append({"id": related_id, "name": "[unavailable]"})

        if related_experiments:
            click.echo(f"Including {len(related_experiments)} related experiment(s):")
            if with_dependencies:
                dependency_count = len(
                    _collect_dependencies_recursive(
                        filter_obj.manager.storage,
                        original_exp_ids,
                        include_archived=False,
                    )
                )
                click.echo(f"  - {dependency_count} upstream dependencies")
            if with_dependents:
                dependent_count = len(
                    _collect_dependents_recursive(
                        filter_obj.manager.storage,
                        original_exp_ids,
                        include_archived=False,
                    )
                )
                click.echo(f"  - {dependent_count} downstream dependents")
            click.echo(
                f"Total to archive: {len(experiments) + len(related_experiments)}"
            )
            click.echo()

            # Add related experiments to archive list
            experiments.extend(related_experiments)

    # Show experiments and get confirmation
    if not confirm_experiment_operation(
        experiments, "archive", force, default_yes=True
    ):
        click.echo("Archive operation cancelled.")
        return

    # Archive experiments using centralized reporter
    click.echo(f"Archiving {len(experiments)} experiment(s)...")
    reporter = BulkOperationReporter("archive")

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            filter_obj.manager.storage.archive_experiment(experiment_id)
            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
