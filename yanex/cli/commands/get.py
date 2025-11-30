"""
Get specific field values from experiments.

This command is optimized for AI agents and scripting, with support for
bash command substitution to build dynamic sweeps.
"""

import json
from typing import Any

import click

import yanex.results as yr
from yanex.cli.error_handling import CLIErrorHandler
from yanex.cli.filters import ExperimentFilter
from yanex.cli.filters.arguments import experiment_filter_options
from yanex.utils.dict_utils import get_nested_value

from .confirm import find_experiment

# Fields that come directly from metadata
METADATA_FIELDS = {
    "id",
    "name",
    "status",
    "description",
    "tags",
    "script_path",
    "created_at",
    "started_at",
    "completed_at",
    "failed_at",
    "cancelled_at",
    "error_message",
    "cancellation_reason",
}


def resolve_field_value(
    exp: yr.Experiment, field: str, default_value: str
) -> tuple[Any, bool]:
    """
    Resolve a field value from an experiment.

    Args:
        exp: Experiment object
        field: Field path (e.g., "status", "params.lr", "metrics.accuracy")
        default_value: Default value for missing fields

    Returns:
        Tuple of (value, found) where found indicates if the field was found
    """
    # Handle special field: dependencies
    if field == "dependencies":
        deps = exp.dependencies
        if deps:
            return deps, True
        return {}, True

    # Handle special field: params (list of available parameter names)
    if field == "params":
        params = exp.get_params()
        if params:
            return sorted(params.keys()), True
        return [], True

    # Handle special field: metrics (list of available metric names)
    if field == "metrics":
        metrics = exp.get_metrics(as_dataframe=False)
        if metrics:
            # Get unique metric names (excluding step/timestamp)
            metric_names = set()
            for entry in metrics:
                for key in entry.keys():
                    if key not in ("step", "timestamp", "last_updated"):
                        metric_names.add(key)
            return sorted(metric_names), True
        return [], True

    # Handle params.* fields
    if field.startswith("params."):
        param_key = field[7:]  # Remove "params." prefix
        value = exp.get_param(param_key)
        if value is not None:
            return value, True
        return default_value, False

    # Handle metrics.* fields - get last logged value
    if field.startswith("metrics."):
        metric_name = field[8:]  # Remove "metrics." prefix
        value = exp.get_metric(metric_name)
        if value is not None:
            # If it's a list, return the last value
            if isinstance(value, list) and len(value) > 0:
                return value[-1], True
            return value, True
        return default_value, False

    # Handle git.* fields
    if field.startswith("git."):
        git_key = field[4:]  # Remove "git." prefix
        metadata = exp._load_metadata()
        git_info = metadata.get("git", {})
        if git_key in git_info:
            return git_info[git_key], True
        return default_value, False

    # Handle environment.* fields
    if field.startswith("environment."):
        env_path = field[12:]  # Remove "environment." prefix
        metadata = exp._load_metadata()
        env_info = metadata.get("environment", {})
        value = get_nested_value(env_info, env_path)
        if value is not None:
            return value, True
        return default_value, False

    # Handle direct metadata fields
    if field == "id":
        return exp.id, True
    if field == "name":
        return exp.name if exp.name else default_value, exp.name is not None
    if field == "status":
        return exp.status, True
    if field == "description":
        return (
            exp.description if exp.description else default_value,
            exp.description is not None,
        )
    if field == "tags":
        return exp.tags, True
    if field == "script_path":
        return str(exp.script_path) if exp.script_path else default_value, (
            exp.script_path is not None
        )

    # Handle timestamp fields
    if field in (
        "created_at",
        "started_at",
        "completed_at",
        "failed_at",
        "cancelled_at",
    ):
        metadata = exp._load_metadata()
        value = metadata.get(field)
        if value:
            return value, True
        return default_value, False

    # Handle error fields
    if field in ("error_message", "cancellation_reason"):
        metadata = exp._load_metadata()
        value = metadata.get(field)
        if value:
            return value, True
        return default_value, False

    # Try as nested path in metadata
    metadata = exp._load_metadata()
    value = get_nested_value(metadata, field)
    if value is not None:
        return value, True

    return default_value, False


def format_value(value: Any, json_output: bool = False) -> str:
    """Format a value for output."""
    if json_output:
        return json.dumps(value)

    if value is None:
        return ""

    if isinstance(value, dict):
        # Format dependencies as slot=id pairs
        if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return " ".join(f"{k}={v}" for k, v in sorted(value.items()))
        return json.dumps(value)

    if isinstance(value, list):
        # Format lists as comma-separated
        return ", ".join(str(v) for v in value)

    return str(value)


def format_value_for_csv(value: Any) -> str:
    """Format a value for CSV output (no ID prefix, comma-separated)."""
    if value is None:
        return ""

    if isinstance(value, dict):
        # Format dependencies as slot=id pairs
        if all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return ",".join(f"{k}={v}" for k, v in sorted(value.items()))
        return json.dumps(value)

    if isinstance(value, list):
        return ",".join(str(v) for v in value)

    return str(value)


@click.command("get")
@click.argument("field", required=True)
@click.argument("experiment_id", required=False)
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=True)
@click.option(
    "--no-id",
    is_flag=True,
    help="Omit experiment ID prefix in multi-experiment output",
)
@click.option(
    "--csv",
    "csv_output",
    is_flag=True,
    help="Output comma-separated values on single line (for bash substitution)",
)
@click.option(
    "--json",
    "-j",
    "json_output",
    is_flag=True,
    help="Output as JSON (useful for complex values)",
)
@click.option(
    "--default",
    "default_value",
    default="[not_found]",
    help="Value for missing fields (default: [not_found])",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def get_field(
    ctx,
    experiment_id: str | None,
    field: str,
    # Filter options from decorator
    ids: tuple | None,
    status: str | None,
    name_pattern: str | None,
    script_pattern: str | None,
    tags: tuple | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    archived: bool,
    limit: int | None,
    # Output options
    no_id: bool,
    csv_output: bool,
    json_output: bool,
    default_value: str,
):
    """
    Get a specific field value from experiment(s).

    EXPERIMENT_ID can be an experiment ID, name, or ID prefix.
    If omitted, use filter options to select experiments.

    FIELD is the field path to retrieve:

    \b
      - Core: id, name, status, description, tags
      - Timing: created_at, started_at, completed_at
      - Script: script_path, error_message
      - Git: git.branch, git.commit_hash
      - Environment: environment.python.version
      - Parameters: params (list names), params.<key> (get value)
      - Metrics: metrics (list names), metrics.<key> (last value)
      - Dependencies: dependencies - returns slot=id pairs

    \b
    Examples:
      yanex get status abc123              Get status of experiment abc123
      yanex get params abc123              List available parameter names
      yanex get params.lr abc123           Get learning rate parameter
      yanex get metrics abc123             List available metric names
      yanex get metrics.accuracy abc123    Get last logged accuracy
      yanex get dependencies abc123        Get dependencies as slot=id pairs
      yanex get id -n "train-*"            Get IDs of matching experiments
      yanex get id -n "train-*" --csv      Get IDs comma-separated (for sweeps)
      yanex get params.lr -s completed --csv   Get learning rates for sweep
      yanex get tags abc123 --json         Get tags as JSON array

    \b
    Bash substitution for sweeps:
      yanex run train.py -D data=$(yanex get id -n "*-prep-*" --csv)
      yanex run train.py -p lr=$(yanex get params.lr -s completed --csv)
    """
    # Determine if we have filters or a single experiment
    has_filters = any(
        [
            ids,
            status,
            name_pattern,
            script_pattern,
            tags,
            started_after,
            started_before,
            ended_after,
            ended_before,
        ]
    )

    if experiment_id and has_filters:
        raise click.ClickException(
            "Cannot specify both EXPERIMENT_ID and filter options. Use one or the other."
        )

    if not experiment_id and not has_filters:
        raise click.ClickException(
            "Must specify either EXPERIMENT_ID or filter options (e.g., --name, --status)."
        )

    # Single experiment mode
    if experiment_id:
        filter_obj = ExperimentFilter()
        experiment = find_experiment(filter_obj, experiment_id, archived)

        if experiment is None:
            raise click.ClickException(
                f"No experiment found with ID or name '{experiment_id}'"
            )

        if isinstance(experiment, list):
            # Multiple matches - show them and exit
            click.echo(f"Multiple experiments found with name '{experiment_id}':")
            for exp in experiment:
                click.echo(f"  {exp['id']}: {exp.get('name', '[unnamed]')}")
            raise click.ClickException("Please use a specific experiment ID.")

        # Get the experiment as Experiment object
        exp = yr.get_experiment(experiment["id"])
        value, found = resolve_field_value(exp, field, default_value)

        # Output single value
        if json_output:
            click.echo(json.dumps(value))
        elif csv_output:
            click.echo(format_value_for_csv(value), nl=False)
        else:
            click.echo(format_value(value))

        return

    # Multi-experiment mode with filters
    # Build filter kwargs
    filter_kwargs = {}
    if ids:
        filter_kwargs["ids"] = list(ids)
    if status:
        filter_kwargs["status"] = status
    if name_pattern:
        filter_kwargs["name"] = name_pattern
    if script_pattern:
        filter_kwargs["script_pattern"] = script_pattern
    if tags:
        filter_kwargs["tags"] = list(tags)
    if started_after:
        filter_kwargs["started_after"] = started_after
    if started_before:
        filter_kwargs["started_before"] = started_before
    if ended_after:
        filter_kwargs["ended_after"] = ended_after
    if ended_before:
        filter_kwargs["ended_before"] = ended_before
    if archived:
        filter_kwargs["archived"] = archived
    if limit:
        filter_kwargs["limit"] = limit

    # Get experiments using Results API
    experiments = yr.get_experiments(**filter_kwargs)

    if not experiments:
        # No output for empty results (useful for scripting)
        return

    # Collect all values
    results = []
    for exp in experiments:
        value, found = resolve_field_value(exp, field, default_value)
        results.append((exp.id, value))

    # Output based on mode
    if json_output:
        if field == "id":
            # Just list of IDs
            output = [exp_id for exp_id, _ in results]
        else:
            # List of {id, value} objects
            output = [{"id": exp_id, "value": value} for exp_id, value in results]
        click.echo(json.dumps(output))
        return

    if csv_output:
        # Comma-separated values, no ID prefix
        values = [format_value_for_csv(value) for _, value in results]
        # Output without trailing newline for bash substitution
        click.echo(",".join(values), nl=False)
        return

    # Default mode: one line per experiment
    for exp_id, value in results:
        formatted = format_value(value)
        if field == "id" or no_id:
            # Just the value (no ID prefix for 'id' field or when --no-id)
            click.echo(formatted)
        else:
            # ID: value format
            click.echo(f"{exp_id}: {formatted}")
