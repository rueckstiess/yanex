"""
Get specific field values from experiments.

This command is optimized for AI agents and scripting, with support for
bash command substitution to build dynamic sweeps.
"""

import json
import sys
import time
from typing import Any

import click

import yanex.results as yr
from yanex.cli.error_handling import CLIErrorHandler
from yanex.cli.filters import ExperimentFilter
from yanex.cli.filters.arguments import experiment_filter_options
from yanex.cli.formatters import format_markdown_table
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

# Fields that support --head/--tail options (returns first/last N lines)
HEAD_TAIL_SUPPORTED_FIELDS = {"stdout", "stderr"}


def reconstruct_cli_command(cli_args: dict[str, Any]) -> str:
    """Reconstruct original CLI command from cli_args (includes sweep syntax).

    Args:
        cli_args: The cli_args dictionary from experiment metadata.

    Returns:
        The reconstructed yanex run command string.
    """
    parts = ["yanex run"]

    if cli_args.get("script"):
        parts.append(cli_args["script"])

    if cli_args.get("clone_from"):
        parts.append(f"--clone-from {cli_args['clone_from']}")

    for config in cli_args.get("config", []):
        parts.append(f"-c {config}")

    for dep in cli_args.get("depends_on", []):
        parts.append(f"-D {dep}")

    for param in cli_args.get("param", []):
        parts.append(f'-p "{param}"')

    if cli_args.get("name"):
        parts.append(f'-n "{cli_args["name"]}"')

    if cli_args.get("description"):
        parts.append(f'-d "{cli_args["description"]}"')

    for tag in cli_args.get("tag", []):
        parts.append(f'-t "{tag}"')

    if cli_args.get("stage"):
        parts.append("--stage")

    if cli_args.get("parallel"):
        parts.append(f"-j {cli_args['parallel']}")

    return " ".join(parts)


def reconstruct_run_command(exp: yr.Experiment) -> str:
    """Reconstruct reproducible command using resolved config values.

    This returns a command that can reproduce this specific experiment,
    using resolved parameter values instead of sweep syntax.

    Args:
        exp: The experiment object.

    Returns:
        The reconstructed yanex run command string with resolved values.
    """
    metadata = exp._load_metadata()
    cli_args = metadata.get("cli_args", {})
    config = exp.get_params()  # Resolved parameters

    parts = ["yanex run"]

    if cli_args.get("script"):
        parts.append(cli_args["script"])

    if cli_args.get("clone_from"):
        parts.append(f"--clone-from {cli_args['clone_from']}")

    for cfg in cli_args.get("config", []):
        parts.append(f"-c {cfg}")

    # Use resolved dependencies from metadata
    deps = exp.dependencies or {}
    for slot, dep_id in deps.items():
        parts.append(f"-D {slot}={dep_id}")

    # Only output parameters that were passed via CLI -p flags
    # Use resolved values (for sweep parameters that got expanded)
    for param_str in cli_args.get("param", []):
        # Parse the key from "key=value" or "key=sweep(...)"
        if "=" in param_str:
            key = param_str.split("=", 1)[0]
            # Get the resolved value from config using dot notation
            resolved_value = get_nested_value(config, key)
            if resolved_value is not None:
                parts.append(f'-p "{key}={resolved_value}"')
            else:
                # Fallback to original if key not found (shouldn't happen)
                parts.append(f'-p "{param_str}"')

    if exp.name:
        parts.append(f'-n "{exp.name}"')

    if cli_args.get("description"):
        parts.append(f'-d "{cli_args["description"]}"')

    for tag in exp.tags or []:
        parts.append(f'-t "{tag}"')

    return " ".join(parts)


def resolve_field_value(
    exp: yr.Experiment,
    field: str,
    default_value: str,
    tail: int | None = None,
    head: int | None = None,
) -> tuple[Any, bool]:
    """
    Resolve a field value from an experiment.

    Args:
        exp: Experiment object
        field: Field path (e.g., "status", "params.lr", "metrics.accuracy")
        default_value: Default value for missing fields
        tail: If specified, return last N lines (only for stdout/stderr fields)
        head: If specified, return first N lines (only for stdout/stderr fields)

    Returns:
        Tuple of (value, found) where found indicates if the field was found
    """
    # Handle stdout/stderr fields (read from artifacts)
    if field in ("stdout", "stderr"):
        artifact_name = f"{field}.txt"
        artifact_path = exp.artifacts_dir / artifact_name
        if artifact_path.exists():
            content = artifact_path.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Handle head and tail combination
            if head is not None and tail is not None and head > 0 and tail > 0:
                # Show first N lines, ..., last N lines
                total_lines = len(lines)
                if head + tail >= total_lines:
                    # Just return all lines if head + tail covers everything
                    content = "\n".join(lines)
                else:
                    head_lines = lines[:head]
                    tail_lines = lines[-tail:]
                    content = "\n".join(head_lines) + "\n...\n" + "\n".join(tail_lines)
            elif head is not None and head > 0:
                content = "\n".join(lines[:head])
            elif tail is not None and tail > 0:
                content = "\n".join(lines[-tail:])

            return content, True
        return default_value, False

    # Handle cli-command field (original CLI invocation with sweep syntax)
    if field == "cli-command":
        metadata = exp._load_metadata()
        cli_args = metadata.get("cli_args", {})
        if cli_args and cli_args.get("script"):
            return reconstruct_cli_command(cli_args), True
        return default_value, False

    # Handle run-command field (reproducible command with resolved values)
    if field == "run-command":
        metadata = exp._load_metadata()
        cli_args = metadata.get("cli_args", {})
        if cli_args and cli_args.get("script"):
            return reconstruct_run_command(exp), True
        return default_value, False

    # Handle experiment-dir field (experiment directory path)
    if field == "experiment-dir":
        return str(exp.experiment_dir), True

    # Handle artifacts-dir field (artifacts directory path)
    if field == "artifacts-dir":
        return str(exp.artifacts_dir), True

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


def follow_output(
    exp: yr.Experiment,
    field: str,
    initial_tail: int | None = None,
    poll_interval: float = 0.5,
) -> None:
    """Stream output from experiment in real-time.

    Args:
        exp: Experiment object to follow
        field: Either "stdout" or "stderr"
        initial_tail: If set, show last N lines initially instead of full content
        poll_interval: How often to check for new content (seconds)
    """
    artifact_path = exp.artifacts_dir / f"{field}.txt"
    position = 0

    # Show initial content (optionally last N lines)
    if artifact_path.exists():
        content = artifact_path.read_text(encoding="utf-8")
        if initial_tail and content:
            lines = content.splitlines()
            if len(lines) > initial_tail:
                click.echo(f"[showing last {initial_tail} lines]")
                click.echo("\n".join(lines[-initial_tail:]))
            else:
                click.echo(content, nl=False)
                if content and not content.endswith("\n"):
                    click.echo()  # Ensure newline
        else:
            click.echo(content, nl=False)
            if content and not content.endswith("\n"):
                click.echo()  # Ensure newline
        position = artifact_path.stat().st_size

    # Check initial status
    current_status = yr.get_experiment(exp.id).status
    if current_status != "running":
        click.echo(f"[experiment {current_status}]")
        return

    click.echo(f"[following {field}, Ctrl+C to stop]")

    # Poll for new content until experiment completes
    try:
        while True:
            # Check for new content
            if artifact_path.exists():
                current_size = artifact_path.stat().st_size
                if current_size > position:
                    with open(artifact_path, encoding="utf-8") as f:
                        f.seek(position)
                        new_content = f.read()
                        sys.stdout.write(new_content)
                        sys.stdout.flush()
                        position = current_size

            # Check if experiment is still running
            current_status = yr.get_experiment(exp.id).status
            if current_status != "running":
                # Final read to catch any remaining output
                if artifact_path.exists():
                    current_size = artifact_path.stat().st_size
                    if current_size > position:
                        with open(artifact_path, encoding="utf-8") as f:
                            f.seek(position)
                            final_content = f.read()
                            sys.stdout.write(final_content)
                            sys.stdout.flush()
                click.echo(f"\n[experiment {current_status}]")
                break

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        click.echo("\n[interrupted]")


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
    "--markdown",
    "-m",
    "markdown_output",
    is_flag=True,
    help="Output as GitHub-flavored markdown table (for multi-experiment output)",
)
@click.option(
    "--default",
    "default_value",
    default="[not_found]",
    help="Value for missing fields (default: [not_found])",
)
@click.option(
    "--tail",
    type=int,
    default=None,
    help="Return last N lines (only for stdout/stderr fields)",
)
@click.option(
    "--head",
    type=int,
    default=None,
    help="Return first N lines (only for stdout/stderr fields)",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    help="Follow output in real-time (only for stdout/stderr on single experiment)",
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
    markdown_output: bool,
    default_value: str,
    tail: int | None,
    head: int | None,
    follow: bool,
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
      - Output: stdout, stderr (with --head/--tail N, or --follow/-f)
      - Commands: cli-command (original), run-command (reproducible)
      - Paths: experiment-dir, artifacts-dir
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
      yanex get stdout abc123              Get full stdout of experiment
      yanex get stdout abc123 --tail 50    Get last 50 lines of stdout
      yanex get stdout abc123 --head 10    Get first 10 lines of stdout
      yanex get stdout abc123 --head 5 --tail 5  Show first 5 and last 5 lines
      yanex get stdout abc123 -f           Follow stdout in real-time
      yanex get stdout abc123 --tail 20 -f Show last 20 lines then follow
      yanex get stderr abc123              Get stderr output
      yanex get stdout -s running --tail 5 Check running experiments progress
      yanex get cli-command abc123         Get original CLI invocation (with sweep syntax)
      yanex get run-command abc123         Get reproducible command (resolved values)
      yanex get experiment-dir abc123      Get experiment directory path
      yanex get artifacts-dir abc123       Get artifacts directory path
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
    # Validate mutually exclusive output modes
    output_mode_count = sum([json_output, csv_output, markdown_output])
    if output_mode_count > 1:
        raise click.ClickException(
            "Cannot specify multiple output formats. Choose one of --json, --csv, or --markdown."
        )

    # Validate --head/--tail only applies to stdout/stderr
    if tail is not None and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--tail option only applies to stdout/stderr fields, not '{field}'"
        )
    if head is not None and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--head option only applies to stdout/stderr fields, not '{field}'"
        )

    # Validate --csv and --markdown not supported for stdout/stderr
    if csv_output and field in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(f"--csv output is not supported for '{field}' field")
    if markdown_output and field in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--markdown output is not supported for '{field}' field"
        )

    # Validate --follow only applies to stdout/stderr
    if follow and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--follow option only applies to stdout/stderr fields, not '{field}'"
        )

    # Validate --follow incompatible with --csv/--json/--markdown
    if follow and csv_output:
        raise click.ClickException("--follow cannot be used with --csv output")
    if follow and json_output:
        raise click.ClickException("--follow cannot be used with --json output")
    if follow and markdown_output:
        raise click.ClickException("--follow cannot be used with --markdown output")

    # Validate --follow incompatible with --head (but --tail is ok for initial display)
    if follow and head is not None:
        raise click.ClickException(
            "--follow cannot be used with --head (use --tail to show last N lines before following)"
        )

    # Determine if we have filters or a single experiment
    # Note: name_pattern="" is a valid filter for unnamed experiments
    has_filters = any(
        [
            ids,
            status,
            name_pattern is not None,
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

    # Validate --follow requires single experiment (not filters)
    if follow and not experiment_id:
        raise click.ClickException(
            "--follow requires a single experiment ID, not filter options"
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

        # Handle --follow mode for stdout/stderr
        if follow:
            follow_output(exp, field, initial_tail=tail)
            return

        value, found = resolve_field_value(exp, field, default_value, tail, head)

        # Output single value
        if json_output:
            click.echo(json.dumps(value))
        elif csv_output:
            click.echo(format_value_for_csv(value), nl=False)
        elif markdown_output:
            # Single row markdown table
            rows = [{"ID": experiment["id"], field: format_value(value)}]
            click.echo(format_markdown_table(rows, ["ID", field]))
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
    if name_pattern is not None:
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
        value, found = resolve_field_value(exp, field, default_value, tail, head)
        results.append((exp.id, value))

    # Output based on mode
    if json_output:
        if field == "id":
            # Just list of IDs
            output = [exp_id for exp_id, _ in results]
        elif field in HEAD_TAIL_SUPPORTED_FIELDS:
            # For stdout/stderr, use the field name as key
            output = [{"id": exp_id, field: value} for exp_id, value in results]
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

    if markdown_output:
        # Markdown table with ID and value columns
        rows = []
        for exp_id, value in results:
            rows.append({"ID": exp_id, field: format_value(value)})
        click.echo(format_markdown_table(rows, ["ID", field]))
        return

    # Special header format for stdout/stderr multi-experiment output
    if field in HEAD_TAIL_SUPPORTED_FIELDS:
        for i, (exp_id, value) in enumerate(results):
            if i > 0:
                click.echo()  # Blank line between experiments
            click.echo(f"[experiment: {exp_id}]")
            if value and value != default_value:
                click.echo(value)
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
