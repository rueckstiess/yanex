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
from yanex.cli.formatters import (
    GETTER_TYPES,
    GetterOutput,
    OutputFormat,
    format_options,
    get_getter_type,
    resolve_output_format,
)
from yanex.cli.formatters.lineage import (
    get_lineage_ids,
    lineage_to_csv,
    lineage_to_json,
)
from yanex.core.dependency_graph import DependencyGraph
from yanex.core.storage import ExperimentStorage
from yanex.utils.dict_utils import get_nested_value

from .confirm import find_experiment

# Valid field prefixes that accept dynamic suffixes (e.g., params.lr, metrics.accuracy)
DYNAMIC_FIELD_PREFIXES = ("params.", "metrics.", "environment.")


def validate_field(field: str) -> None:
    """Validate that a field name is recognized.

    Args:
        field: Field name to validate.

    Raises:
        click.ClickException: If field is not a valid getter field.
    """
    # Check exact match in known fields
    if field in GETTER_TYPES:
        return

    # Check dynamic prefixes (params.*, metrics.*, environment.*)
    for prefix in DYNAMIC_FIELD_PREFIXES:
        if field.startswith(prefix):
            return

    # Field is not recognized - build helpful error message
    valid_fields = sorted(GETTER_TYPES.keys())

    # Find similar fields for typo suggestions
    suggestions = _find_similar_fields(field, valid_fields)

    error_msg = f"Unknown field: '{field}'"
    if suggestions:
        error_msg += f"\n\nDid you mean: {', '.join(suggestions)}?"
    error_msg += "\n\nUse 'yanex get --help' to see available fields."

    raise click.ClickException(error_msg)


def _find_similar_fields(field: str, valid_fields: list[str]) -> list[str]:
    """Find fields similar to the given field (for typo suggestions).

    Uses simple substring matching and edit distance heuristics.

    Args:
        field: The unknown field entered by user.
        valid_fields: List of valid field names.

    Returns:
        List of up to 3 similar field names.
    """
    suggestions = []

    field_lower = field.lower()

    for valid in valid_fields:
        valid_lower = valid.lower()

        # Check if field is a substring or vice versa
        if field_lower in valid_lower or valid_lower in field_lower:
            suggestions.append(valid)
            continue

        # Check for common prefix (at least 3 chars)
        common_prefix_len = 0
        for i, (c1, c2) in enumerate(zip(field_lower, valid_lower, strict=False)):
            if c1 == c2:
                common_prefix_len = i + 1
            else:
                break

        if common_prefix_len >= 3:
            suggestions.append(valid)

    # Return up to 3 unique suggestions
    return list(dict.fromkeys(suggestions))[:3]


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

# Fields that display lineage graphs (support --depth and --ids-only)
LINEAGE_FIELDS = {"upstream", "downstream", "lineage"}


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

    # Handle artifacts field (list all artifact paths, one per line)
    if field == "artifacts":
        artifacts_dir = exp.artifacts_dir
        if artifacts_dir.exists():
            artifact_paths = sorted(artifacts_dir.rglob("*"))
            # Filter to only files (not directories)
            artifact_files = [str(p) for p in artifact_paths if p.is_file()]
            if artifact_files:
                return "\n".join(artifact_files), True
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


def _handle_lineage_field(
    exp_ids: list[str],
    field: str,
    depth: int,
    ids_only: bool,
    fmt: OutputFormat,
) -> None:
    """Handle lineage field output (upstream, downstream, lineage).

    Args:
        exp_ids: List of experiment IDs to get lineage for.
        field: One of "upstream", "downstream", or "lineage".
        depth: Maximum traversal depth.
        ids_only: If True, output only IDs comma-separated.
        fmt: Output format.
    """
    import os
    from pathlib import Path

    from rich.console import Console

    from yanex.cli.formatters.lineage import render_lineage_components

    # Get experiments directory from environment (same logic as ExperimentManager)
    env_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
    if env_dir:
        experiments_dir = Path(env_dir)
    else:
        experiments_dir = Path.home() / ".yanex" / "experiments"

    # Build dependency graph (loads all experiments once)
    storage = ExperimentStorage(experiments_dir)
    dep_graph = DependencyGraph(storage)

    # Validate all experiments exist
    missing = [eid for eid in exp_ids if not dep_graph.experiment_exists(eid)]
    if missing:
        raise click.ClickException(f"Experiment(s) not found: {', '.join(missing)}")

    target_set = set(exp_ids)

    # Get the appropriate combined graph based on field
    if field == "upstream":
        graph = dep_graph.get_multi_upstream(exp_ids, max_depth=depth)
    elif field == "downstream":
        graph = dep_graph.get_multi_downstream(exp_ids, max_depth=depth)
    else:  # lineage
        graph = dep_graph.get_multi_lineage(exp_ids, max_depth=depth)

    # Handle --ids-only output
    if ids_only:
        ids = get_lineage_ids(graph)
        click.echo(",".join(ids), nl=False)
        return

    # Handle different output formats
    if fmt == OutputFormat.JSON:
        output = lineage_to_json(graph, target_set)
        click.echo(json.dumps(output, indent=2))
    elif fmt == OutputFormat.CSV:
        output = lineage_to_csv(graph)
        click.echo(output)
    elif fmt == OutputFormat.MARKDOWN:
        # For markdown, output as a simple edge list table
        click.echo("| From | To | Slot |")
        click.echo("| --- | --- | --- |")
        for u, v, data in graph.edges(data=True):
            slot = data.get("slot", "")
            click.echo(f"| {u} | {v} | {slot} |")
    elif fmt == OutputFormat.SWEEP:
        # Sweep format: comma-separated IDs
        ids = get_lineage_ids(graph)
        click.echo(",".join(ids), nl=False)
    else:
        # Default: render as ASCII DAG(s)
        from yanex.cli.formatters.theme import TARGET_STYLE

        console = Console()
        components = render_lineage_components(graph, target_set)

        # Show legend explaining the target marker
        console.print(
            f"\n[dim]The[/dim] [{TARGET_STYLE}]<*>[/{TARGET_STYLE}] [dim]marker indicates experiments matching the ID or filter.[/dim]",
            highlight=False,
        )

        for i, component_output in enumerate(components):
            if i > 0:
                console.print()  # Blank line between components
            console.print()  # Blank line before each tree
            # Disable highlight to prevent Rich from auto-colorizing numbers
            console.print(component_output, highlight=False)


@click.command("get")
@click.argument("field", required=True)
@click.argument("experiment_id", required=False)
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=True)
@format_options(include_sweep=True)
@click.option(
    "--no-id",
    is_flag=True,
    help="Omit experiment ID prefix in multi-experiment output (legacy, use --format instead)",
    hidden=True,
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
@click.option(
    "--depth",
    type=int,
    default=10,
    help="Maximum depth for lineage traversal (default: 10, only for lineage fields)",
)
@click.option(
    "--ids-only",
    is_flag=True,
    help="Output only experiment IDs comma-separated (only for lineage fields)",
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
    # Output format options
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
    # Other options
    no_id: bool,
    default_value: str,
    tail: int | None,
    head: int | None,
    follow: bool,
    depth: int,
    ids_only: bool,
):
    """
    Get a specific field value from experiment(s).

    EXPERIMENT_ID can be an experiment ID, name, or ID prefix.
    If omitted, use filter options to select experiments.

    \b
    Available fields:
      id, name, status, description, tags
          Experiment metadata (single values or list for tags)
      created_at, started_at, completed_at, failed_at, cancelled_at
          Timestamps in ISO format
      script_path, error_message, cancellation_reason
          Script path and failure/cancellation details
      params
          List available parameter names
      params.<key>
          Get specific parameter value (e.g., params.lr, params.model.size)
      metrics
          List available metric names
      metrics.<key>
          Get last logged metric value (e.g., metrics.accuracy)
      stdout, stderr
          Output content (supports --head N, --tail N, --follow/-f)
      artifacts
          List all artifact file paths (one per line)
      cli-command
          Original CLI invocation (preserves sweep syntax)
      run-command
          Reproducible command (with resolved parameter values)
      experiment-dir, artifacts-dir
          Directory paths for the experiment
      dependencies
          Dependency slot=id pairs (e.g., data=abc123 model=def456)
      upstream
          DAG of dependencies (what this experiment depends on)
      downstream
          DAG of dependents (what depends on this experiment)
      lineage
          Full DAG (upstream + downstream combined)
      git.branch, git.commit_hash, git.dirty, git.remote_url
          Git state at experiment creation
      environment.python.version, environment.<path>
          Environment details (nested paths supported)

    \b
    Output formats (--format / -F):
      default    Human-readable (ID: value for multi-experiment)
      json       JSON with {"id": ..., "value": ...} structure
      csv        CSV with ID column and headers
      markdown   GitHub-flavored markdown table
      sweep      Comma-separated values only (for bash substitution)

    \b
    Examples (single experiment):
      yanex get status abc123              Experiment status
      yanex get params.lr abc123           Parameter value
      yanex get metrics.accuracy abc123    Last logged metric
      yanex get stdout abc123 --tail 20    Last 20 lines of output
      yanex get stdout abc123 -f           Follow output in real-time
      yanex get cli-command abc123         Original CLI invocation
      yanex get artifacts abc123           List artifact files

    \b
    Examples (lineage visualization):
      yanex get upstream abc123            Show dependencies as DAG
      yanex get downstream abc123          Show dependents as DAG
      yanex get lineage abc123             Show full dependency graph
      yanex get lineage abc123 --depth 3   Limit traversal depth
      yanex get lineage abc123 --ids-only  Get IDs only (for scripting)
      yanex get lineage abc123 -F json     JSON format (nodes + edges)

    \b
    Examples (multi-experiment lineage):
      yanex get lineage -n "train-*"       Lineage of all train-* experiments
      yanex get upstream -s completed      Dependencies of completed experiments
      yanex get downstream -t baseline     What depends on baseline-tagged exps
      yanex get lineage --ids a1,b2,c3     Lineage of specific experiments

    \b
    Examples (multiple experiments with filters):
      yanex get id -s completed            IDs of completed experiments
      yanex get id -n "train-*" -F sweep   IDs comma-separated for sweeps
      yanex get params.lr -t baseline      Learning rates from tagged exps
      yanex get stdout -s running --tail 5 Check progress of running exps

    \b
    Bash substitution:
      yanex run train.py -D data=$(yanex get id -n "*-prep-*" -F sweep)
      yanex run train.py -p lr=$(yanex get params.lr -s completed -F sweep)
    """
    # Resolve output format from --format option or legacy flags
    # Legacy --csv maps to SWEEP for backwards compatibility
    fmt = resolve_output_format(
        output_format, json_flag, csv_flag, markdown_flag, csv_means_sweep=True
    )

    # Validate field name is recognized (catch typos early)
    validate_field(field)

    # Validate --head/--tail only applies to stdout/stderr
    if tail is not None and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--tail option only applies to stdout/stderr fields, not '{field}'"
        )
    if head is not None and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--head option only applies to stdout/stderr fields, not '{field}'"
        )

    # Validate --follow only applies to stdout/stderr
    if follow and field not in HEAD_TAIL_SUPPORTED_FIELDS:
        raise click.ClickException(
            f"--follow option only applies to stdout/stderr fields, not '{field}'"
        )

    # Validate --follow incompatible with non-default formats
    if follow and fmt != OutputFormat.DEFAULT:
        raise click.ClickException(f"--follow cannot be used with --format {fmt.value}")

    # Validate --follow incompatible with --head (but --tail is ok for initial display)
    if follow and head is not None:
        raise click.ClickException(
            "--follow cannot be used with --head (use --tail to show last N lines before following)"
        )

    # Validate --depth only applies to lineage fields
    if depth != 10 and field not in LINEAGE_FIELDS:
        raise click.ClickException(
            f"--depth option only applies to lineage fields (upstream, downstream, lineage), not '{field}'"
        )

    # Validate --ids-only only applies to lineage fields
    if ids_only and field not in LINEAGE_FIELDS:
        raise click.ClickException(
            f"--ids-only option only applies to lineage fields (upstream, downstream, lineage), not '{field}'"
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

        # Handle lineage fields (upstream, downstream, lineage)
        if field in LINEAGE_FIELDS:
            _handle_lineage_field([exp.id], field, depth, ids_only, fmt)
            return

        value, found = resolve_field_value(exp, field, default_value, tail, head)

        # Determine getter type and output using unified handler
        getter_type = get_getter_type(field, value)
        output_handler = GetterOutput(field, fmt)
        output_handler.output([(exp.id, value)], getter_type)

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

    # Handle lineage fields with multiple targets
    if field in LINEAGE_FIELDS:
        target_ids = [exp.id for exp in experiments]
        _handle_lineage_field(target_ids, field, depth, ids_only, fmt)
        return

    # Collect all values
    results = []
    for exp in experiments:
        value, found = resolve_field_value(exp, field, default_value, tail, head)
        results.append((exp.id, value))

    # Determine getter type using first non-None value
    first_value = next((v for _, v in results if v is not None), None)
    getter_type = get_getter_type(field, first_value)

    # Output using unified handler
    output_handler = GetterOutput(field, fmt)
    output_handler.output(results, getter_type)
