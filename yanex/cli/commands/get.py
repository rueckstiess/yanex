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
from yanex.core.access_resolver import AccessResolver
from yanex.core.dependency_graph import DependencyGraph
from yanex.core.storage import ExperimentStorage
from yanex.utils.dict_utils import get_nested_value
from yanex.utils.exceptions import AmbiguousKeyError, KeyNotFoundError

from .confirm import find_experiment

# Valid field prefixes that accept dynamic suffixes
# New syntax: param:lr, metric:accuracy (colon separator)
# Legacy syntax: params.lr, metrics.accuracy (dot separator with plural prefix)
DYNAMIC_FIELD_PREFIXES = (
    # New canonical syntax (singular with colon)
    "param:",
    "metric:",
    "meta:",
    # Legacy syntax (plural with dot)
    "params.",
    "metrics.",
    "environment.",
)


def _contains_pattern(field: str) -> bool:
    """Check if a field contains glob pattern characters.

    Args:
        field: Field name to check.

    Returns:
        True if field contains *, ?, or [] patterns.
    """
    return any(c in field for c in "*?[]")


def validate_field(field: str) -> None:
    """Validate that a field name is recognized.

    For `yanex get`, patterns are not allowed (use compare for patterns).
    Unqualified fields (like 'lr') are allowed and resolved later via AccessResolver.

    Args:
        field: Field name to validate.

    Raises:
        click.ClickException: If field contains patterns or is invalid.
    """
    # Check for patterns - not allowed in get (single-value context)
    if _contains_pattern(field):
        raise click.ClickException(
            f"Pattern matching is not supported for 'yanex get'.\n\n"
            f"Field '{field}' contains pattern characters (*, ?, []).\n"
            f"Use 'yanex compare --params' or '--metrics' for pattern matching."
        )

    # Check exact match in known fields
    if field in GETTER_TYPES:
        return

    # Check dynamic prefixes (params.*, metrics.*, environment.*, param:*, etc.)
    for prefix in DYNAMIC_FIELD_PREFIXES:
        if field.startswith(prefix):
            return

    # For unqualified fields, check if they look like typos of static GETTER_TYPES.
    # If similar to a static field, reject early with helpful suggestions.
    # If not similar to any static field, allow through for AccessResolver
    # (they might be param/metric short keys like 'lr' -> 'param:model.lr')
    valid_fields = sorted(GETTER_TYPES.keys())
    suggestions = _find_similar_fields(field, valid_fields)

    if suggestions:
        # Field looks like a typo of a static getter - reject with suggestions
        error_msg = f"Unknown field: '{field}'"
        error_msg += f"\n\nDid you mean: {', '.join(suggestions)}?"
        error_msg += "\n\nUse 'yanex get --help' to see available fields."
        error_msg += (
            "\n\nTip: For parameters use 'param:<key>', for metrics use 'metric:<key>'."
        )
        raise click.ClickException(error_msg)

    # No similarity to static fields - allow through for AccessResolver resolution
    # This enables sub-path resolution like 'lr' -> 'param:advisor.lr'
    return


def resolve_field_for_experiment(exp, field: str, include_deps: bool = False) -> str:
    """Resolve an unqualified or partial field name using AccessResolver.

    This enables sub-path resolution like 'lr' -> 'param:advisor.lr'.

    Args:
        exp: Experiment object with params, metrics, and metadata.
        field: Field name to resolve (may be unqualified like 'lr').
        include_deps: If True, include parameters from dependencies when resolving.

    Returns:
        Resolved canonical field name (e.g., 'param:advisor.lr').

    Raises:
        click.ClickException: If field is ambiguous or not found.
    """
    # Skip resolution for known static fields
    if field in GETTER_TYPES:
        return field

    # Legacy prefixes use exact paths - skip resolution entirely
    # The get_field_value function handles these directly
    LEGACY_PREFIXES = ("params.", "metrics.", "environment.")
    for prefix in LEGACY_PREFIXES:
        if field.startswith(prefix):
            return field

    # New canonical prefixes (param:, metric:, meta:) with explicit full paths
    # should be passed through directly. Resolution is only needed for short keys.
    # e.g., 'param:model.lr' should pass through, 'param:lr' needs resolution
    CANONICAL_PREFIXES = ("param:", "metric:", "meta:")
    for prefix in CANONICAL_PREFIXES:
        if field.startswith(prefix):
            path = field[len(prefix) :]
            # If path contains a dot, treat as fully-qualified and pass through
            # This allows get_field_value to handle it directly
            if "." in path:
                return field

    # New canonical prefixes with short paths need sub-path resolution
    # e.g., 'param:lr' should resolve to 'param:advisor.lr' if that's the only match
    # Unqualified fields (like 'lr') also need resolution

    # Build AccessResolver with experiment data
    params = exp.get_params(include_deps=include_deps)
    metrics_data = exp.get_metrics(as_dataframe=False)

    # Extract metric names from the list of dicts
    metrics_dict = {}
    if metrics_data:
        for entry in metrics_data:
            for key, value in entry.items():
                if key not in ("step", "timestamp", "last_updated"):
                    # Use the last value for each metric
                    metrics_dict[key] = value

    metadata = exp._load_metadata()

    resolver = AccessResolver(params=params, metrics=metrics_dict, meta=metadata)

    try:
        # Try to resolve the field
        canonical_key = resolver.resolve(field)
        return canonical_key
    except AmbiguousKeyError as e:
        # Multiple matches - show helpful error
        matches_str = ", ".join(e.matches)
        raise click.ClickException(
            f"Ambiguous field '{field}' matches multiple keys:\n"
            f"  {matches_str}\n\n"
            f"Please specify the full path or use a group prefix (param:, metric:, meta:)."
        )
    except KeyNotFoundError:
        # Not found via resolver - let the original validation logic handle it
        # This preserves the original error message with suggestions
        valid_fields = sorted(GETTER_TYPES.keys())
        suggestions = _find_similar_fields(field, valid_fields)

        error_msg = f"Unknown field: '{field}'"
        if suggestions:
            error_msg += f"\n\nDid you mean: {', '.join(suggestions)}?"
        error_msg += "\n\nUse 'yanex get --help' to see available fields."
        error_msg += (
            "\n\nTip: For parameters use 'param:<key>', for metrics use 'metric:<key>'."
        )

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

# Fields that display lineage graphs (support --depth)
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
    include_deps: bool = False,
) -> tuple[Any, bool]:
    """
    Resolve a field value from an experiment.

    Args:
        exp: Experiment object
        field: Field path (e.g., "status", "params.lr", "metrics.accuracy")
        default_value: Default value for missing fields
        tail: If specified, return last N lines (only for stdout/stderr fields)
        head: If specified, return first N lines (only for stdout/stderr fields)
        include_deps: If True, include parameters from dependencies

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
        params = exp.get_params(include_deps=include_deps)
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

    # Handle param:* fields (new syntax)
    if field.startswith("param:"):
        param_key = field[6:]  # Remove "param:" prefix
        value = exp.get_param(param_key, include_deps=include_deps)
        if value is not None:
            return value, True
        return default_value, False

    # Handle params.* fields (legacy syntax)
    if field.startswith("params."):
        param_key = field[7:]  # Remove "params." prefix
        value = exp.get_param(param_key, include_deps=include_deps)
        if value is not None:
            return value, True
        return default_value, False

    # Handle metric:* fields (new syntax) - get last logged value
    if field.startswith("metric:"):
        metric_name = field[7:]  # Remove "metric:" prefix
        value = exp.get_metric(metric_name)
        if value is not None:
            # If it's a list, return the last value
            if isinstance(value, list) and len(value) > 0:
                return value[-1], True
            return value, True
        return default_value, False

    # Handle metrics.* fields (legacy syntax) - get last logged value
    if field.startswith("metrics."):
        metric_name = field[8:]  # Remove "metrics." prefix
        value = exp.get_metric(metric_name)
        if value is not None:
            # If it's a list, return the last value
            if isinstance(value, list) and len(value) > 0:
                return value[-1], True
            return value, True
        return default_value, False

    # Handle meta:* fields (new syntax)
    if field.startswith("meta:"):
        meta_key = field[5:]  # Remove "meta:" prefix
        # Check top-level experiment attributes
        if meta_key == "id":
            return exp.id, True
        if meta_key == "name":
            return exp.name if exp.name else default_value, exp.name is not None
        if meta_key == "status":
            return exp.status, True
        if meta_key == "description":
            return (
                exp.description if exp.description else default_value,
                exp.description is not None,
            )
        if meta_key == "tags":
            return exp.tags, True
        if meta_key == "script_path":
            return str(exp.script_path) if exp.script_path else default_value, (
                exp.script_path is not None
            )
        # Handle nested paths (e.g., meta:git.branch)
        if meta_key.startswith("git."):
            git_key = meta_key[4:]  # Remove "git." prefix
            metadata = exp._load_metadata()
            git_info = metadata.get("git", {})
            if git_key in git_info:
                return git_info[git_key], True
            return default_value, False
        # Try as timestamp or other metadata field
        metadata = exp._load_metadata()
        value = get_nested_value(metadata, meta_key)
        if value is not None:
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
    fmt: OutputFormat,
) -> None:
    """Handle lineage field output (upstream, downstream, lineage).

    Args:
        exp_ids: List of experiment IDs to get lineage for.
        field: One of "upstream", "downstream", or "lineage".
        depth: Maximum traversal depth.
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
    "--include-deps",
    is_flag=True,
    help="Include parameters from dependencies (only for param: fields)",
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def get_field(
    ctx,
    experiment_id: str | None,
    field: str,
    # Filter options from decorator
    ids: str | None,
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
    include_deps: bool,
):
    """
    Get a specific field value from experiment(s).

    EXPERIMENT_ID can be an experiment ID, name, or ID prefix.
    If omitted, use filter options to select experiments.

    \b
    Available fields (new canonical syntax with group:path):
      param:<key>
          Get parameter value (e.g., param:lr, param:model.size)
      metric:<key>
          Get last logged metric value (e.g., metric:accuracy, metric:train.loss)
      meta:<key>
          Get metadata field (e.g., meta:status, meta:name, meta:git.branch)

    \b
    Metadata fields (accessible via meta:<key> or directly):
      id, name, status, description, tags
          Experiment metadata (single values or list for tags)
      created_at, started_at, completed_at, failed_at, cancelled_at
          Timestamps in ISO format
      script_path, error_message, cancellation_reason
          Script path and failure/cancellation details

    \b
    List fields:
      params
          List available parameter names
      metrics
          List available metric names

    \b
    Legacy syntax (still supported):
      params.<key>
          Get parameter value (e.g., params.lr)
      metrics.<key>
          Get metric value (e.g., metrics.accuracy)

    \b
    Special fields:
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
      yanex get meta:status abc123         Experiment status (new syntax)
      yanex get param:lr abc123            Parameter value (new syntax)
      yanex get metric:accuracy abc123     Last logged metric (new syntax)
      yanex get status abc123              Metadata field (direct access)
      yanex get params.lr abc123           Parameter (legacy syntax)
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
      yanex get lineage abc123 -F sweep    Get IDs only (for scripting)
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
      yanex get param:lr -t baseline       Learning rates from tagged exps
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

    # Validate --include-deps only applies to param fields
    is_param_field = (
        field.startswith("param:") or field.startswith("params.") or field == "params"
    )
    if include_deps and not is_param_field:
        raise click.ClickException(
            f"--include-deps option only applies to param fields (param:*, params.*, params), not '{field}'"
        )

    # Parse comma-separated IDs into a list
    ids_list = None
    if ids:
        ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

    # Determine if we have filters or a single experiment
    # Note: name_pattern="" is a valid filter for unnamed experiments
    has_filters = any(
        [
            ids_list,
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
            _handle_lineage_field([exp.id], field, depth, fmt)
            return

        # Resolve field name using AccessResolver (enables sub-path resolution)
        resolved_field = resolve_field_for_experiment(exp, field, include_deps)

        value, found = resolve_field_value(
            exp, resolved_field, default_value, tail, head, include_deps
        )

        # Determine getter type and output using unified handler
        getter_type = get_getter_type(field, value)
        output_handler = GetterOutput(field, fmt)
        output_handler.output([(exp.id, value)], getter_type)

        return

    # Multi-experiment mode with filters
    # Build filter kwargs
    filter_kwargs = {}
    if ids_list:
        filter_kwargs["ids"] = ids_list
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
        _handle_lineage_field(target_ids, field, depth, fmt)
        return

    # Resolve field name using AccessResolver (use first experiment for resolution)
    # The resolved field should be the same across all experiments
    resolved_field = resolve_field_for_experiment(experiments[0], field, include_deps)

    # Collect all values
    results = []
    for exp in experiments:
        value, found = resolve_field_value(
            exp, resolved_field, default_value, tail, head, include_deps
        )
        results.append((exp.id, value))

    # Determine getter type using first non-None value
    first_value = next((v for _, v in results if v is not None), None)
    getter_type = get_getter_type(field, first_value)

    # Output using unified handler
    output_handler = GetterOutput(field, fmt)
    output_handler.output(results, getter_type)
