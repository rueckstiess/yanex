"""
Show detailed information about a specific experiment.
"""

from typing import Any

import click

from yanex.cli.filters import ExperimentFilter
from yanex.cli.formatters import (
    ERROR_STYLE,
    ID_STYLE,
    LABEL_STYLE,
    METRICS_STYLE,
    NAME_STYLE,
    PARAMS_STYLE,
    SCRIPT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    STEP_STYLE,
    TIMESTAMP_STYLE,
    WARNING_STYLE,
    WARNING_SYMBOL,
    OutputFormat,
    experiment_to_dict,
    format_options,
    output_json,
    resolve_output_format,
)
from yanex.cli.formatters.console import ExperimentTableFormatter
from yanex.core.access_resolver import AccessResolver, parse_canonical_key
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import AmbiguousKeyError, KeyNotFoundError

from .confirm import find_experiment


def _resolve_metrics_for_experiment(
    manager: ExperimentManager,
    experiment_id: str,
    requested_metrics: list[str],
    include_archived: bool = False,
) -> list[str]:
    """Resolve metric names using AccessResolver for sub-path resolution and patterns.

    Args:
        manager: ExperimentManager instance
        experiment_id: The experiment ID
        requested_metrics: List of metric names/patterns to resolve
        include_archived: Whether to include archived experiments

    Returns:
        List of resolved metric names (paths without group prefix)

    Raises:
        click.ClickException: If resolution fails
    """
    # Load experiment results to build resolver
    try:
        results = manager.storage.load_results(experiment_id, include_archived)
    except Exception:
        results = []

    # Build metrics dict from results
    all_metrics: dict = {}
    if results:
        for result in results:
            for key, value in result.items():
                if key not in ["step", "timestamp"] and key not in all_metrics:
                    all_metrics[key] = value

    if not all_metrics:
        # No metrics available, return original list to let display handle warnings
        return requested_metrics

    # Build resolver with metrics only (we only care about metric scope)
    resolver = AccessResolver(params={}, metrics=all_metrics, meta={})

    try:
        # Resolve each metric (handles sub-path resolution and patterns)
        canonical_keys = resolver.resolve_list(requested_metrics, scope="metric")

        # Strip group prefixes to get paths
        paths = []
        for key in canonical_keys:
            _, path = parse_canonical_key(key)
            paths.append(path)

        return paths

    except AmbiguousKeyError as e:
        matches_str = ", ".join(e.matches[:5])
        if len(e.matches) > 5:
            matches_str += f", ... ({len(e.matches)} total)"
        raise click.ClickException(
            f"Ambiguous metric '{e.key}' matches multiple: {matches_str}\n"
            f"Use a more specific path or the full metric name"
        )
    except KeyNotFoundError:
        # Return original list and let display function handle warnings
        # This provides better UX by showing which metrics exist
        return requested_metrics


@click.command("show")
@click.argument("experiment_identifier", required=True)
@format_options()
@click.option(
    "--metrics",
    "metrics",
    help="Comma-separated list of specific metrics to show in results table (e.g., 'accuracy,loss,f1_score')",
)
@click.option("--archived", is_flag=True, help="Include archived experiments in search")
@click.pass_context
def show_experiment(
    ctx,
    experiment_identifier: str,
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
    metrics: str | None,
    archived: bool,
):
    """
    Show detailed information about an experiment.

    EXPERIMENT_IDENTIFIER can be:
    - An experiment ID (8-character string)
    - A prefix of an experiment ID
    - An experiment name

    Supports multiple output formats:

    \b
      --format json      Output as JSON (for scripting/AI processing)
      --format csv       Output as CSV (for data analysis)
      --format markdown  Output as GitHub-flavored markdown

    If multiple experiments have the same name, a list will be shown
    and you'll need to use the unique experiment ID instead.
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)

    try:
        # Create filter and formatter (filter creates default manager)
        filter_obj = ExperimentFilter()
        formatter = ExperimentTableFormatter()

        # Try to find the experiment
        experiment = find_experiment(filter_obj, experiment_identifier, archived)

        if experiment is None:
            click.echo(
                f"Error: No experiment found with ID or name '{experiment_identifier}'",
                err=True,
            )
            ctx.exit(1)

        # Check if we got multiple experiments (name collision)
        if isinstance(experiment, list):
            click.echo(
                f"Multiple experiments found with name '{experiment_identifier}':"
            )
            click.echo()

            # Show a filtered list using the existing list formatter
            formatter.print_experiments_table(experiment)
            click.echo()
            click.echo(
                "Please use the specific experiment ID with 'yanex show <id>' to view details."
            )
            ctx.exit(1)

        # Parse metrics if provided
        requested_metrics = None
        if metrics:
            requested_metrics = [
                metric.strip() for metric in metrics.split(",") if metric.strip()
            ]
            # Resolve metric names (sub-path resolution and patterns)
            requested_metrics = _resolve_metrics_for_experiment(
                filter_obj.manager, experiment["id"], requested_metrics, archived
            )

        # Handle different output formats
        if fmt == OutputFormat.JSON:
            output_experiment_json(filter_obj.manager, experiment, archived)
            return
        elif fmt == OutputFormat.CSV:
            output_experiment_csv(filter_obj.manager, experiment, archived)
            return
        elif fmt == OutputFormat.MARKDOWN:
            output_experiment_markdown(
                filter_obj.manager, experiment, requested_metrics, archived
            )
            return

        # Default: Rich console output
        display_experiment_details(
            filter_obj.manager, experiment, formatter, requested_metrics, archived
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _print_section_header(console, title: str) -> None:
    """Print a simple section header without panel borders."""
    from rich.rule import Rule

    console.print(Rule(title, style=TIMESTAMP_STYLE, align="center"))


def display_experiment_details(
    manager: ExperimentManager,
    experiment: dict[str, Any],
    formatter: ExperimentTableFormatter,
    requested_metrics: list[str] | None = None,
    include_archived: bool = False,
):
    """Display comprehensive experiment details."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    experiment_id = experiment["id"]

    # Add newline at the beginning for visual separation
    console.print()

    # Header with experiment overview (no panel, just styled text)
    status = experiment.get("status", "unknown")
    status_color = STATUS_COLORS.get(status, "white")
    status_emoji = STATUS_SYMBOLS.get(status, "○")

    # First line: Experiment name and ID
    header_text = Text()
    header_text.append("Experiment: ", style=LABEL_STYLE)
    header_text.append(
        f"{experiment.get('name', '[unnamed]')} ", style=f"bold {NAME_STYLE}"
    )
    header_text.append(f"({experiment_id})", style=ID_STYLE)
    console.print(header_text)

    # Build metadata lines with aligned labels (right-padded to 11 chars)
    def print_field(label: str, value: str, style: str = "") -> None:
        padded_label = f"{label:>11}: "
        line = Text()
        line.append(padded_label, style=LABEL_STYLE)
        line.append(value, style=style)
        console.print(line)

    # Status
    print_field("Status", f"{status_emoji} {status}", f"bold {status_color}")

    # Directory path
    try:
        exp_dir = manager.storage.get_experiment_dir(experiment_id, include_archived)
        print_field("Directory", str(exp_dir), SCRIPT_STYLE)
    except Exception:
        pass  # Skip directory path if not available

    # Timing information
    created_at = experiment.get("created_at")
    started_at = experiment.get("started_at")
    completed_at = experiment.get("completed_at")
    failed_at = experiment.get("failed_at")
    cancelled_at = experiment.get("cancelled_at")

    if created_at:
        print_field("Created", formatter._format_time(created_at))
    if started_at:
        print_field("Started", formatter._format_time(started_at))

    # Show end time based on status
    end_time = completed_at or failed_at or cancelled_at
    if end_time:
        end_label = (
            "Completed" if completed_at else ("Failed" if failed_at else "Cancelled")
        )
        print_field(end_label, formatter._format_time(end_time))

        # Calculate and show duration
        if started_at:
            duration = formatter._calculate_duration(started_at, end_time)
            print_field("Duration", duration)
    elif started_at:
        # Still running
        duration = formatter._calculate_duration(started_at, None)
        print_field("Duration", duration)

    console.print()

    # Tags and Description (inline, with aligned labels)
    tags = experiment.get("tags", [])
    description = experiment.get("description")

    if tags:
        tags_text = ", ".join(tags)
        print_field("Tags", tags_text)

    if description:
        print_field("Description", description)

    if tags or description:
        console.print()

    # Configuration
    try:
        from yanex.utils.dict_utils import flatten_dict

        config = manager.storage.load_config(experiment_id, include_archived)
        if config:
            # Flatten nested configuration for better readability
            flat_config = flatten_dict(config)

            _print_section_header(console, "Parameters")

            config_table = Table(
                show_header=True, header_style=LABEL_STYLE, box=box.SIMPLE
            )
            config_table.add_column("Parameter", style=PARAMS_STYLE)
            config_table.add_column("Value", style=METRICS_STYLE)

            for key, value in sorted(flat_config.items()):
                # Format value for display
                if isinstance(value, list):
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                else:
                    value_str = str(value)

                config_table.add_row(key, value_str)

            console.print(config_table)
            console.print()
    except Exception:
        pass  # Skip config if not available

    # Results
    try:
        results = manager.storage.load_results(experiment_id, include_archived)
        if results:
            # Get all unique metric names
            all_metrics = set()
            for result in results:
                for key in result.keys():
                    if key not in ["step", "timestamp"]:
                        all_metrics.add(key)

            all_metrics = sorted(all_metrics)

            # Determine which metrics to show
            if requested_metrics:
                # User specified specific metrics
                shown_metrics = []
                missing_metrics = []
                for metric in requested_metrics:
                    if metric in all_metrics:
                        shown_metrics.append(metric)
                    else:
                        missing_metrics.append(metric)

                # Warn about missing metrics
                if missing_metrics:
                    warning_text = Text(
                        f"{WARNING_SYMBOL} Warning: ", style=f"bold {WARNING_STYLE}"
                    )
                    warning_text.append(
                        f"Requested metrics not found: {', '.join(missing_metrics)}",
                        style=WARNING_STYLE,
                    )
                    console.print(warning_text)

                if not shown_metrics:
                    console.print(
                        Text(
                            "No requested metrics found in experiment results.",
                            style=ERROR_STYLE,
                        )
                    )
                    return

                # Build section header with info
                title = f"Metrics ({len(shown_metrics)} of {len(all_metrics)} total)"
                if len(results) > 10:
                    title += f" - last 10 of {len(results)} steps"
                _print_section_header(console, title)

                results_table = Table(
                    show_header=True, header_style=LABEL_STYLE, box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style=STEP_STYLE)
                results_table.add_column("Timestamp", style=TIMESTAMP_STYLE)

                for metric in shown_metrics:
                    results_table.add_column(
                        metric, justify="right", style=METRICS_STYLE
                    )

                # Add rows
                for result in results[-10:]:  # Show last 10 results
                    row = [
                        str(result.get("step", "-")),
                        formatter._format_time(result.get("timestamp", "")),
                    ]

                    for metric in shown_metrics:
                        value = result.get(metric)
                        if value is not None:
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append("-")

                    results_table.add_row(*row)

                console.print(results_table)
                console.print()

            elif len(all_metrics) > 8:
                # Too many metrics - show summary table with key metrics and count
                # Show key metrics first, then fill up to 8 total metrics
                key_metrics = [
                    "accuracy",
                    "loss",
                    "epoch",
                    "learning_rate",
                    "f1_score",
                    "precision",
                    "recall",
                ]
                shown_metrics = []

                # Add key metrics that exist
                for metric in key_metrics:
                    if metric in all_metrics and len(shown_metrics) < 8:
                        shown_metrics.append(metric)

                # Fill remaining slots with other metrics (alphabetically)
                remaining_metrics = [m for m in all_metrics if m not in shown_metrics]
                for metric in remaining_metrics:
                    if len(shown_metrics) < 8:
                        shown_metrics.append(metric)
                    else:
                        break

                other_count = len(all_metrics) - len(shown_metrics)

                # Build section header with info
                title = f"Metrics ({len(all_metrics)} total)"
                if len(results) > 10:
                    title += f" - last 10 of {len(results)} steps"
                _print_section_header(console, title)

                results_table = Table(
                    show_header=True, header_style=LABEL_STYLE, box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style=STEP_STYLE)
                results_table.add_column("Timestamp", style=TIMESTAMP_STYLE)

                # Add columns for shown metrics
                for metric in shown_metrics:
                    results_table.add_column(
                        metric, justify="right", style=METRICS_STYLE
                    )

                # Add other metrics column to show count
                if other_count > 0:
                    results_table.add_column(
                        f"(+{other_count} more)",
                        justify="center",
                        style=TIMESTAMP_STYLE,
                    )

                # Add rows
                for result in results[-10:]:  # Show last 10 results
                    row = [
                        str(result.get("step", "-")),
                        formatter._format_time(result.get("timestamp", "")),
                    ]

                    for metric in shown_metrics:
                        value = result.get(metric)
                        if value is not None:
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append("-")

                    if other_count > 0:
                        row.append("...")

                    results_table.add_row(*row)

                console.print(results_table)

                # Show all metrics list below table for reference
                metrics_text = Text("All metrics: ", style=LABEL_STYLE)
                metrics_text.append(", ".join(all_metrics), style=TIMESTAMP_STYLE)
                console.print(metrics_text)
                console.print()

            else:
                # Few metrics - show normal table
                title = f"Metrics ({len(all_metrics)} total)"
                if len(results) > 10:
                    title += f" - last 10 of {len(results)} steps"
                _print_section_header(console, title)

                results_table = Table(
                    show_header=True, header_style=LABEL_STYLE, box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style=STEP_STYLE)
                results_table.add_column("Timestamp", style=TIMESTAMP_STYLE)

                # Add columns for each metric
                for metric in all_metrics:
                    results_table.add_column(
                        metric, justify="right", style=METRICS_STYLE
                    )

                # Add rows
                for result in results[-10:]:  # Show last 10 results
                    row = [
                        str(result.get("step", "-")),
                        formatter._format_time(result.get("timestamp", "")),
                    ]

                    for metric in all_metrics:
                        value = result.get(metric)
                        if value is not None:
                            # Format numbers nicely
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append("-")

                    results_table.add_row(*row)

                console.print(results_table)
                console.print()
    except Exception:
        pass  # Skip results if not available

    # Artifacts
    try:
        experiment_dir = manager.storage.get_experiment_dir(
            experiment_id, include_archived
        )
        artifacts_dir = experiment_dir / "artifacts"

        if artifacts_dir.exists():
            artifacts = list(artifacts_dir.iterdir())
            if artifacts:
                _print_section_header(console, "Artifacts")

                artifacts_table = Table(
                    show_header=True, header_style=LABEL_STYLE, box=box.SIMPLE
                )
                artifacts_table.add_column("Artifact", style=PARAMS_STYLE)
                artifacts_table.add_column("Size", justify="right", style=METRICS_STYLE)
                artifacts_table.add_column("Modified", style=TIMESTAMP_STYLE)

                for artifact_path in sorted(artifacts):
                    if artifact_path.is_file():
                        size = artifact_path.stat().st_size
                        size_str = formatter._format_file_size(size)
                        mtime = artifact_path.stat().st_mtime
                        mtime_str = formatter._format_timestamp(mtime)

                        artifacts_table.add_row(artifact_path.name, size_str, mtime_str)

                console.print(artifacts_table)
                console.print()
    except Exception:
        pass  # Skip artifacts if not available

    # Environment and execution info
    try:
        metadata = manager.storage.load_metadata(experiment_id, include_archived)
        env_info = metadata.get("environment", {})
        git_info = metadata.get("git", {})

        if env_info or git_info:
            _print_section_header(console, "Environment")

            # Git information
            if git_info:
                branch = git_info.get("branch", "unknown")
                commit_hash = git_info.get(
                    "commit_hash_short", git_info.get("commit_hash", "unknown")
                )
                if commit_hash != "unknown" and len(commit_hash) > 12:
                    commit_hash = commit_hash[:12]

                print_field("Git Branch", branch)
                print_field("Git Commit", commit_hash)

                # Check for uncommitted changes from environment git info
                env_git_info = env_info.get("git", {})
                if env_git_info.get("has_uncommitted_changes"):
                    console.print(
                        f"             [{WARNING_STYLE}]{WARNING_SYMBOL} Uncommitted changes[/{WARNING_STYLE}]"
                    )

            # Python version information
            python_info = env_info.get("python", {})
            if python_info:
                python_version = python_info.get("python_version", "unknown")
                # Extract just the version number for cleaner display
                if python_version != "unknown" and "(" in python_version:
                    python_version = python_version.split(" (")[
                        0
                    ]  # e.g., "3.11.9" from "3.11.9 (main, ...)"
                print_field("Python", python_version)

                # Platform from python info (more readable than system platform)
                python_platform = python_info.get("platform", "unknown")
                if python_platform != "unknown":
                    print_field("Platform", python_platform)

            # System information (fallback if python platform not available)
            if python_info.get("platform") == "unknown" or not python_info:
                system_info = env_info.get("system", {})
                platform_info = system_info.get("platform", {})
                if platform_info:
                    system_name = platform_info.get("system", "unknown")
                    machine = platform_info.get("machine", "")
                    if system_name != "unknown":
                        platform_display = system_name
                        if machine:
                            platform_display += f" ({machine})"
                        print_field("Platform", platform_display)

            # Script information
            script_path = metadata.get("script_path")
            if script_path:
                print_field("Script", script_path)

            console.print()
    except Exception:
        pass  # Skip environment if not available

    # Error information if failed
    if status in ["failed", "cancelled"]:
        error_msg = experiment.get("error_message")
        cancel_reason = experiment.get("cancellation_reason")

        if error_msg or cancel_reason:
            error_text = error_msg or cancel_reason
            label = "Error" if error_msg else "Cancellation Reason"
            _print_section_header(console, label)
            console.print(Text(error_text, style=ERROR_STYLE))
            console.print()


def output_experiment_json(
    manager: ExperimentManager,
    experiment: dict[str, Any],
    include_archived: bool = False,
) -> None:
    """Output experiment details as JSON."""
    experiment_id = experiment["id"]

    # Build comprehensive experiment data
    data = experiment_to_dict(experiment, flatten=True)

    # Add config
    try:
        config = manager.storage.load_config(experiment_id, include_archived)
        if config:
            data["config"] = config
    except Exception:
        pass

    # Add results/metrics
    try:
        results = manager.storage.load_results(experiment_id, include_archived)
        if results:
            data["results"] = results
    except Exception:
        pass

    # Add metadata (environment, git info)
    try:
        metadata = manager.storage.load_metadata(experiment_id, include_archived)
        if metadata:
            data["environment"] = metadata.get("environment", {})
            data["git"] = metadata.get("git", {})
    except Exception:
        pass

    # Add artifacts list
    try:
        exp_dir = manager.storage.get_experiment_dir(experiment_id, include_archived)
        artifacts_dir = exp_dir / "artifacts"
        if artifacts_dir.exists():
            artifacts = []
            for artifact_path in sorted(artifacts_dir.iterdir()):
                if artifact_path.is_file():
                    artifacts.append(
                        {
                            "name": artifact_path.name,
                            "size": artifact_path.stat().st_size,
                            "path": str(artifact_path),
                        }
                    )
            if artifacts:
                data["artifacts"] = artifacts
    except Exception:
        pass

    # Add experiment directory path
    try:
        exp_dir = manager.storage.get_experiment_dir(experiment_id, include_archived)
        data["experiment_dir"] = str(exp_dir)
    except Exception:
        pass

    output_json(data)


def output_experiment_csv(
    manager: ExperimentManager,
    experiment: dict[str, Any],
    include_archived: bool = False,
) -> None:
    """Output experiment details as CSV (single row with key fields)."""
    from yanex.cli.formatters import output_csv

    experiment_id = experiment["id"]

    # Create a single row with key experiment fields
    row = experiment_to_dict(experiment, flatten=True)

    # Add config as flattened params
    try:
        from yanex.utils.dict_utils import flatten_dict

        config = manager.storage.load_config(experiment_id, include_archived)
        if config:
            flat_config = flatten_dict(config)
            for key, value in flat_config.items():
                row[f"param:{key}"] = value
    except Exception:
        pass

    # Add latest metrics
    try:
        results = manager.storage.load_results(experiment_id, include_archived)
        if results:
            latest = results[-1]
            for key, value in latest.items():
                if key not in ["step", "timestamp"]:
                    row[f"metric:{key}"] = value
    except Exception:
        pass

    output_csv([row])


def output_experiment_markdown(
    manager: ExperimentManager,
    experiment: dict[str, Any],
    requested_metrics: list[str] | None = None,
    include_archived: bool = False,
) -> None:
    """Output experiment details as markdown."""
    from yanex.cli.formatters import format_markdown_table

    experiment_id = experiment["id"]
    status = experiment.get("status", "unknown")
    status_emoji = STATUS_SYMBOLS.get(status, "○")

    lines = []

    # Header
    name = experiment.get("name", "[unnamed]")
    lines.append(f"# Experiment: {name}")
    lines.append("")
    lines.append(f"**ID:** `{experiment_id}`")
    lines.append(f"**Status:** {status_emoji} {status}")

    # Timing
    if experiment.get("created_at"):
        lines.append(f"**Created:** {experiment.get('created_at')}")
    if experiment.get("started_at"):
        lines.append(f"**Started:** {experiment.get('started_at')}")

    end_time = (
        experiment.get("completed_at")
        or experiment.get("failed_at")
        or experiment.get("cancelled_at")
    )
    if end_time:
        end_label = (
            "Completed"
            if experiment.get("completed_at")
            else ("Failed" if experiment.get("failed_at") else "Cancelled")
        )
        lines.append(f"**{end_label}:** {end_time}")

    # Tags and description
    tags = experiment.get("tags", [])
    if tags:
        lines.append(f"**Tags:** {', '.join(tags)}")

    description = experiment.get("description")
    if description:
        lines.append(f"**Description:** {description}")

    lines.append("")

    # Configuration
    try:
        from yanex.utils.dict_utils import flatten_dict

        config = manager.storage.load_config(experiment_id, include_archived)
        if config:
            flat_config = flatten_dict(config)
            lines.append("## Configuration")
            lines.append("")
            config_rows = [
                {"Parameter": k, "Value": str(v)}
                for k, v in sorted(flat_config.items())
            ]
            lines.append(format_markdown_table(config_rows, ["Parameter", "Value"]))
            lines.append("")
    except Exception:
        pass

    # Results
    try:
        results = manager.storage.load_results(experiment_id, include_archived)
        if results:
            lines.append("## Results")
            lines.append("")

            # Get all metric names
            all_metrics = set()
            for result in results:
                for key in result.keys():
                    if key not in ["step", "timestamp"]:
                        all_metrics.add(key)
            all_metrics = sorted(all_metrics)

            # Filter by requested metrics if specified
            if requested_metrics:
                shown_metrics = [m for m in requested_metrics if m in all_metrics]
            else:
                shown_metrics = all_metrics[:8]  # Limit to 8 metrics

            columns = ["step", "timestamp"] + shown_metrics
            headers = {"step": "Step", "timestamp": "Timestamp"}
            for m in shown_metrics:
                headers[m] = m

            # Format results for table
            result_rows = []
            for result in results[-10:]:  # Last 10 results
                row = {
                    "step": str(result.get("step", "-")),
                    "timestamp": result.get("timestamp", "-"),
                }
                for m in shown_metrics:
                    value = result.get(m)
                    if value is not None:
                        if isinstance(value, float):
                            row[m] = f"{value:.4f}"
                        else:
                            row[m] = str(value)
                    else:
                        row[m] = "-"
                result_rows.append(row)

            lines.append(format_markdown_table(result_rows, columns, headers))
            lines.append("")
    except Exception:
        pass

    # Artifacts
    try:
        exp_dir = manager.storage.get_experiment_dir(experiment_id, include_archived)
        artifacts_dir = exp_dir / "artifacts"
        if artifacts_dir.exists():
            artifacts = list(artifacts_dir.iterdir())
            if artifacts:
                lines.append("## Artifacts")
                lines.append("")
                artifact_rows = []
                for artifact_path in sorted(artifacts):
                    if artifact_path.is_file():
                        size = artifact_path.stat().st_size
                        artifact_rows.append(
                            {"Name": artifact_path.name, "Size": _format_size(size)}
                        )
                lines.append(format_markdown_table(artifact_rows, ["Name", "Size"]))
                lines.append("")
    except Exception:
        pass

    # Error information
    if status in ["failed", "cancelled"]:
        error_msg = experiment.get("error_message")
        cancel_reason = experiment.get("cancellation_reason")
        if error_msg or cancel_reason:
            lines.append("## Error")
            lines.append("")
            lines.append(f"```\n{error_msg or cancel_reason}\n```")
            lines.append("")

    click.echo("\n".join(lines))


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
