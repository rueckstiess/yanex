"""
Show detailed information about a specific experiment.
"""

from typing import Any, Dict, List, Optional

import click

from yanex.cli.filters import ExperimentFilter
from yanex.cli.formatters.console import ExperimentTableFormatter
from yanex.core.manager import ExperimentManager


@click.command("show")
@click.argument("experiment_identifier", required=True)
@click.option(
    "--show-metric",
    "show_metrics",
    help="Comma-separated list of specific metrics to show in results table (e.g., 'accuracy,loss,f1_score')",
)
@click.option("--archived", is_flag=True, help="Include archived experiments in search")
@click.pass_context
def show_experiment(
    ctx, experiment_identifier: str, show_metrics: Optional[str], archived: bool
):
    """
    Show detailed information about an experiment.

    EXPERIMENT_IDENTIFIER can be either:
    - An experiment ID (8-character string)
    - An experiment name

    If multiple experiments have the same name, a list will be shown
    and you'll need to use the unique experiment ID instead.
    """
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

        # Parse show_metrics if provided
        requested_metrics = None
        if show_metrics:
            requested_metrics = [
                metric.strip() for metric in show_metrics.split(",") if metric.strip()
            ]

        # Display detailed experiment information
        display_experiment_details(
            filter_obj.manager, experiment, formatter, requested_metrics, archived
        )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def find_experiment(
    filter_obj: ExperimentFilter, identifier: str, include_archived: bool = False
) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
    """
    Find experiment by ID or name.

    Args:
        filter_obj: ExperimentFilter instance
        identifier: Experiment ID or name
        include_archived: Whether to search archived experiments

    Returns:
        - Single experiment dict if found by ID or unique name
        - List of experiments if multiple names match
        - None if not found
    """
    # First, try to find by exact ID match
    try:
        all_experiments = filter_obj._load_all_experiments(include_archived)

        # Try ID match first (exact 8-character match)
        if len(identifier) == 8:
            for exp in all_experiments:
                if exp.get("id") == identifier:
                    return exp

        # Try name match
        name_matches = []
        for exp in all_experiments:
            exp_name = exp.get("name")
            if exp_name and exp_name == identifier:
                name_matches.append(exp)

        # Return based on name matches
        if len(name_matches) == 1:
            return name_matches[0]
        elif len(name_matches) > 1:
            return name_matches  # Multiple matches - let caller handle

        # No matches found
        return None

    except Exception:
        return None


def display_experiment_details(
    manager: ExperimentManager,
    experiment: Dict[str, Any],
    formatter: ExperimentTableFormatter,
    requested_metrics: Optional[List[str]] = None,
    include_archived: bool = False,
):
    """Display comprehensive experiment details."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    experiment_id = experiment["id"]

    # Header with experiment overview
    status = experiment.get("status", "unknown")
    status_color = formatter.STATUS_COLORS.get(status, "white")
    status_emoji = formatter.STATUS_SYMBOLS.get(status, "○")

    header_text = Text()
    header_text.append("Experiment: ", style="bold")
    header_text.append(f"{experiment.get('name', '[unnamed]')} ", style="bold cyan")
    header_text.append(f"({experiment_id})", style="dim")
    header_text.append(f"\nStatus: {status_emoji} ", style="")
    header_text.append(f"{status}", style=f"bold {status_color}")

    # Add directory path
    try:
        exp_dir = manager.storage.get_experiment_dir(experiment_id, include_archived)
        header_text.append(f"\nDirectory: {exp_dir}", style="dim cyan")
    except Exception:
        pass  # Skip directory path if not available

    # Add timing information
    created_at = experiment.get("created_at")
    started_at = experiment.get("started_at")
    completed_at = experiment.get("completed_at")
    failed_at = experiment.get("failed_at")
    cancelled_at = experiment.get("cancelled_at")

    if created_at:
        header_text.append(f"\nCreated: {formatter._format_time(created_at)}")
    if started_at:
        header_text.append(f"\nStarted: {formatter._format_time(started_at)}")

    # Show end time based on status
    end_time = completed_at or failed_at or cancelled_at
    if end_time:
        end_label = (
            "Completed" if completed_at else ("Failed" if failed_at else "Cancelled")
        )
        header_text.append(f"\n{end_label}: {formatter._format_time(end_time)}")

        # Calculate and show duration
        if started_at:
            duration = formatter._calculate_duration(started_at, end_time)
            header_text.append(f"\nDuration: {duration}")
    elif started_at:
        # Still running
        duration = formatter._calculate_duration(started_at, None)
        header_text.append(f"\nDuration: {duration}")

    console.print(Panel(header_text, box=box.ROUNDED, border_style=status_color))
    console.print()

    # Tags and Description
    tags = experiment.get("tags", [])
    description = experiment.get("description")

    if tags or description:
        info_table = Table.grid(padding=(0, 2))
        info_table.add_column("Field", style="bold")
        info_table.add_column("Value")

        if tags:
            tags_text = ", ".join(tags) if tags else "-"
            info_table.add_row("Tags:", tags_text)

        if description:
            info_table.add_row("Description:", description)

        console.print(
            Panel(info_table, title="[bold]Experiment Info[/bold]", box=box.ROUNDED)
        )
        console.print()

    # Configuration
    try:
        config = manager.storage.load_config(experiment_id, include_archived)
        if config:
            config_table = Table(
                show_header=True, header_style="bold magenta", box=box.SIMPLE
            )
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="green")

            for key, value in config.items():
                # Format value for display
                if isinstance(value, (dict, list)):
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                else:
                    value_str = str(value)

                config_table.add_row(key, value_str)

            console.print(
                Panel(config_table, title="[bold]Configuration[/bold]", box=box.ROUNDED)
            )
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
                    warning_text = Text("Warning: ", style="bold yellow")
                    warning_text.append(
                        f"Requested metrics not found: {', '.join(missing_metrics)}",
                        style="yellow",
                    )
                    console.print(warning_text)

                if not shown_metrics:
                    console.print(
                        Text(
                            "No requested metrics found in experiment results.",
                            style="red",
                        )
                    )
                    return

                results_table = Table(
                    show_header=True, header_style="bold magenta", box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style="cyan")
                results_table.add_column("Timestamp", style="dim")

                for metric in shown_metrics:
                    results_table.add_column(metric, justify="right", style="green")

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

                title = f"[bold]Results[/bold] (showing {len(shown_metrics)} of {len(all_metrics)} metrics)"
                if len(results) > 10:
                    title += f" (last 10 of {len(results)} steps)"

                console.print(Panel(results_table, title=title, box=box.ROUNDED))
                console.print()

            elif len(all_metrics) > 8:
                # Too many metrics - show summary table with key metrics and count
                results_table = Table(
                    show_header=True, header_style="bold magenta", box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style="cyan")
                results_table.add_column("Timestamp", style="dim")

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

                # Add columns for shown metrics
                for metric in shown_metrics:
                    results_table.add_column(metric, justify="right", style="green")

                # Add other metrics column to show count
                other_count = len(all_metrics) - len(shown_metrics)
                if other_count > 0:
                    results_table.add_column(
                        f"(+{other_count} more)", justify="center", style="dim"
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

                title = f"[bold]Results[/bold] ({len(all_metrics)} metrics total)"
                if len(results) > 10:
                    title += f" (showing last 10 of {len(results)} steps)"

                console.print(Panel(results_table, title=title, box=box.ROUNDED))

                # Show all metrics list below table for reference
                metrics_text = Text("All metrics: ", style="bold")
                metrics_text.append(", ".join(all_metrics), style="dim")
                console.print(metrics_text)
                console.print()

            else:
                # Few metrics - show normal table
                results_table = Table(
                    show_header=True, header_style="bold magenta", box=box.SIMPLE
                )
                results_table.add_column("Step", justify="right", style="cyan")
                results_table.add_column("Timestamp", style="dim")

                # Add columns for each metric
                for metric in all_metrics:
                    results_table.add_column(metric, justify="right", style="green")

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

                title = "[bold]Results[/bold]"
                if len(results) > 10:
                    title += f" (showing last 10 of {len(results)})"

                console.print(Panel(results_table, title=title, box=box.ROUNDED))
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
                artifacts_table = Table(
                    show_header=True, header_style="bold magenta", box=box.SIMPLE
                )
                artifacts_table.add_column("Artifact", style="cyan")
                artifacts_table.add_column("Size", justify="right", style="green")
                artifacts_table.add_column("Modified", style="dim")

                for artifact_path in sorted(artifacts):
                    if artifact_path.is_file():
                        size = artifact_path.stat().st_size
                        size_str = formatter._format_file_size(size)
                        mtime = artifact_path.stat().st_mtime
                        mtime_str = formatter._format_timestamp(mtime)

                        artifacts_table.add_row(artifact_path.name, size_str, mtime_str)

                console.print(
                    Panel(
                        artifacts_table, title="[bold]Artifacts[/bold]", box=box.ROUNDED
                    )
                )
                console.print()
    except Exception:
        pass  # Skip artifacts if not available

    # Environment and execution info
    try:
        metadata = manager.storage.load_metadata(experiment_id, include_archived)
        env_info = metadata.get("environment", {})
        git_info = metadata.get("git", {})

        if env_info or git_info:
            env_table = Table.grid(padding=(0, 2))
            env_table.add_column("Field", style="bold")
            env_table.add_column("Value")

            # Git information
            if git_info:
                branch = git_info.get("branch", "unknown")
                commit_hash = git_info.get(
                    "commit_hash_short", git_info.get("commit_hash", "unknown")
                )
                if commit_hash != "unknown" and len(commit_hash) > 12:
                    commit_hash = commit_hash[:12]

                env_table.add_row("Git Branch:", branch)
                env_table.add_row("Git Commit:", commit_hash)

                # Check for uncommitted changes from environment git info
                env_git_info = env_info.get("git", {})
                if env_git_info.get("has_uncommitted_changes"):
                    env_table.add_row("", "[yellow]⚠ Uncommitted changes[/yellow]")

            # Python version information
            python_info = env_info.get("python", {})
            if python_info:
                python_version = python_info.get("python_version", "unknown")
                # Extract just the version number for cleaner display
                if python_version != "unknown" and "(" in python_version:
                    python_version = python_version.split(" (")[
                        0
                    ]  # e.g., "3.11.9" from "3.11.9 (main, ...)"
                env_table.add_row("Python:", python_version)

                # Platform from python info (more readable than system platform)
                python_platform = python_info.get("platform", "unknown")
                if python_platform != "unknown":
                    env_table.add_row("Platform:", python_platform)

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
                        env_table.add_row("Platform:", platform_display)

            # Script information
            script_path = metadata.get("script_path")
            if script_path:
                env_table.add_row("Script:", script_path)

            console.print(
                Panel(env_table, title="[bold]Environment[/bold]", box=box.ROUNDED)
            )
            console.print()
    except Exception:
        pass  # Skip environment if not available

    # Error information if failed
    if status in ["failed", "cancelled"]:
        error_msg = experiment.get("error_message")
        cancel_reason = experiment.get("cancellation_reason")

        if error_msg or cancel_reason:
            error_text = error_msg or cancel_reason
            console.print(
                Panel(
                    Text(error_text, style="red"),
                    title=f"[bold red]{'Error' if error_msg else 'Cancellation Reason'}[/bold red]",
                    box=box.ROUNDED,
                    border_style="red",
                )
            )
