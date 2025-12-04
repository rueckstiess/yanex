"""
Compare experiments - interactive table with parameters and metrics.
"""

import click

from ...core.comparison import ExperimentComparisonData
from ...ui.compare_table import run_comparison_table
from ..filters import ExperimentFilter, parse_time_spec
from ..filters.arguments import experiment_filter_options
from ..formatters import (
    OutputFormat,
    echo_format_info,
    format_csv,
    format_json,
    format_markdown_table,
    format_options,
    format_verbose,
    resolve_output_format,
)
from .confirm import find_experiments_by_filters, find_experiments_by_identifiers


@click.command("compare")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(
    include_ids=False, include_archived=True, include_limit=False
)
@click.option(
    "--params",
    help="Show only specified parameters (comma-separated, e.g., 'learning_rate,epochs')",
)
@click.option(
    "--metrics",
    help="Show only specified metrics (comma-separated, e.g., 'accuracy,loss')",
)
@click.option(
    "--only-different",
    is_flag=True,
    help="Show only columns where values differ between experiments",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="Print static table instead of interactive view",
)
@click.option(
    "--max-rows",
    type=int,
    help="Limit number of experiments displayed",
)
@click.pass_context
def compare_experiments(
    ctx,
    experiment_identifiers: tuple,
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    script_pattern: str | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    archived: bool,
    params: str | None,
    metrics: str | None,
    only_different: bool,
    no_interactive: bool,
    max_rows: int | None,
):
    """
    Compare experiments in an interactive table showing parameters and metrics.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Supports multiple output formats:

    \b
      --format json      Output as JSON (for scripting/AI processing)
      --format csv       Output as CSV (pipe to file: --format csv > results.csv)
      --format markdown  Output as GitHub-flavored markdown

    The interactive table supports:
    - Navigation with arrow keys or hjkl
    - Sorting by any column (s/S for asc/desc, 1/2 for numeric)
    - Export to CSV (press 'e' in interactive mode)
    - Help (press '?' for keyboard shortcuts)

    Examples:

    \b
        yanex compare                                    # All experiments
        yanex compare exp1 exp2 exp3                    # Specific experiments
        yanex compare -s completed                      # Completed experiments
        yanex compare -t training --only-different      # Training experiments, show differences only
        yanex compare --params learning_rate,epochs    # Show only specified parameters
        yanex compare --format csv > results.csv       # Export to CSV file
        yanex compare --format json                     # Export as JSON
        yanex compare --no-interactive                  # Static table output
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
    try:
        filter_obj = ExperimentFilter()

        # Validate mutually exclusive targeting
        has_identifiers = len(experiment_identifiers) > 0
        # Note: name_pattern="" is a valid filter for unnamed experiments
        has_filters = any(
            [
                status,
                name_pattern is not None,
                tags,
                started_after,
                started_before,
                ended_after,
                ended_before,
            ]
        )

        if has_identifiers and has_filters:
            click.echo(
                "Error: Cannot use both experiment identifiers and filter options. Choose one approach.",
                err=True,
            )
            ctx.exit(1)

        if not has_identifiers and not has_filters:
            # No filters specified - use all experiments
            pass

        # Parse time specifications
        started_after_dt = parse_time_spec(started_after) if started_after else None
        started_before_dt = parse_time_spec(started_before) if started_before else None
        ended_after_dt = parse_time_spec(ended_after) if ended_after else None
        ended_before_dt = parse_time_spec(ended_before) if ended_before else None

        # Find experiments to compare
        if experiment_identifiers:
            # Compare specific experiments by ID/name
            experiments = find_experiments_by_identifiers(
                filter_obj, list(experiment_identifiers), archived=archived
            )
        else:
            # Compare experiments by filter criteria
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
            )

        if not experiments:
            location = "archived" if archived else "regular"
            click.echo(f"No {location} experiments found to compare.")
            return

        # Limit number of experiments if requested
        if max_rows and len(experiments) > max_rows:
            experiments = experiments[:max_rows]
            click.echo(f"Limiting display to first {max_rows} experiments.")

        # Extract experiment IDs from the experiment metadata dictionaries
        experiment_ids = [exp["id"] for exp in experiments]

        # Parse parameter and metric lists
        param_list = None
        if params:
            param_list = [p.strip() for p in params.split(",") if p.strip()]

        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

        # Get comparison data
        comparison_data_extractor = ExperimentComparisonData()
        comparison_data = comparison_data_extractor.get_comparison_data(
            experiment_ids=experiment_ids,
            params=param_list,
            metrics=metric_list,
            only_different=only_different,
        )

        if not comparison_data.get("rows"):
            echo_format_info("No comparison data available.", fmt)
            return

        # Handle output formats
        if fmt == OutputFormat.JSON:
            _output_comparison_json(comparison_data)
            return

        if fmt == OutputFormat.CSV:
            _output_comparison_csv(comparison_data)
            return

        if fmt == OutputFormat.MARKDOWN:
            _output_comparison_markdown(comparison_data)
            return

        # Handle static table mode
        if no_interactive:
            _print_static_table(comparison_data)
            return

        # Run interactive table
        total_experiments = comparison_data.get("total_experiments", 0)
        param_count = len(comparison_data.get("param_columns", []))
        metric_count = len(comparison_data.get("metric_columns", []))

        title = f"yanex compare - {total_experiments} experiments, {param_count} params, {metric_count} metrics"

        run_comparison_table(
            comparison_data, title=title, export_path="yanex_comparison.csv"
        )

    except click.ClickException:
        raise  # Re-raise ClickException to show proper error message
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _output_comparison_json(comparison_data: dict) -> None:
    """Output comparison data as JSON to stdout."""
    rows = comparison_data.get("rows", [])

    # Convert rows to clean format (remove prefixes from keys)
    clean_rows = []
    for row in rows:
        clean_row = {}
        for key, value in row.items():
            if key.startswith("param:"):
                clean_row[f"param_{key[6:]}"] = value
            elif key.startswith("metric:"):
                clean_row[f"metric_{key[7:]}"] = value
            else:
                clean_row[key] = value
        clean_rows.append(clean_row)

    json_output = format_json(
        {
            "experiments": clean_rows,
            "total_experiments": comparison_data.get("total_experiments", len(rows)),
            "param_columns": comparison_data.get("param_columns", []),
            "metric_columns": comparison_data.get("metric_columns", []),
        }
    )
    click.echo(json_output)


def _output_comparison_csv(comparison_data: dict) -> None:
    """Output comparison data as CSV to stdout."""
    rows = comparison_data.get("rows", [])
    if not rows:
        return

    # Get column order and generate clean headers
    first_row = rows[0]
    column_keys = list(first_row.keys())

    headers = {}
    for key in column_keys:
        if key.startswith("param:"):
            headers[key] = f"param_{key[6:]}"
        elif key.startswith("metric:"):
            headers[key] = f"metric_{key[7:]}"
        else:
            headers[key] = key

    csv_output = format_csv(rows, columns=column_keys, headers=headers)
    click.echo(csv_output, nl=False)


def _output_comparison_markdown(comparison_data: dict) -> None:
    """Output comparison data as markdown table to stdout."""
    rows = comparison_data.get("rows", [])
    if not rows:
        return

    # Get column order and generate clean headers
    first_row = rows[0]
    column_keys = list(first_row.keys())

    headers = {}
    for key in column_keys:
        if key.startswith("param:"):
            headers[key] = f"param_{key[6:]}"
        elif key.startswith("metric:"):
            headers[key] = f"metric_{key[7:]}"
        elif key == "id":
            headers[key] = "ID"
        elif key == "name":
            headers[key] = "Name"
        else:
            headers[key] = key.title()

    click.echo(format_markdown_table(rows, column_keys, headers))


def _print_static_table(
    comparison_data: dict, max_params: int = 5, max_metrics: int = 5
) -> None:
    """Print comparison data as a static table.

    Uses ExperimentTableFormatter for consistent styling across all CLI commands.

    Args:
        comparison_data: Comparison data dict with rows, param_columns, metric_columns
        max_params: Maximum number of parameter columns to display
        max_metrics: Maximum number of metric columns to display
    """
    from rich.console import Console

    from ..formatters import ExperimentTableFormatter

    console = Console()
    rows = comparison_data.get("rows", [])

    if not rows:
        console.print("No data to display")
        return

    # Get param and metric columns from comparison_data
    all_param_columns = comparison_data.get("param_columns", [])
    all_metric_columns = comparison_data.get("metric_columns", [])

    # Limit param and metric columns
    shown_params = all_param_columns[:max_params]
    hidden_params = len(all_param_columns) - len(shown_params)

    shown_metrics = all_metric_columns[:max_metrics]
    hidden_metrics = len(all_metric_columns) - len(shown_metrics)

    # Use shared ExperimentTableFormatter for consistent styling
    # Exclude script, duration, tags, started to save space for params/metrics
    # (compare typically shows experiments from the same script)
    formatter = ExperimentTableFormatter(console)
    table = formatter.format_experiments_table(
        experiments=rows,
        param_columns=shown_params,
        metric_columns=shown_metrics,
        exclude_columns=["script", "duration", "tags", "started"],
    )

    # Print table
    console.print(table)

    # Print summary with hidden column info
    total_params = len(all_param_columns)
    total_metrics = len(all_metric_columns)

    summary_parts = [f"Showing {len(rows)} experiments"]

    if hidden_params > 0 or hidden_metrics > 0:
        shown_parts = []
        if total_params > 0:
            if hidden_params > 0:
                shown_parts.append(f"{len(shown_params)}/{total_params} params")
            else:
                shown_parts.append(f"{total_params} params")
        if total_metrics > 0:
            if hidden_metrics > 0:
                shown_parts.append(f"{len(shown_metrics)}/{total_metrics} metrics")
            else:
                shown_parts.append(f"{total_metrics} metrics")
        summary_parts.append(f"with {', '.join(shown_parts)}")

        # Add hint about viewing more
        hidden_hints = []
        if hidden_params > 0:
            hidden_hints.append(f"+{hidden_params} params")
        if hidden_metrics > 0:
            hidden_hints.append(f"+{hidden_metrics} metrics")
        summary_parts.append(f"({', '.join(hidden_hints)} hidden)")
    else:
        summary_parts.append(f"with {total_params} params, {total_metrics} metrics")

    console.print("\n" + format_verbose(" ".join(summary_parts)))
    if hidden_params > 0 or hidden_metrics > 0:
        console.print(
            format_verbose(
                "Use --params/--metrics to select specific columns, or --csv/--json for full data"
            )
        )
