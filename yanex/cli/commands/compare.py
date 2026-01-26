"""
Compare experiments - interactive table with parameters and metrics.
"""

import click

from ...core.access_resolver import AccessResolver, parse_canonical_key
from ...core.comparison import ExperimentComparisonData
from ...ui.compare_table import run_comparison_table
from ...utils.exceptions import AmbiguousKeyError, KeyNotFoundError
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


def _build_resolver_from_experiments(
    comparison_extractor: ExperimentComparisonData,
    experiment_ids: list[str],
    include_archived: bool = False,
    include_dep_params: bool = False,
) -> AccessResolver | None:
    """Build an AccessResolver from experiment data.

    Aggregates params, metrics, and metadata across all experiments to build
    a comprehensive resolver that can resolve any key present in any experiment.

    Args:
        comparison_extractor: ExperimentComparisonData instance
        experiment_ids: List of experiment IDs
        include_archived: Whether to include archived experiments
        include_dep_params: Whether to include parameters from dependencies

    Returns:
        AccessResolver instance or None if no experiments could be loaded
    """
    from ...utils.dict_utils import flatten_dict

    # Aggregate data from all experiments
    all_params: dict = {}
    all_metrics: dict = {}
    all_meta: dict = {}

    for exp_id in experiment_ids:
        try:
            exp_data = comparison_extractor._extract_single_experiment(
                exp_id, include_archived, include_dep_params=include_dep_params
            )
            if not exp_data:
                continue

            # Aggregate params (config)
            config = exp_data.get("config", {})
            flat_config = flatten_dict(config)
            for key, value in flat_config.items():
                if key not in all_params:
                    all_params[key] = value

            # Aggregate metrics
            results = exp_data.get("results", {})
            if isinstance(results, dict):
                for key, value in results.items():
                    if key not in all_metrics:
                        all_metrics[key] = value
            elif isinstance(results, list) and results:
                # Use last entry for metrics
                last_entry = results[-1] if isinstance(results[-1], dict) else {}
                for key, value in last_entry.items():
                    if key not in all_metrics:
                        all_metrics[key] = value

            # Aggregate meta from metadata
            metadata = exp_data.get("metadata", {})
            flat_meta = flatten_dict(metadata)
            for key, value in flat_meta.items():
                if key not in all_meta:
                    all_meta[key] = value

            # Add convenience fields to meta
            for field in ["id", "name", "status", "description", "tags"]:
                if field not in all_meta and field in exp_data:
                    all_meta[field] = exp_data[field]

        except Exception:
            # Continue with other experiments if one fails
            continue

    if not all_params and not all_metrics and not all_meta:
        return None

    return AccessResolver(params=all_params, metrics=all_metrics, meta=all_meta)


def _resolve_column_spec(
    resolver: AccessResolver,
    spec: str | list[str],
    scope: str,
) -> str | list[str]:
    """Resolve a column specification using AccessResolver.

    Args:
        resolver: AccessResolver instance
        spec: Column specification - "auto", "all", "none", or list of keys/patterns
        scope: Scope for resolution - "param", "metric", or "meta"

    Returns:
        Special value unchanged, or list of resolved paths (without group prefix)

    Raises:
        click.ClickException: If resolution fails
    """
    # Special values pass through unchanged
    if spec in ("auto", "all", "none"):
        return spec

    # Resolve list of keys/patterns
    if not isinstance(spec, list):
        return spec

    try:
        # Resolve each value (handles both single keys and patterns)
        canonical_keys = resolver.resolve_list(spec, scope=scope)

        # Strip group prefixes to get paths for comparison extractor
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
            f"Ambiguous key '{e.key}' in --{scope}s matches multiple: {matches_str}\n"
            f"Use a more specific path or group prefix (e.g., {scope}:{e.key})"
        )
    except KeyNotFoundError as e:
        available = resolver.get_paths(scope=scope)[:10]
        available_str = ", ".join(available) if available else "(none)"
        if len(resolver.get_paths(scope=scope)) > 10:
            available_str += ", ..."
        raise click.ClickException(
            f"Key '{e.key}' not found in {scope}s.\nAvailable: {available_str}"
        )


@click.command("compare")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=False)
@click.option(
    "--params",
    default="auto",
    help="Parameters to show: 'auto' (differing only), 'all', 'none', or comma-separated list",
)
@click.option(
    "--metrics",
    default="auto",
    help="Metrics to show: 'auto' (differing only), 'all', 'none', or comma-separated list",
)
@click.option(
    "--meta",
    default="auto",
    help="Metadata to show: 'auto' (id,name,status), 'all', 'none', or comma-separated list",
)
@click.option(
    "--include-dep-params",
    is_flag=True,
    help="Include parameters from dependency experiments (merged with local params)",
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
    params: str,
    metrics: str,
    meta: str,
    include_dep_params: bool,
    no_interactive: bool,
    max_rows: int | None,
):
    """
    Compare experiments in an interactive table showing parameters and metrics.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    \b
    Column selection (--params, --metrics, --meta):
      auto     Show only columns that differ across experiments (default)
      all      Show all columns in that category
      none     Hide all columns in that category
      <list>   Comma-separated list of specific columns

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
        yanex compare                                    # All experiments (auto mode)
        yanex compare exp1 exp2 exp3                    # Specific experiments
        yanex compare -s completed                      # Completed experiments
        yanex compare --params all --metrics none      # All params, no metrics
        yanex compare --params lr,epochs               # Show only specified parameters
        yanex compare --format csv > results.csv       # Export to CSV file
        yanex compare --format json                     # Export as JSON
        yanex compare --no-interactive                  # Static table output
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
    try:
        filter_obj = ExperimentFilter()

        # Parse comma-separated IDs into a list
        ids_list = None
        if ids:
            ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

        # Validate mutually exclusive targeting
        has_identifiers = len(experiment_identifiers) > 0
        # Note: name_pattern="" is a valid filter for unnamed experiments
        has_filters = any(
            [
                ids_list,
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

        # Parse parameter, metric, and meta column specifications
        # Special values "auto", "all", "none" are passed as strings
        # Otherwise, parse as comma-separated list
        def parse_column_spec(spec: str) -> str | list[str]:
            if spec in ("auto", "all", "none"):
                return spec
            return [s.strip() for s in spec.split(",") if s.strip()]

        params_spec = parse_column_spec(params)
        metrics_spec = parse_column_spec(metrics)
        meta_spec = parse_column_spec(meta)

        # Get comparison data extractor
        comparison_data_extractor = ExperimentComparisonData()

        # Build resolver for sub-path resolution and pattern matching
        # Only needed if any spec is a list (not "auto", "all", "none")
        needs_resolution = any(
            isinstance(s, list) for s in [params_spec, metrics_spec, meta_spec]
        )

        if needs_resolution:
            resolver = _build_resolver_from_experiments(
                comparison_data_extractor,
                experiment_ids,
                include_archived=archived,
                include_dep_params=include_dep_params,
            )
            if resolver:
                # Resolve column specs (handles sub-path resolution and patterns)
                params_spec = _resolve_column_spec(resolver, params_spec, "param")
                metrics_spec = _resolve_column_spec(resolver, metrics_spec, "metric")
                meta_spec = _resolve_column_spec(resolver, meta_spec, "meta")

        # Get comparison data
        comparison_data = comparison_data_extractor.get_comparison_data(
            experiment_ids=experiment_ids,
            params=params_spec,
            metrics=metrics_spec,
            meta=meta_spec,
            include_dep_params=include_dep_params,
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
