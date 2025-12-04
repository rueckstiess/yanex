"""
Batch experiment execution module for Yanex.

Provides programmatic API for executing multiple experiments sequentially or in parallel.
This module is used by both the Python API (yanex.run_multiple()) and internally by the CLI
for parameter sweep execution.
"""

import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from .core.manager import ExperimentManager
from .core.script_executor import ScriptExecutor


@dataclass
class ExperimentSpec:
    """Specification for a single experiment to run.

    Defines what to execute (script or function) and how to configure it.
    Currently only subprocess execution is supported.

    Attributes:
        script_path: Path to Python script to execute (subprocess mode)
        script_args: Arguments to pass to script via sys.argv
        function: Function to execute inline (not yet supported)
        config: Configuration parameters for the experiment
        name: Optional experiment name
        tags: List of tags for organization
        description: Optional experiment description
        dependencies: Dict mapping slot names to experiment IDs this depends on
        cli_args: Parsed CLI arguments dictionary (yanex flags only, not script_args)
    """

    # Subprocess execution (primary mode)
    script_path: Path | None = None
    script_args: list[str] = field(default_factory=list)

    # Inline execution (future enhancement)
    function: Callable | None = None

    # Common configuration
    config: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str | None = None
    dependencies: dict[str, str] = field(default_factory=dict)
    cli_args: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate that exactly one execution mode is specified.

        Raises:
            ValueError: If neither or both of script_path/function are specified
            NotImplementedError: If function execution is requested (not yet supported)
        """
        if (self.script_path is None) == (self.function is None):
            raise ValueError(
                "Must specify exactly one of script_path or function. "
                "Currently only script_path (subprocess) is supported."
            )

        if self.function is not None:
            raise NotImplementedError(
                "Inline function execution not yet supported. "
                "Use script_path for subprocess execution."
            )


@dataclass
class ExperimentResult:
    """Result of running a single experiment.

    Attributes:
        experiment_id: Unique experiment ID (8-char hex)
        status: Experiment status ("completed", "failed", "cancelled")
        error_message: Error message if failed, None otherwise
        duration: Execution duration in seconds, None if not completed
        name: Experiment name if provided
    """

    experiment_id: str
    status: str  # "completed", "failed", "cancelled"
    error_message: str | None = None
    duration: float | None = None
    name: str | None = None


def run_multiple(
    experiments: list[ExperimentSpec],
    parallel: int | None = None,
    verbose: bool = False,
) -> list[ExperimentResult]:
    """
    Run multiple experiments sequentially or in parallel.

    This is the main entry point for programmatic batch experiment execution.
    Used by both the Python API and internally by the CLI for sweep execution.

    Args:
        experiments: List of ExperimentSpec objects defining what to run
        parallel: Number of parallel workers (None=sequential, 0=auto-detect CPUs)
        verbose: Show detailed execution output

    Returns:
        List of ExperimentResult objects with IDs and status

    Raises:
        ValueError: If experiments list is empty or specs are invalid

    Example:
        >>> import yanex
        >>> from pathlib import Path
        >>>
        >>> # K-fold cross-validation
        >>> experiments = [
        ...     yanex.ExperimentSpec(
        ...         script_path=Path("train.py"),
        ...         config={"learning_rate": 0.01, "_fold_idx": i},
        ...         script_args=["--data-exp", "abc123"],
        ...         name=f"kfold-{i}",
        ...         tags=["kfold", "training"]
        ...     )
        ...     for i in range(5)
        ... ]
        >>>
        >>> # Run in parallel
        >>> results = yanex.run_multiple(experiments, parallel=5)
        >>>
        >>> # Check results
        >>> completed = [r for r in results if r.status == "completed"]
        >>> print(f"Completed {len(completed)}/5 folds")
    """
    # Validation
    if not experiments:
        raise ValueError("experiments list cannot be empty")

    # Validate all specs
    for i, spec in enumerate(experiments):
        try:
            spec.validate()
        except Exception as e:
            raise ValueError(f"Invalid ExperimentSpec at index {i}: {e}") from e

    # Route to sequential or parallel execution
    if parallel is None:
        return _run_sequential(experiments, verbose)
    else:
        return _run_parallel(experiments, parallel, verbose)


def _run_sequential(
    experiments: list[ExperimentSpec],
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments one by one sequentially.

    Args:
        experiments: List of experiment specifications
        verbose: Show verbose output

    Returns:
        List of ExperimentResult objects
    """
    console = Console()
    results: list[ExperimentResult] = []

    console.print(f"Running {len(experiments)} experiments sequentially...")

    for i, spec in enumerate(experiments, 1):
        console.print(
            f"\n[cyan]Experiment {i}/{len(experiments)}: {spec.name or 'unnamed'}[/]"
        )

        try:
            # Create experiment manager
            manager = ExperimentManager()

            # Create experiment
            experiment_id = manager.create_experiment(
                script_path=spec.script_path,
                name=spec.name,
                config=spec.config,
                tags=spec.tags,
                description=spec.description,
                dependencies=spec.dependencies,
                script_args=spec.script_args,
                cli_args=spec.cli_args,
            )

            # Start experiment
            manager.start_experiment(experiment_id)

            # Execute script
            executor = ScriptExecutor(manager)
            executor.execute_script(
                experiment_id,
                spec.script_path,
                spec.config,
                verbose,
                spec.script_args,
            )

            # Success - get duration
            metadata = manager.get_experiment_metadata(experiment_id)
            duration = metadata.get("duration")

            results.append(
                ExperimentResult(
                    experiment_id=experiment_id,
                    status="completed",
                    name=spec.name,
                    duration=duration,
                )
            )
            console.print(f"  [green]✓ Completed: {experiment_id}[/]")

        except Exception as e:
            # Failure - try to mark experiment as failed
            error_msg = str(e)
            try:
                if "experiment_id" in locals():
                    # For click.Abort exceptions, error is already stored in metadata
                    # Read it back to get the actual error message
                    import click

                    if isinstance(e, click.Abort):
                        try:
                            metadata = manager.get_experiment_metadata(experiment_id)
                            error_msg = metadata.get(
                                "error_message", "Script execution failed"
                            )
                        except Exception:
                            error_msg = "Script execution failed"
                    else:
                        # Non-Abort exception - mark as failed
                        manager.fail_experiment(experiment_id, error_msg)

                    results.append(
                        ExperimentResult(
                            experiment_id=experiment_id,
                            status="failed",
                            error_message=error_msg,
                            name=spec.name,
                        )
                    )
                else:
                    # Experiment creation failed
                    results.append(
                        ExperimentResult(
                            experiment_id="unknown",
                            status="failed",
                            error_message=error_msg,
                            name=spec.name,
                        )
                    )
            except Exception:
                # If we can't even mark as failed, still record the failure
                results.append(
                    ExperimentResult(
                        experiment_id="unknown",
                        status="failed",
                        error_message=error_msg,
                        name=spec.name,
                    )
                )

            console.print(f"  [red]✗ Failed: {error_msg}[/]")

            # Continue to next experiment (don't abort whole batch)
            continue

    # Summary
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    console.print("\n[bold]Batch execution completed:[/]")
    console.print(f"  ✓ Completed: {len(completed)}/{len(experiments)}")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)}/{len(experiments)}")

    return results


def _run_parallel(
    experiments: list[ExperimentSpec],
    max_workers: int,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments in parallel using process pool.

    Args:
        experiments: List of experiment specifications
        max_workers: Maximum number of parallel workers (0=auto-detect)
        verbose: Show verbose output

    Returns:
        List of ExperimentResult objects
    """
    console = Console()

    # Auto-detect CPU count if requested
    if max_workers == 0:
        max_workers = multiprocessing.cpu_count()
        console.print(f"Auto-detected {max_workers} CPUs")

    console.print(
        f"Running {len(experiments)} experiments with {max_workers} parallel workers..."
    )

    results: list[ExperimentResult] = []
    interrupted = False

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_spec = {
            executor.submit(_execute_single_experiment, spec, verbose): spec
            for spec in experiments
        }

        try:
            # Process as they complete
            completed_count = 0
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result.status == "completed":
                        completed_count += 1
                        console.print(
                            f"[green]✓ {completed_count}/{len(experiments)} completed: {result.experiment_id}[/]"
                        )
                    elif result.status == "cancelled":
                        console.print(f"[yellow]✖ Cancelled: {result.experiment_id}[/]")
                    else:
                        console.print(
                            f"[red]✗ Failed: {spec.name} - {result.error_message}[/]"
                        )

                except Exception as e:
                    # Worker process crashed
                    console.print(f"[red]✗ Worker crashed for {spec.name}: {e}[/]")
                    results.append(
                        ExperimentResult(
                            experiment_id="unknown",
                            status="failed",
                            error_message=f"Worker process crashed: {e}",
                            name=spec.name,
                        )
                    )

        except KeyboardInterrupt:
            interrupted = True
            console.print("\n[yellow]Interrupted! Cancelling pending experiments...[/]")

            # Cancel pending futures (ones not yet started)
            for future in future_to_spec:
                future.cancel()

            # Collect results from futures that completed before interrupt
            for future in future_to_spec:
                if future.done() and not future.cancelled():
                    try:
                        result = future.result()
                        if result not in results:
                            results.append(result)
                    except Exception:
                        pass

            # Wait briefly for workers to finish (they should catch the signal)
            # ProcessPoolExecutor.__exit__ will call shutdown(wait=True)

    # After executor shutdown, mark any still-running experiments as cancelled
    if interrupted:
        _cancel_running_experiments(results, experiments, console)

    # Summary
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    cancelled = [r for r in results if r.status == "cancelled"]

    console.print("\n[bold]Parallel execution completed:[/]")
    console.print(f"  ✓ Completed: {len(completed)}/{len(experiments)}")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)}/{len(experiments)}")
    if cancelled:
        console.print(f"  ✖ Cancelled: {len(cancelled)}/{len(experiments)}")

    if interrupted:
        raise KeyboardInterrupt

    return results


def _cancel_running_experiments(
    results: list[ExperimentResult],
    experiments: list[ExperimentSpec],
    console: Console,
) -> None:
    """Cancel any experiments that are still in 'running' status after interrupt.

    This handles the case where worker processes were terminated before they could
    update the experiment status.

    Args:
        results: List of results collected so far
        experiments: Original list of experiment specs
        console: Console for output
    """
    from .core.filtering import ExperimentFilter
    from .core.manager import ExperimentManager

    # Get IDs of experiments we already have results for
    known_ids = {r.experiment_id for r in results if r.experiment_id != "unknown"}

    # Find experiments that are still running
    try:
        filter_obj = ExperimentFilter()
        running_experiments = filter_obj.filter_experiments(
            status="running", include_all=True
        )

        # Cancel any running experiments that might be from this batch
        # We check by matching experiment names or script paths
        batch_names = {spec.name for spec in experiments if spec.name}
        batch_scripts = {
            str(spec.script_path) for spec in experiments if spec.script_path
        }

        for exp in running_experiments:
            exp_id = exp.get("id")
            if exp_id and exp_id not in known_ids:
                # Check if this experiment matches our batch
                exp_name = exp.get("name")
                exp_script = exp.get("script_path")

                if (exp_name and exp_name in batch_names) or (
                    exp_script and exp_script in batch_scripts
                ):
                    try:
                        manager = ExperimentManager()
                        manager.cancel_experiment(
                            exp_id, "Interrupted by user (Ctrl+C)"
                        )
                        console.print(f"[yellow]✖ Cancelled: {exp_id}[/]")
                        results.append(
                            ExperimentResult(
                                experiment_id=exp_id,
                                status="cancelled",
                                error_message="Interrupted by user (Ctrl+C)",
                                name=exp_name,
                            )
                        )
                    except Exception:
                        pass  # Best effort

    except Exception:
        pass  # Best effort cleanup


def _execute_single_experiment(
    spec: ExperimentSpec,
    verbose: bool,
) -> ExperimentResult:
    """
    Execute a single experiment (runs in separate process for parallel mode).

    Must create its own ExperimentManager since each process is isolated.

    Args:
        spec: Experiment specification
        verbose: Show verbose output

    Returns:
        ExperimentResult with status and details
    """
    experiment_id = None
    manager = None

    try:
        # Create manager (each process needs its own)
        manager = ExperimentManager()

        # Create experiment
        experiment_id = manager.create_experiment(
            script_path=spec.script_path,
            name=spec.name,
            config=spec.config,
            tags=spec.tags,
            description=spec.description,
            dependencies=spec.dependencies,
            script_args=spec.script_args,
            cli_args=spec.cli_args,
        )

        # Start experiment
        manager.start_experiment(experiment_id)

        # Execute script
        executor = ScriptExecutor(manager)
        executor.execute_script(
            experiment_id,
            spec.script_path,
            spec.config,
            verbose,
            spec.script_args,
        )

        # Get duration
        metadata = manager.get_experiment_metadata(experiment_id)
        duration = metadata.get("duration")

        return ExperimentResult(
            experiment_id=experiment_id,
            status="completed",
            name=spec.name,
            duration=duration,
        )

    except KeyboardInterrupt:
        # User interrupted - mark as cancelled
        if experiment_id and manager:
            try:
                manager.cancel_experiment(experiment_id, "Interrupted by user (Ctrl+C)")
            except Exception:
                pass  # Best effort

        return ExperimentResult(
            experiment_id=experiment_id or "unknown",
            status="cancelled",
            error_message="Interrupted by user (Ctrl+C)",
            name=spec.name,
        )

    except Exception as e:
        # Try to mark as failed
        error_msg = str(e)
        try:
            if experiment_id and manager:
                # For click.Abort exceptions, error is already stored in metadata
                # Read it back to get the actual error message
                import click

                if isinstance(e, click.Abort):
                    try:
                        metadata = manager.get_experiment_metadata(experiment_id)
                        error_msg = metadata.get(
                            "error_message", "Script execution failed"
                        )
                    except Exception:
                        error_msg = "Script execution failed"
                else:
                    # Non-Abort exception - mark as failed
                    manager.fail_experiment(experiment_id, error_msg)

                return ExperimentResult(
                    experiment_id=experiment_id,
                    status="failed",
                    error_message=error_msg,
                    name=spec.name,
                )
        except Exception:
            pass

        # Return failure result
        return ExperimentResult(
            experiment_id=experiment_id or "unknown",
            status="failed",
            error_message=error_msg,
            name=spec.name,
        )
