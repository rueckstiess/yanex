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
    allow_dirty: bool = False,
    verbose: bool = False,
) -> list[ExperimentResult]:
    """
    Run multiple experiments sequentially or in parallel.

    This is the main entry point for programmatic batch experiment execution.
    Used by both the Python API and internally by the CLI for sweep execution.

    Args:
        experiments: List of ExperimentSpec objects defining what to run
        parallel: Number of parallel workers (None=sequential, 0=auto-detect CPUs)
        allow_dirty: Allow running with uncommitted git changes
        verbose: Show detailed execution output

    Returns:
        List of ExperimentResult objects with IDs and status

    Raises:
        ValueError: If experiments list is empty or specs are invalid
        ExperimentContextError: If called from within CLI context

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

    # Check we're not in CLI context
    if _is_cli_context():
        from .utils.exceptions import ExperimentContextError

        raise ExperimentContextError(
            "Cannot use yanex.run_multiple() from within 'yanex run' context. "
            "Use this API when running scripts directly: python script.py"
        )

    # Validate all specs
    for i, spec in enumerate(experiments):
        try:
            spec.validate()
        except Exception as e:
            raise ValueError(f"Invalid ExperimentSpec at index {i}: {e}") from e

    # Route to sequential or parallel execution
    if parallel is None:
        return _run_sequential(experiments, allow_dirty, verbose)
    else:
        return _run_parallel(experiments, parallel, allow_dirty, verbose)


def _is_cli_context() -> bool:
    """Check if currently running in a yanex CLI-managed experiment.

    Returns:
        True if YANEX_CLI_ACTIVE environment variable is set
    """
    return bool(os.environ.get("YANEX_CLI_ACTIVE"))


def _run_sequential(
    experiments: list[ExperimentSpec],
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments one by one sequentially.

    Args:
        experiments: List of experiment specifications
        allow_dirty: Allow running with uncommitted changes
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
                allow_dirty=allow_dirty,
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
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments in parallel using process pool.

    Args:
        experiments: List of experiment specifications
        max_workers: Maximum number of parallel workers (0=auto-detect)
        allow_dirty: Allow running with uncommitted changes
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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_spec = {
            executor.submit(
                _execute_single_experiment, spec, allow_dirty, verbose
            ): spec
            for spec in experiments
        }

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

    # Summary
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    console.print("\n[bold]Parallel execution completed:[/]")
    console.print(f"  ✓ Completed: {len(completed)}/{len(experiments)}")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)}/{len(experiments)}")

    return results


def _execute_single_experiment(
    spec: ExperimentSpec,
    allow_dirty: bool,
    verbose: bool,
) -> ExperimentResult:
    """
    Execute a single experiment (runs in separate process for parallel mode).

    Must create its own ExperimentManager since each process is isolated.

    Args:
        spec: Experiment specification
        allow_dirty: Allow uncommitted changes
        verbose: Show verbose output

    Returns:
        ExperimentResult with status and details
    """
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
            allow_dirty=allow_dirty,
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

    except Exception as e:
        # Try to mark as failed
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
            experiment_id="unknown",
            status="failed",
            error_message=error_msg,
            name=spec.name,
        )
