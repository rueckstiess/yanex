# Programmatic Multi-Experiment Execution API

**Status:** Planning - Architecture Finalized (Unified Approach)
**Date:** 2025-01-04
**Updated:** 2025-01-04
**Context:** Add API for scripts to programmatically spawn and execute multiple yanex experiments (sequential or parallel)
**Architecture:** Unified - CLI will be refactored to use shared `ExperimentSpec` and `run_multiple()` executor

## Background

### Current Limitation
The yanex API only supports single-experiment execution via context managers:
```python
with yanex.create_experiment(...) as exp:
    # Single experiment execution
    pass
```

### Use Case
Scripts need to programmatically spawn multiple experiments. Primary use case: k-fold cross-validation where a training script needs to:
1. Detect it's orchestrating multiple folds
2. Spawn N separate yanex experiments (one per fold)
3. Run them in parallel
4. Optionally aggregate results

But this pattern should be **generic** - any script that needs to run multiple sub-experiments.

## Design Decisions

### 1. API Approach: **Unified API with ExperimentSpec**

**Selected:** Option B from investigation - Full unified API with subprocess support

**Why:**
- Clean, extensible API design
- Not significantly more work than MVP
- Supports subprocess execution (leveraging existing CLI machinery)
- Room to add inline function execution later
- Future-proof

**Deferred:** Inline function execution (Option C) - can add later if needed

### 2. Orchestration/Execution Pattern

Scripts will use a **dual-mode pattern** to prevent infinite recursion:

```python
# train.py
import yanex
from pathlib import Path

# Check if we're in orchestration or execution mode
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # ORCHESTRATION MODE: Spawn multiple experiments

    # Load metadata to detect k-fold setup
    data_exp_id = args.data_exp
    metadata = load_metadata(f"~/.yanex/experiments/{data_exp_id}/...")
    n_folds = metadata['n_folds']

    print(f"Detected {n_folds}-fold setup, spawning {n_folds} experiments...")

    # Create experiment specs
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),  # THIS script
            config={
                "_fold_idx": i,  # Signal execution mode
                "learning_rate": yanex.get_param('learning_rate', default=0.01),
                # ... other hyperparameters
            },
            script_args=["--data-exp", data_exp_id],  # Pass through
            name=f"kfold-{i}",
            tags=["kfold", "training", f"data:{data_exp_id}"]
        )
        for i in range(n_folds)
    ]

    # Run all experiments
    results = yanex.run_multiple(experiments, parallel=n_folds)

    # Report results
    completed = [r for r in results if r.status == "completed"]
    print(f"Completed {len(completed)}/{n_folds} folds")

    sys.exit(0)  # Exit orchestrator

else:
    # EXECUTION MODE: Train specific fold
    print(f"Training fold {fold_idx}...")

    # Normal training code
    train_data = load_fold_data(args.data_exp, fold_idx)
    model = train_model(train_data, learning_rate, ...)

    yanex.log_metrics({"fold": fold_idx, "accuracy": accuracy})
```

**Key conventions:**
- Use `_` prefix for system/coordination parameters (e.g., `_fold_idx`)
- Orchestrator exits early after spawning
- Execution mode runs normal training logic
- Script args pass through via `script_args=`

## Architecture Decision: Unified Approach

**Decision**: Refactor CLI to use `ExperimentSpec` and share execution code with API, rather than duplicating logic.

**Rationale**:
- CLI sweep execution already builds similar dict structures
- ~200 lines of duplicate code can be eliminated
- Single source of truth for execution logic
- CLI and API guaranteed to behave identically
- Only 2-4 hours more work upfront, massive long-term benefits

**Trade-offs**:
- More upfront work to refactor CLI
- Need to ensure all CLI tests still pass
- But: Cleaner architecture, better maintainability, no drift over time

## Implementation Plan

### Phase 1: Core Executor Infrastructure

#### 1.1 Create New Module: `yanex/executor.py`

**Purpose:** Batch experiment execution API

**Components:**

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

@dataclass
class ExperimentSpec:
    """Specification for a single experiment to run."""

    # For subprocess execution (primary mode)
    script_path: Path | None = None
    script_args: list[str] = field(default_factory=list)

    # For inline execution (future enhancement)
    function: Callable | None = None

    # Common configuration
    config: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    description: str | None = None

    def validate(self) -> None:
        """Validate that exactly one execution mode is specified."""
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
    """Result of running a single experiment."""
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


def _run_sequential(
    experiments: list[ExperimentSpec],
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments sequentially."""
    # Implementation similar to CLI's _execute_sweep_sequential
    # but works with ExperimentSpec objects
    pass


def _run_parallel(
    experiments: list[ExperimentSpec],
    max_workers: int,
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments in parallel using ProcessPoolExecutor."""
    # Implementation similar to CLI's _execute_sweep_parallel
    # but works with ExperimentSpec objects
    pass


def _execute_single_experiment(
    spec: ExperimentSpec,
    allow_dirty: bool,
    verbose: bool,
) -> ExperimentResult:
    """
    Worker function for executing a single experiment.

    Runs in separate process for parallel execution.
    Must create its own ExperimentManager.
    """
    # Implementation similar to CLI's _execute_single_sweep_experiment
    # but works with ExperimentSpec
    pass
```

#### 1.2 Refactor CLI Code for Reuse

**Extract from `yanex/cli/commands/run.py`:**

Move worker function to shared location:
```python
# Currently: _execute_single_sweep_experiment() in run.py
# Move to: yanex/core/execution.py or yanex/executor.py

# Make it work with both:
# - CLI's internal data structures
# - API's ExperimentSpec objects
```

**Approach:**
- Keep CLI code as-is initially
- Duplicate logic in executor.py (avoid premature abstraction)
- Refactor for code sharing later if patterns stabilize

#### 1.3 Update API Exports

**`yanex/__init__.py`:**
```python
# Existing exports
from .api import (
    create_experiment,
    get_params,
    get_param,
    log_metrics,
    # ...
)

# NEW: Batch execution API
from .executor import (
    ExperimentSpec,
    ExperimentResult,
    run_multiple,
)

__all__ = [
    # ... existing ...
    "ExperimentSpec",
    "ExperimentResult",
    "run_multiple",
]
```

**`yanex/api.py`:**
```python
def _is_cli_context() -> bool:
    """Check if currently running in a yanex CLI-managed experiment."""
    return bool(os.environ.get("YANEX_CLI_ACTIVE"))

# Add similar check for batch execution context if needed
def _is_batch_execution() -> bool:
    """Check if currently in batch execution orchestration."""
    return bool(os.environ.get("YANEX_BATCH_EXECUTION"))
```

#### 1.2 Implementation Details

**Sequential Execution:**

```python
def _run_sequential(
    experiments: list[ExperimentSpec],
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments one by one."""
    from rich.console import Console
    from .core.manager import ExperimentManager
    from .core.script_executor import ScriptExecutor

    console = Console()
    results = []

    console.print(f"Running {len(experiments)} experiments sequentially...")

    for i, spec in enumerate(experiments, 1):
        console.print(f"\n[cyan]Experiment {i}/{len(experiments)}: {spec.name or 'unnamed'}[/]")

        try:
            # Create experiment
            manager = ExperimentManager()
            experiment_id = manager.create_experiment(
                script_path=spec.script_path,
                name=spec.name,
                config=spec.config,
                tags=spec.tags,
                description=spec.description,
                allow_dirty=allow_dirty,
                script_args=spec.script_args,
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
                spec.script_args
            )

            # Success
            duration = manager.get_experiment_metadata(experiment_id).get("duration")
            results.append(ExperimentResult(
                experiment_id=experiment_id,
                status="completed",
                name=spec.name,
                duration=duration,
            ))
            console.print(f"  [green]✓ Completed: {experiment_id}[/]")

        except Exception as e:
            # Failure - try to mark experiment as failed
            error_msg = str(e)
            try:
                if 'experiment_id' in locals():
                    manager.fail_experiment(experiment_id, error_msg)
                    results.append(ExperimentResult(
                        experiment_id=experiment_id,
                        status="failed",
                        error_message=error_msg,
                        name=spec.name,
                    ))
            except:
                pass

            console.print(f"  [red]✗ Failed: {error_msg}[/]")

            # Continue to next experiment (don't abort whole batch)
            continue

    # Summary
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    console.print(f"\n[bold]Batch execution completed:[/]")
    console.print(f"  ✓ Completed: {len(completed)}/{len(experiments)}")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)}/{len(experiments)}")

    return results
```

**Parallel Execution:**

```python
def _run_parallel(
    experiments: list[ExperimentSpec],
    max_workers: int,
    allow_dirty: bool,
    verbose: bool,
) -> list[ExperimentResult]:
    """Execute experiments in parallel using process pool."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from rich.console import Console

    console = Console()

    # Auto-detect CPU count if requested
    if max_workers == 0:
        max_workers = multiprocessing.cpu_count()
        console.print(f"Auto-detected {max_workers} CPUs")

    console.print(f"Running {len(experiments)} experiments with {max_workers} parallel workers...")

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_spec = {
            executor.submit(
                _execute_single_experiment,
                spec,
                allow_dirty,
                verbose
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
                    console.print(f"[green]✓ {completed_count}/{len(experiments)} completed: {result.experiment_id}[/]")
                else:
                    console.print(f"[red]✗ Failed: {spec.name} - {result.error_message}[/]")

            except Exception as e:
                # Worker process crashed
                console.print(f"[red]✗ Worker crashed for {spec.name}: {e}[/]")
                results.append(ExperimentResult(
                    experiment_id="unknown",
                    status="failed",
                    error_message=f"Worker process crashed: {e}",
                    name=spec.name,
                ))

    # Summary
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]
    console.print(f"\n[bold]Parallel execution completed:[/]")
    console.print(f"  ✓ Completed: {len(completed)}/{len(experiments)}")
    if failed:
        console.print(f"  ✗ Failed: {len(failed)}/{len(experiments)}")

    return results
```

**Worker Function:**

```python
def _execute_single_experiment(
    spec: ExperimentSpec,
    allow_dirty: bool,
    verbose: bool,
) -> ExperimentResult:
    """
    Execute a single experiment (runs in separate process for parallel mode).

    Must create its own ExperimentManager since each process is isolated.
    """
    from .core.manager import ExperimentManager
    from .core.script_executor import ScriptExecutor

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
            spec.script_args
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
            if 'experiment_id' in locals():
                manager.fail_experiment(experiment_id, error_msg)
                return ExperimentResult(
                    experiment_id=experiment_id,
                    status="failed",
                    error_message=error_msg,
                    name=spec.name,
                )
        except:
            pass

        # Return failure result
        return ExperimentResult(
            experiment_id="unknown",
            status="failed",
            error_message=error_msg,
            name=spec.name,
        )
```

### Phase 2: Refactor CLI to Use Executor

**Goal:** Eliminate ~200 lines of duplicate execution code in CLI by using the shared executor.

#### 2.1 Refactor `_execute_sweep_experiments()` in `run.py`

**Current approach** (~250 lines):
- `_execute_sweep_experiments()` routes to `_execute_sweep_sequential()` or `_execute_sweep_parallel()`
- `_execute_sweep_sequential()`: Loop, create experiment, execute (~70 lines)
- `_execute_sweep_parallel()`: ProcessPoolExecutor setup (~90 lines)
- `_execute_single_sweep_experiment()`: Worker function (~40 lines)

**New approach** (~50 lines):
```python
def _execute_sweep_experiments(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
    max_workers: int | None = None,
    script_args: list[str] | None = None,
) -> None:
    """Execute parameter sweep directly using shared executor."""
    from ...executor import ExperimentSpec, run_multiple

    console = Console()

    # Validate git (CLI-specific, done once before any experiments)
    if not ignore_dirty:
        from ...core.git_utils import get_git_repo, validate_clean_working_directory
        repo = get_git_repo(script.parent)
        validate_clean_working_directory(repo)

    # Expand parameter sweeps (CLI-specific config parsing)
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    console.print(f"✓ Parameter sweep detected: running {len(expanded_configs)} experiments")

    # Add "sweep" tag (CLI convention)
    sweep_tags = list(tags) if tags else []
    if "sweep" not in sweep_tags:
        sweep_tags.append("sweep")

    # Build ExperimentSpec objects
    experiments = [
        ExperimentSpec(
            script_path=script,
            config=expanded_config,
            name=_generate_sweep_experiment_name(name, expanded_config, sweep_param_paths),
            tags=sweep_tags,
            description=description,
            script_args=script_args if script_args else [],
        )
        for expanded_config in expanded_configs
    ]

    # Use shared executor
    results = run_multiple(
        experiments,
        parallel=max_workers,
        allow_dirty=ignore_dirty,
        verbose=verbose
    )

    # Print CLI-specific summary
    _print_sweep_summary(results, len(expanded_configs))
```

#### 2.2 Add Helper Function

```python
def _print_sweep_summary(results: list[ExperimentResult], total: int) -> None:
    """Print CLI-friendly summary of sweep execution."""
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    click.echo("\n✓ Sweep execution completed")
    click.echo(f"  Total: {total}")
    click.echo(f"  Completed: {len(completed)}")
    if failed:
        click.echo(f"  Failed: {len(failed)}")
```

#### 2.3 Delete Duplicate Functions

Remove from `run.py`:
- `_execute_sweep_sequential()` (~70 lines)
- `_execute_sweep_parallel()` (~90 lines)
- `_execute_single_sweep_experiment()` (~40 lines)

**Net change: ~200 lines removed, ~50 lines added = 150 lines eliminated**

#### 2.4 Keep CLI-Specific Helpers

These remain because they're CLI-specific concerns:
- `_generate_sweep_experiment_name()` - Formats names with parameter values
- `_normalize_tags()` - Converts config tag formats to list
- New: `_print_sweep_summary()` - Pretty CLI output

#### 2.5 Consider: Refactor Staged Execution

Optional additional cleanup - `_execute_all_staged_experiments()` could also use `run_multiple()`:
```python
def _execute_all_staged_experiments(parallel: int | None, verbose: bool) -> None:
    """Execute all staged experiments using shared executor."""
    manager = ExperimentManager()
    staged_ids = manager.get_staged_experiments()

    # Build ExperimentSpec from metadata
    experiments = []
    for exp_id in staged_ids:
        metadata = manager.storage.load_metadata(exp_id)
        experiments.append(
            ExperimentSpec(
                script_path=Path(metadata["script_path"]),
                config=manager.storage.load_config(exp_id),
                name=metadata.get("name"),
                tags=metadata.get("tags", []),
                description=metadata.get("description"),
                script_args=metadata.get("script_args", []),
            )
        )

    # Mark as running before execution
    for exp_id in staged_ids:
        manager.execute_staged_experiment(exp_id)

    # Use shared executor
    results = run_multiple(experiments, parallel=parallel, allow_dirty=True, verbose=verbose)
```

### Phase 3: Testing

#### 3.1 New API Tests

**`tests/api/test_batch_execution.py`:**
```python
"""Tests for programmatic batch experiment execution."""

import os
from pathlib import Path
import pytest

from yanex import ExperimentSpec, run_multiple
from yanex.core.manager import ExperimentManager


class TestBatchExecution:
    """Test batch experiment execution API."""

    def test_run_multiple_sequential(self, tmp_path):
        """Test sequential execution of multiple experiments."""
        # Create simple test script
        script = tmp_path / "test.py"
        script.write_text("""
import yanex
idx = yanex.get_param('_idx')
yanex.log_metrics({'idx': idx, 'value': idx * 10})
""")

        # Setup isolated experiment directory
        old_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create experiment specs
            experiments = [
                ExperimentSpec(
                    script_path=script,
                    config={"_idx": i},
                    name=f"test-{i}",
                    tags=["test"]
                )
                for i in range(3)
            ]

            # Run sequentially
            results = run_multiple(experiments, parallel=None, allow_dirty=True)

            # Verify all completed
            assert len(results) == 3
            assert all(r.status == "completed" for r in results)
            assert all(r.experiment_id for r in results)

            # Verify experiments were created
            manager = ExperimentManager(experiments_dir=tmp_path / "experiments")
            all_exps = manager.list_experiments()
            assert len(all_exps) == 3

        finally:
            if old_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_run_multiple_parallel(self, tmp_path):
        """Test parallel execution of multiple experiments."""
        # Similar to sequential but with parallel=2
        pass

    def test_experiment_spec_validation(self):
        """Test ExperimentSpec validation."""
        # Must have either script_path or function
        spec = ExperimentSpec()
        with pytest.raises(ValueError, match="exactly one"):
            spec.validate()

        # Function not yet supported
        spec = ExperimentSpec(function=lambda x: x)
        with pytest.raises(NotImplementedError, match="Inline function"):
            spec.validate()

    def test_cli_context_prevention(self):
        """Test that run_multiple cannot be called from CLI context."""
        os.environ["YANEX_CLI_ACTIVE"] = "1"
        try:
            experiments = [ExperimentSpec(script_path=Path("test.py"))]
            with pytest.raises(ExperimentContextError, match="Cannot use"):
                run_multiple(experiments)
        finally:
            del os.environ["YANEX_CLI_ACTIVE"]

    def test_error_handling_continues(self, tmp_path):
        """Test that batch execution continues after individual failures."""
        # Create script that fails on certain indices
        script = tmp_path / "test.py"
        script.write_text("""
import yanex
idx = yanex.get_param('_idx')
if idx == 1:
    raise ValueError("Intentional failure")
yanex.log_metrics({'idx': idx})
""")

        # Test that 2 succeed, 1 fails, but execution continues
        pass
```

#### 3.2 Verify Existing CLI Tests

**Critical**: All existing CLI tests must pass to ensure backward compatibility:
- Run `pytest tests/cli/test_run.py` - Single experiment tests
- Run `pytest tests/cli/test_sweep.py` - Parameter sweep tests (if exists)
- Run `pytest tests/cli/test_parallel.py` - Parallel execution tests (if exists)
- Verify CLI output formatting unchanged

#### 3.3 Integration Testing

Test that CLI refactor works correctly:
- CLI sweeps execute using new executor
- Results are identical to old implementation
- Error handling behaves the same
- Progress output is user-friendly

**Test coverage goals**:
- Sequential execution
- Parallel execution with various worker counts
- Auto-detect CPU count (`parallel=0`)
- Error handling (individual experiment failures don't abort batch)
- Validation (empty list, invalid specs)
- Context prevention (can't call from CLI)
- Script args pass-through
- Tag/name/description propagation
- Result object correctness

### Phase 4: Documentation & Examples

#### 4.1 Example: K-Fold Training

**`examples/api/kfold_training.py`:**
```python
#!/usr/bin/env python3
"""
Example: K-fold cross-validation with programmatic experiment spawning.

This demonstrates the orchestration/execution pattern where a script
detects it needs to run multiple folds and spawns sub-experiments.

Usage:
    # Run the orchestrator (which spawns 5 fold experiments)
    yanex run kfold_training.py --data-exp abc123 -p learning_rate=0.01 --ignore-dirty

    # Each fold runs as a separate yanex experiment
"""

import argparse
import json
import sys
from pathlib import Path

import yanex


def load_data_experiment_metadata(exp_id: str) -> dict:
    """Load metadata from a data preparation experiment."""
    metadata_path = (
        Path.home() / ".yanex/experiments" / exp_id / "artifacts/metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Data experiment {exp_id} not found")

    return json.loads(metadata_path.read_text())


def train_fold(data_exp_id: str, fold_idx: int, learning_rate: float) -> dict:
    """Train model on a specific fold."""
    print(f"Training fold {fold_idx} with lr={learning_rate}...")

    # Simulate loading fold data
    # In real code: load fold_{fold_idx}_train.pkl, fold_{fold_idx}_val.pkl

    # Simulate training
    accuracy = 0.8 + (fold_idx * 0.02)  # Dummy metric

    # Log metrics
    yanex.log_metrics({
        "fold": fold_idx,
        "accuracy": accuracy,
        "learning_rate": learning_rate,
    })

    print(f"Fold {fold_idx} complete: accuracy={accuracy:.4f}")

    return {"accuracy": accuracy}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="K-fold training example")
    parser.add_argument("--data-exp", required=True, help="Data experiment ID")
    args = parser.parse_args()

    # Get hyperparameters from yanex
    learning_rate = yanex.get_param("learning_rate", default=0.01)

    # Check if we're in orchestration or execution mode
    fold_idx = yanex.get_param("_fold_idx", default=None)

    if fold_idx is None:
        # ORCHESTRATION MODE: Spawn fold experiments

        print(f"Loading data experiment metadata: {args.data_exp}")
        metadata = load_data_experiment_metadata(args.data_exp)

        if metadata.get("split_strategy") != "kfold":
            print(f"Error: Data experiment is not k-fold (strategy: {metadata.get('split_strategy')})")
            sys.exit(1)

        n_folds = metadata["n_folds"]
        print(f"\n🎯 Detected {n_folds}-fold cross-validation setup")
        print(f"Spawning {n_folds} parallel training experiments...\n")

        # Create experiment specs for each fold
        experiments = [
            yanex.ExperimentSpec(
                script_path=Path(__file__),  # This script
                config={
                    "_fold_idx": i,  # Signal which fold
                    "learning_rate": learning_rate,
                },
                script_args=["--data-exp", args.data_exp],
                name=f"kfold-{i}",
                tags=["kfold", "training", f"data:{args.data_exp}"],
                description=f"K-fold training: fold {i}/{n_folds}",
            )
            for i in range(n_folds)
        ]

        # Run all folds in parallel
        results = yanex.run_multiple(experiments, parallel=n_folds)

        # Report results
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]

        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Complete")
        print(f"{'='*60}")
        print(f"✓ Completed: {len(completed)}/{n_folds} folds")
        if failed:
            print(f"✗ Failed: {len(failed)}/{n_folds} folds")
            for r in failed:
                print(f"  - {r.name}: {r.error_message}")

        # Could aggregate results here using yanex.results API

        sys.exit(0 if not failed else 1)

    else:
        # EXECUTION MODE: Train specific fold

        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*60}\n")

        result = train_fold(args.data_exp, fold_idx, learning_rate)

        print(f"\n✓ Fold {fold_idx} training complete")
        print(f"  Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
```

#### 4.2 Example: Generic Batch Execution

**`examples/api/batch_execution.py`:**
```python
#!/usr/bin/env python3
"""
Example: Generic batch experiment execution.

This shows how to programmatically run multiple experiments with
different configurations without using parameter sweep syntax.

Usage:
    python batch_execution.py
"""

from pathlib import Path
import yanex


def main():
    # Define experiments to run
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]

    print(f"Creating {len(learning_rates) * len(batch_sizes)} experiments...")

    experiments = []
    for lr in learning_rates:
        for bs in batch_sizes:
            experiments.append(
                yanex.ExperimentSpec(
                    script_path=Path("train.py"),
                    config={
                        "learning_rate": lr,
                        "batch_size": bs,
                        "epochs": 10,
                    },
                    name=f"lr{lr}-bs{bs}",
                    tags=["grid-search", "training"],
                    description=f"Grid search: lr={lr}, bs={bs}",
                )
            )

    # Run with 4 parallel workers
    results = yanex.run_multiple(experiments, parallel=4)

    # Analyze results
    completed = [r for r in results if r.status == "completed"]
    print(f"\nCompleted {len(completed)}/{len(experiments)} experiments")

    # Could use yanex.results to fetch metrics and find best config


if __name__ == "__main__":
    main()
```

#### 4.3 Update CLAUDE.md

Add section on programmatic execution:

```markdown
### Programmatic Multi-Experiment Execution

Scripts can programmatically spawn multiple yanex experiments using `yanex.run_multiple()`:

```python
import yanex
from pathlib import Path

experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": lr, "_fold_idx": i},
        script_args=["--data-exp", "abc123"],
        name=f"fold-{i}",
        tags=["kfold"]
    )
    for i, lr in enumerate([0.01, 0.001, 0.0001])
]

results = yanex.run_multiple(experiments, parallel=3)
```

**Orchestration/Execution Pattern:**
Use `_fold_idx` (or similar) parameter to distinguish between:
- **Orchestration mode**: `_fold_idx=None` → spawn sub-experiments
- **Execution mode**: `_fold_idx=N` → run specific fold

See `examples/api/kfold_training.py` for complete example.
```

### Phase 5: Polish

#### 5.1 Code Quality

- Run `uv run ruff format` to format all code
- Run `uv run ruff check --fix` to auto-fix linting issues
- Run `uv run ruff check` to verify no remaining issues
- Ensure all type hints are correct

#### 5.2 Full Test Suite

- Run `make test` or `pytest` to run complete test suite
- Verify 90%+ test coverage maintained
- Check that all existing tests still pass

#### 5.3 Documentation Updates

- Update CHANGELOG.md with new feature
- Review all docstrings for clarity
- Ensure examples run end-to-end

#### 5.4 Final Verification

- Test k-fold training example manually
- Test batch execution example manually
- Verify CLI sweeps work correctly
- Check that error messages are user-friendly

## Future Enhancements (Deferred)

### Inline Function Execution

Add support for:
```python
def train_model(config):
    lr = config["learning_rate"]
    # ... training code
    yanex.log_metrics({"accuracy": 0.95})

experiments = [
    ExperimentSpec(
        function=train_model,
        config={"learning_rate": lr}
    )
    for lr in [0.01, 0.001]
]

results = yanex.run_multiple(experiments, parallel=2)
```

**Challenges:**
- Function serialization (needs `cloudpickle`)
- Context isolation
- Error handling

### Progress Reporting

Add callback for progress updates:
```python
def on_progress(completed, total, result):
    print(f"{completed}/{total}: {result.experiment_id} - {result.status}")

results = yanex.run_multiple(
    experiments,
    parallel=5,
    on_progress=on_progress
)
```

### Result Aggregation Helpers

```python
# Automatic cross-validation statistics
cv_results = yanex.aggregate_cv_results(results)
print(f"Mean accuracy: {cv_results.mean_accuracy} ± {cv_results.std_accuracy}")
```

## Original Test Structure (For Reference)

**`tests/api/test_batch_execution.py`:**
```python
"""Tests for programmatic batch experiment execution."""

import os
from pathlib import Path
import pytest

from yanex import ExperimentSpec, run_multiple
from yanex.core.manager import ExperimentManager


class TestBatchExecution:
    """Test batch experiment execution API."""

    def test_run_multiple_sequential(self, tmp_path):
        """Test sequential execution of multiple experiments."""
        # Create simple test script
        script = tmp_path / "test.py"
        script.write_text("""
import yanex
idx = yanex.get_param('_idx')
yanex.log_metrics({'idx': idx, 'value': idx * 10})
""")

        # Setup isolated experiment directory
        old_dir = os.environ.get("YANEX_EXPERIMENTS_DIR")
        os.environ["YANEX_EXPERIMENTS_DIR"] = str(tmp_path / "experiments")

        try:
            # Create experiment specs
            experiments = [
                ExperimentSpec(
                    script_path=script,
                    config={"_idx": i},
                    name=f"test-{i}",
                    tags=["test"]
                )
                for i in range(3)
            ]

            # Run sequentially
            results = run_multiple(experiments, parallel=None, allow_dirty=True)

            # Verify all completed
            assert len(results) == 3
            assert all(r.status == "completed" for r in results)
            assert all(r.experiment_id for r in results)

            # Verify experiments were created
            manager = ExperimentManager(experiments_dir=tmp_path / "experiments")
            all_exps = manager.list_experiments()
            assert len(all_exps) == 3

        finally:
            if old_dir:
                os.environ["YANEX_EXPERIMENTS_DIR"] = old_dir
            elif "YANEX_EXPERIMENTS_DIR" in os.environ:
                del os.environ["YANEX_EXPERIMENTS_DIR"]

    def test_run_multiple_parallel(self, tmp_path):
        """Test parallel execution of multiple experiments."""
        # Similar to sequential but with parallel=2
        pass

    def test_experiment_spec_validation(self):
        """Test ExperimentSpec validation."""
        # Must have either script_path or function
        spec = ExperimentSpec()
        with pytest.raises(ValueError, match="exactly one"):
            spec.validate()

        # Function not yet supported
        spec = ExperimentSpec(function=lambda x: x)
        with pytest.raises(NotImplementedError, match="Inline function"):
            spec.validate()

    def test_cli_context_prevention(self):
        """Test that run_multiple cannot be called from CLI context."""
        os.environ["YANEX_CLI_ACTIVE"] = "1"
        try:
            experiments = [ExperimentSpec(script_path=Path("test.py"))]
            with pytest.raises(ExperimentContextError, match="Cannot use"):
                run_multiple(experiments)
        finally:
            del os.environ["YANEX_CLI_ACTIVE"]

    def test_error_handling_continues(self, tmp_path):
        """Test that batch execution continues after individual failures."""
        # Create script that fails on certain indices
        script = tmp_path / "test.py"
        script.write_text("""
import yanex
idx = yanex.get_param('_idx')
if idx == 1:
    raise ValueError("Intentional failure")
yanex.log_metrics({'idx': idx})
""")

        # ... test that 2 succeed, 1 fails, but execution continues
        pass
```

## Implementation Checklist

### Phase 1: Core Executor Infrastructure (5-6 hours)
- [ ] Create `yanex/executor.py` module
- [ ] Define `ExperimentSpec` dataclass with validation
- [ ] Define `ExperimentResult` dataclass
- [ ] Implement `run_multiple()` main function
- [ ] Implement `_run_sequential()` function
- [ ] Implement `_run_parallel()` function with ProcessPoolExecutor
- [ ] Implement `_execute_single_experiment()` worker function
- [ ] Update `yanex/__init__.py` exports
- [ ] Add context check to `yanex/api.py`

### Phase 2: Refactor CLI to Use Executor (3-4 hours)
- [ ] Refactor `_execute_sweep_experiments()` to build ExperimentSpec list
- [ ] Replace calls to `_execute_sweep_sequential/parallel` with `run_multiple()`
- [ ] Delete `_execute_sweep_sequential()` function (~70 lines)
- [ ] Delete `_execute_sweep_parallel()` function (~90 lines)
- [ ] Delete `_execute_single_sweep_experiment()` function (~40 lines)
- [ ] Add `_print_sweep_summary()` helper for CLI output
- [ ] Keep CLI-specific helpers (`_generate_sweep_experiment_name`, `_normalize_tags`)
- [ ] Verify net reduction of ~150 lines in run.py

### Phase 3: Testing (4-5 hours)
- [ ] Create `tests/api/test_batch_execution.py`
- [ ] Test sequential execution
- [ ] Test parallel execution with various worker counts
- [ ] Test auto-detect CPU count (`parallel=0`)
- [ ] Test error handling (individual failures don't abort batch)
- [ ] Test validation (empty list, invalid specs)
- [ ] Test context prevention (can't call from CLI)
- [ ] Test script args pass-through
- [ ] Run all existing CLI tests (backward compatibility)
- [ ] Achieve 90%+ coverage on new code

### Phase 4: Examples & Documentation (2-3 hours)
- [ ] Create `examples/api/kfold_training.py` with orchestration/execution pattern
- [ ] Create `examples/api/batch_execution.py` for generic usage
- [ ] Update `CLAUDE.md` with programmatic execution section
- [ ] Add docstrings to all new functions
- [ ] Update README if needed

### Phase 5: Polish (1 hour)
- [ ] Run `uv run ruff format`
- [ ] Run `uv run ruff check --fix`
- [ ] Run `uv run ruff check` to verify clean
- [ ] Run full test suite (`make test`)
- [ ] Update CHANGELOG.md
- [ ] Test k-fold example end-to-end
- [ ] Test batch example end-to-end
- [ ] Verify CLI sweeps work correctly

## Design Rationale

### Why Unified Architecture? (CLI Uses API)

**Decision**: Refactor CLI to use `ExperimentSpec` and `run_multiple()`, eliminating duplicate code.

**Benefits**:
1. **Single Source of Truth**: One execution path for both CLI and API
2. **Code Reduction**: ~150 lines eliminated from run.py
3. **Guaranteed Consistency**: CLI sweeps and API batches behave identically
4. **Maintainability**: Bug fixes automatically benefit both paths
5. **Future-Proof**: New features (progress callbacks, etc.) work everywhere

**Trade-offs**:
- +3-4 hours upfront to refactor CLI
- Need to ensure backward compatibility with CLI tests
- But: Massive long-term benefits outweigh short-term cost

### Why Subprocess-Only Initially?

1. **Leverage Existing Code**: CLI already has battle-tested parallel execution
2. **Process Isolation**: True isolation, no state pollution
3. **Simpler Implementation**: No function serialization complexity
4. **Real-World Use Case**: K-fold training needs subprocess execution anyway
5. **Can Add Later**: Inline functions can be added without breaking changes

### Why ExperimentSpec Pattern?

1. **Extensibility**: Easy to add `function` parameter later
2. **Clarity**: Explicit specification of what to run
3. **Validation**: Can validate before running
4. **Type Safety**: Dataclass with type hints
5. **Natural Fit**: Matches CLI's internal data structures

### Why CLI and API Share Code?

**Original consideration**: Keep CLI and API separate to avoid complexity.

**Better approach**: Share execution logic because:
1. **Data structures are identical**: CLI already builds experiment dicts matching ExperimentSpec
2. **Logic is identical**: Both need ProcessPoolExecutor, error handling, progress tracking
3. **Not more complex**: Actually simpler to maintain one path
4. **CLI benefits from API improvements**: And vice versa

### Why Allow Individual Failures?

1. **Robustness**: One bad fold shouldn't kill entire k-fold run
2. **Debugging**: Can see which specific experiments failed
3. **Partial Results**: Can still use completed experiments
4. **Real-World Usage**: Matches how researchers actually want to work

## Risk Assessment

### Low Risk
- ✅ Subprocess execution (proven in CLI)
- ✅ Process isolation (no shared state issues)
- ✅ ExperimentSpec pattern (clean, extensible)
- ✅ CLI refactoring (backward compatible, well-tested)

### Medium Risk
- ⚠️ Error handling in parallel mode (need comprehensive testing)
- ⚠️ User confusion about orchestration/execution pattern (needs good docs)
- ⚠️ CLI test compatibility (must verify all existing tests pass)

### High Risk (Deferred)
- ❌ Inline function execution (deferred - serialization challenges)
- ❌ Recursive experiment spawning (prevented by context checks)

### Risk Mitigation

**CLI Refactoring Risk**:
- Run full CLI test suite after refactoring
- Keep CLI output formatting identical
- Verify error messages unchanged
- Test manually with existing workflows

**Parallel Execution Risk**:
- Comprehensive unit tests for error scenarios
- Integration tests with real subprocess failures
- Test with various worker counts

## Success Criteria

1. ✅ Scripts can programmatically spawn N experiments via `run_multiple()`
2. ✅ Parallel execution with configurable workers
3. ✅ Individual failures don't abort batch
4. ✅ Works with k-fold cross-validation use case (orchestration/execution pattern)
5. ✅ Clean, documented API with `ExperimentSpec` and `ExperimentResult`
6. ✅ 90%+ test coverage maintained
7. ✅ No breaking changes to existing API or CLI behavior
8. ✅ CLI refactored to use shared executor (~150 lines removed)
9. ✅ All existing CLI tests pass (backward compatibility)
10. ✅ Single source of truth for execution logic

## Timeline Estimate (Unified Approach)

- **Phase 1 (Core Executor)**: 5-6 hours
  - Create executor.py module
  - Implement ExperimentSpec, ExperimentResult, run_multiple()
  - Implement sequential and parallel execution
  - Update exports

- **Phase 2 (CLI Refactor)**: 3-4 hours
  - Refactor _execute_sweep_experiments()
  - Delete duplicate functions (~200 lines)
  - Add CLI output helpers
  - Verify backward compatibility

- **Phase 3 (Testing)**: 4-5 hours
  - API tests (sequential, parallel, error handling)
  - Verify all CLI tests pass
  - Integration testing
  - Achieve 90%+ coverage

- **Phase 4 (Examples/Docs)**: 2-3 hours
  - K-fold training example
  - Generic batch execution example
  - Documentation updates

- **Phase 5 (Polish)**: 1 hour
  - Code formatting and linting
  - Full test suite
  - Manual verification

**Total**: 15-19 hours (~2-3 days)

**Comparison to Original Plan**:
- Original (duplicate code): 11-16 hours
- Unified (shared code): 15-19 hours
- **Additional cost: 2-4 hours**
- **Benefit: ~150 lines eliminated, single source of truth, guaranteed consistency**

## Related Issues/PRs

- Script argument pass-through (#31) - Enables script_args in ExperimentSpec
- Direct sweep execution (v0.6.0) - Parallel execution machinery to reuse
