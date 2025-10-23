# Implementation Plan: Parallel Experiment Execution

**Status**: In Progress
**Created**: 2025-01-23
**Target Version**: v0.5.0

## Executive Summary

Enable Yanex to run multiple experiments in parallel using multiprocessing, removing the current single-experiment restriction. Add a `--parallel N` flag to `yanex run --staged` to execute N experiments concurrently.

## Motivation

Users with multi-core systems (e.g., 128GB MacBook with multiple CPU/GPU cores) want to train 3-4 neural networks simultaneously during parameter sweeps. Currently, Yanex enforces sequential execution even when running staged experiments, leaving resources underutilized.

## Current Architecture Analysis

### Key Constraint Points

1. **`prevent_concurrent_execution()` in `yanex/core/manager.py:350-361`**
   - Called in `create_experiment()` when `stage_only=False`
   - Scans for any experiment with `status='running'`
   - Raises `ExperimentAlreadyRunningError` if found
   - **Impact**: Only prevents concurrent creation, not concurrent execution

2. **Sequential Loop in `yanex/cli/commands/run.py:392-438`**
   - `_execute_staged_experiments()` uses `for experiment_id in staged_experiments:`
   - Each experiment blocks until completion
   - **Impact**: This is the PRIMARY bottleneck for parallel execution

3. **File-Based Storage (No Locking)**
   - Metadata stored in `metadata.json` per experiment directory
   - Each experiment has isolated directory: `~/.yanex/experiments/{experiment_id}/`
   - Load-modify-save pattern without atomic operations
   - **Impact**: Race conditions possible on shared state (rare since experiments are isolated)

### Storage Isolation Analysis

**Good news**: Experiments are naturally isolated!
- Each experiment ID gets its own directory
- No shared files between experiments
- Only potential conflict is during status updates on same experiment (shouldn't happen)

## Implementation Plan

### Phase 1: Remove Concurrency Restrictions (Core Changes)

#### 1.1 Modify `ExperimentManager.prevent_concurrent_execution()`
**File**: `yanex/core/manager.py:350-361`

**Current**:
```python
def prevent_concurrent_execution(self) -> None:
    """Ensure no other experiment is currently running.

    Raises:
        ExperimentAlreadyRunningError: If another experiment is running
    """
    running_experiment = self.get_running_experiment()
    if running_experiment is not None:
        raise ExperimentAlreadyRunningError(
            f"Experiment {running_experiment} is already running. "
            "Only one experiment can run at a time."
        )
```

**Proposed**:
```python
def prevent_concurrent_execution(self, allow_parallel: bool = False) -> None:
    """Ensure no other experiment is currently running (unless parallel mode enabled).

    Args:
        allow_parallel: If True, skip the concurrent execution check

    Raises:
        ExperimentAlreadyRunningError: If another experiment is running and allow_parallel=False
    """
    if allow_parallel:
        return  # Skip check in parallel mode

    running_experiment = self.get_running_experiment()
    if running_experiment is not None:
        raise ExperimentAlreadyRunningError(
            f"Experiment {running_experiment} is already running. "
            "Only one experiment can run at a time. "
            "Use --parallel flag with --staged to enable parallel execution."
        )
```

**Update callers**:
```python
# In create_experiment() at line 398-399
if not stage_only:
    self.prevent_concurrent_execution(allow_parallel=False)  # Explicit for now
```

#### 1.2 Add `get_running_experiments()` (plural)
**File**: `yanex/core/manager.py` - new method after line 99

**Purpose**: Useful for monitoring and debugging parallel execution

```python
def get_running_experiments(self) -> list[str]:
    """Get all currently running experiments.

    Returns:
        List of experiment IDs with status='running'
    """
    if not self.experiments_dir.exists():
        return []

    running_experiments = []
    for experiment_dir in self.experiments_dir.iterdir():
        if not experiment_dir.is_dir():
            continue

        experiment_id = experiment_dir.name
        if self.storage.experiment_exists(experiment_id):
            try:
                metadata = self.storage.load_metadata(experiment_id)
                if metadata.get("status") == "running":
                    running_experiments.append(experiment_id)
            except Exception:
                # Skip experiments with corrupted metadata
                continue

    return running_experiments
```

### Phase 2: Add Parallel Execution Support

#### 2.1 Add CLI Flag
**File**: `yanex/cli/commands/run.py:44-48`

**Add after `--staged` flag**:
```python
@click.option(
    "--parallel",
    "-j",
    type=int,
    metavar="N",
    help="Execute N experiments in parallel (only valid with --staged). Use 0 for auto (number of CPUs).",
)
```

**Update function signature** at line 50:
```python
def run(
    ctx: click.Context,
    script: Path | None,
    config: Path | None,
    param: list[str],
    name: str | None,
    tag: list[str],
    description: str | None,
    dry_run: bool,
    ignore_dirty: bool,
    stage: bool,
    staged: bool,
    parallel: int | None,  # NEW
) -> None:
```

**Add validation** after line 114:
```python
# Validate parallel flag
if parallel is not None and not staged:
    click.echo("Error: --parallel flag can only be used with --staged", err=True)
    raise click.Abort()

if parallel is not None and parallel < 0:
    click.echo("Error: --parallel must be 0 (auto) or positive integer", err=True)
    raise click.Abort()
```

**Update call to `_execute_staged_experiments()`** at line 118:
```python
if staged:
    _execute_staged_experiments(verbose, console, max_workers=parallel)
    return
```

#### 2.2 Implement Parallel Execution
**File**: `yanex/cli/commands/run.py:392-438`

**Strategy**:
- Keep existing sequential code path for backward compatibility
- Add new parallel code path using `ProcessPoolExecutor`
- Use multiprocessing instead of threading for true parallelism (GIL bypass)

**Replace entire `_execute_staged_experiments()` function**:

```python
def _execute_staged_experiments(
    verbose: bool = False,
    console: Console = None,
    max_workers: int | None = None,
) -> None:
    """Execute all staged experiments, optionally in parallel.

    Args:
        verbose: Show verbose output
        console: Rich console for output
        max_workers: Maximum parallel workers. None=sequential, 0=auto (CPU count)
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if console is None:
        console = Console()

    manager = ExperimentManager()
    staged_experiments = manager.get_staged_experiments()

    if not staged_experiments:
        console.print("[dim]No staged experiments found[/]")
        return

    # Determine execution mode
    if max_workers is None:
        # Sequential execution (backward compatible)
        _execute_staged_sequential(staged_experiments, manager, verbose, console)
    else:
        # Parallel execution
        if max_workers == 0:
            max_workers = multiprocessing.cpu_count()

        _execute_staged_parallel(
            staged_experiments, manager, verbose, console, max_workers
        )


def _execute_staged_sequential(
    staged_experiments: list[str],
    manager: ExperimentManager,
    verbose: bool,
    console: Console,
) -> None:
    """Execute staged experiments sequentially (original behavior)."""
    if verbose:
        console.print(f"[dim]Found {len(staged_experiments)} staged experiments[/]")

    for experiment_id in staged_experiments:
        try:
            if verbose:
                console.print(f"[dim]Executing staged experiment: {experiment_id}[/]")

            # Load experiment metadata
            metadata = manager.storage.load_metadata(experiment_id)
            config = manager.storage.load_config(experiment_id)
            script_path = Path(metadata["script_path"])

            # Transition to running state
            manager.execute_staged_experiment(experiment_id)

            # Execute the script
            _execute_staged_script(
                experiment_id=experiment_id,
                script_path=script_path,
                config=config,
                manager=manager,
                verbose=verbose,
            )

        except Exception as e:
            console.print(
                f"[red]✗ Failed to execute staged experiment {experiment_id}: {e}[/]"
            )
            try:
                manager.fail_experiment(
                    experiment_id, f"Staged execution failed: {str(e)}"
                )
            except Exception:
                pass  # Best effort to record failure


def _execute_staged_parallel(
    staged_experiments: list[str],
    manager: ExperimentManager,
    verbose: bool,
    console: Console,
    max_workers: int,
) -> None:
    """Execute staged experiments in parallel using multiprocessing."""
    console.print(
        f"[dim]Found {len(staged_experiments)} staged experiments[/]"
    )
    console.print(
        f"[dim]Executing with {max_workers} parallel workers[/]"
    )

    # Pre-load all experiment data before forking
    experiment_data = []
    for experiment_id in staged_experiments:
        try:
            metadata = manager.storage.load_metadata(experiment_id)
            config = manager.storage.load_config(experiment_id)
            experiment_data.append({
                "experiment_id": experiment_id,
                "script_path": Path(metadata["script_path"]),
                "config": config,
            })
        except Exception as e:
            console.print(
                f"[red]✗ Failed to load experiment {experiment_id}: {e}[/]"
            )

    # Track results
    completed = 0
    failed = 0

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                _execute_single_experiment_worker,
                exp_data["experiment_id"],
                exp_data["script_path"],
                exp_data["config"],
                verbose,
            ): exp_data["experiment_id"]
            for exp_data in experiment_data
        }

        # Process results as they complete
        for future in as_completed(future_to_exp):
            experiment_id = future_to_exp[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                    console.print(
                        f"[green]✓ Experiment completed: {experiment_id}[/]"
                    )
                else:
                    failed += 1
                    console.print(
                        f"[red]✗ Experiment failed: {experiment_id}[/]"
                    )
            except Exception as e:
                failed += 1
                console.print(
                    f"[red]✗ Experiment error: {experiment_id}: {e}[/]"
                )

    # Summary
    console.print(f"\n[bold]Execution Summary:[/]")
    console.print(f"  Total: {len(experiment_data)}")
    console.print(f"  [green]Completed: {completed}[/]")
    console.print(f"  [red]Failed: {failed}[/]")


def _execute_single_experiment_worker(
    experiment_id: str,
    script_path: Path,
    config: dict[str, Any],
    verbose: bool,
) -> bool:
    """Worker function for parallel experiment execution.

    This runs in a separate process, so it needs to create its own manager.

    Returns:
        True if experiment succeeded, False otherwise
    """
    # Create fresh manager in this process
    manager = ExperimentManager()

    try:
        # Transition to running state
        manager.execute_staged_experiment(experiment_id)

        # Execute the script
        _execute_staged_script(
            experiment_id=experiment_id,
            script_path=script_path,
            config=config,
            manager=manager,
            verbose=verbose,
        )

        return True

    except Exception as e:
        # Record failure
        try:
            manager.fail_experiment(
                experiment_id, f"Parallel execution failed: {str(e)}"
            )
        except Exception:
            pass  # Best effort

        return False
```

### Phase 3: Handle Edge Cases & Race Conditions

#### 3.1 Add PID Tracking (Optional but Recommended)
**File**: `yanex/core/manager.py` - in `start_experiment()` at line 130

**Purpose**: Help with debugging and cleanup of orphaned experiments

```python
def start_experiment(self, experiment_id: str) -> None:
    """Transition experiment to running state."""
    # ... existing validation ...

    # Update status and timestamps
    import os

    now = datetime.utcnow().isoformat()
    metadata["status"] = "running"
    metadata["started_at"] = now
    metadata["process_id"] = os.getpid()  # NEW: Track which process is running this

    # Save updated metadata
    self.storage.save_metadata(experiment_id, metadata)
```

This helps with:
- Debugging which process is running which experiment
- Future cleanup scripts for orphaned experiments
- Process monitoring and resource tracking

### Phase 4: Update Tests

#### 4.1 Test Parallel Execution
**New file**: `tests/cli/test_parallel_execution.py`

```python
"""Tests for parallel experiment execution."""

import pytest
from pathlib import Path
from click.testing import CliRunner
from yanex.cli.main import cli


class TestParallelExecution:
    """Test parallel experiment execution functionality."""

    def test_parallel_flag_requires_staged(self, temp_dir, sample_experiment_script):
        """Test that --parallel requires --staged flag."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", str(sample_experiment_script), "--parallel", "2"]
        )
        assert result.exit_code != 0
        assert "--parallel flag can only be used with --staged" in result.output

    def test_parallel_negative_value_rejected(self):
        """Test that negative --parallel values are rejected."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--staged", "--parallel", "-1"])
        assert result.exit_code != 0
        assert "--parallel must be 0 (auto) or positive" in result.output

    def test_parallel_zero_auto_detects_cpus(self, monkeypatch):
        """Test that --parallel 0 auto-detects CPU count."""
        # Mock multiprocessing.cpu_count()
        import multiprocessing
        monkeypatch.setattr(multiprocessing, "cpu_count", lambda: 8)

        # Test would verify that 8 workers are used
        pass

    def test_parallel_execution_multiple_experiments(
        self, temp_dir, sample_experiment_script, monkeypatch
    ):
        """Test parallel execution of multiple staged experiments."""
        # Set up multiple staged experiments
        # Execute with --parallel 2
        # Verify all completed successfully
        pass

    def test_parallel_execution_handles_failures(
        self, temp_dir, monkeypatch
    ):
        """Test that parallel execution handles experiment failures gracefully."""
        # Create experiments that will fail
        # Verify failed experiments are marked as failed
        # Verify other experiments still complete
        pass
```

#### 4.2 Update Existing Tests
**File**: `tests/core/test_manager.py`

**Add tests for `allow_parallel` parameter**:
```python
def test_prevent_concurrent_execution_with_allow_parallel(self, isolated_manager):
    """Test prevent_concurrent_execution skips check when allow_parallel=True."""
    # Create running experiment
    experiment_id = "running123"
    exp_dir = isolated_manager.experiments_dir / experiment_id
    exp_dir.mkdir(parents=True)

    metadata = TestDataFactory.create_experiment_metadata(
        experiment_id=experiment_id, status="running"
    )
    isolated_manager.storage.save_metadata(experiment_id, metadata)

    # Should NOT raise with allow_parallel=True
    isolated_manager.prevent_concurrent_execution(allow_parallel=True)

def test_get_running_experiments_empty(self, isolated_manager):
    """Test get_running_experiments returns empty list when no running experiments."""
    result = isolated_manager.get_running_experiments()
    assert result == []

def test_get_running_experiments_multiple(self, isolated_manager):
    """Test get_running_experiments returns all running experiments."""
    # Create 3 experiments: 2 running, 1 completed
    running_ids = ["run001", "run002"]
    completed_id = "comp001"

    for exp_id in running_ids:
        exp_dir = isolated_manager.experiments_dir / exp_id
        exp_dir.mkdir(parents=True)
        metadata = TestDataFactory.create_experiment_metadata(
            experiment_id=exp_id, status="running"
        )
        isolated_manager.storage.save_metadata(exp_id, metadata)

    # Create completed experiment
    exp_dir = isolated_manager.experiments_dir / completed_id
    exp_dir.mkdir(parents=True)
    metadata = TestDataFactory.create_experiment_metadata(
        experiment_id=completed_id, status="completed"
    )
    isolated_manager.storage.save_metadata(completed_id, metadata)

    # Should return only running experiments
    result = isolated_manager.get_running_experiments()
    assert set(result) == set(running_ids)
```

### Phase 5: Documentation Updates

#### 5.1 Update CLAUDE.md
Add to "Two Execution Patterns" section:

```markdown
**Parallel Execution:**
- Use `yanex run --staged --parallel N` to run N experiments concurrently
- `--parallel 0` uses auto-detection (number of CPU cores)
- Each experiment runs in isolated process with separate storage
- Useful for parameter sweeps and batch processing on multi-core systems
- Example: `yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --stage && yanex run --staged --parallel 4`
```

#### 5.2 Update CLI Help and Docs
**File**: `docs/commands/run.md`

Add section on parallel execution:

```markdown
### Parallel Execution

For running multiple experiments simultaneously on multi-core systems:

```bash
# Execute staged experiments in parallel with 4 workers
yanex run --staged --parallel 4

# Auto-detect CPU count
yanex run --staged --parallel 0

# Complete workflow: stage parameter sweep, then execute in parallel
yanex run train.py --param "lr=linspace(0.001, 0.1, 10)" --stage
yanex run --staged --parallel 4
```

**Notes:**
- Parallel execution only works with `--staged` flag
- Each experiment runs in a separate process with isolated resources
- Useful for parameter sweeps and batch processing
- Monitor system resources to avoid overloading (memory, GPU)
```

#### 5.3 Add Examples
**New file**: `examples/parallel_execution.md`

Complete example showing parameter sweep with parallel execution.

## Risk Assessment

### Low Risk ✅
- Isolated experiment directories (no shared state between experiments)
- File-based storage already has per-experiment isolation
- Backward compatible (sequential is default)
- Each experiment has unique ID and directory

### Medium Risk ⚠️
- Race condition in `get_running_experiment()` during status check (rare, low impact)
- Subprocess output interleaving in terminal (cosmetic issue - each experiment writes to its own stdout.txt)
- Memory usage with many parallel experiments (user responsibility to set appropriate --parallel N)
- Process cleanup on Ctrl+C (multiprocessing handles this)

### High Risk ❌
- None identified

### Mitigation Strategies
1. ✅ Keep sequential execution as default (requires explicit `--parallel`)
2. ✅ Add PID tracking to metadata for debugging
3. ✅ Use ProcessPoolExecutor for proper process management
4. ✅ Capture experiment output to files (already done in ScriptExecutor)
5. Future: Add `--max-parallel` global config option
6. Future: Add resource monitoring warnings if too many experiments requested

## Implementation Notes

### Why ProcessPoolExecutor over ThreadPoolExecutor?
- Python's GIL prevents true parallelism with threads
- Neural network training is CPU/GPU intensive, needs separate processes
- ProcessPoolExecutor provides isolation (one crash doesn't affect others)

### Why pre-load experiment data?
- Avoid pickling issues with complex objects
- Cleaner error handling before spawning processes
- Faster startup for worker processes

### Output Handling
- Each experiment's output is captured to `stdout.txt` and `stderr.txt` in its directory
- Terminal output from parallel runs will interleave, but that's expected
- Consider adding `--quiet` flag in future to suppress terminal output during parallel runs

## Testing Strategy

1. **Unit Tests**: Test new functions in isolation
   - `prevent_concurrent_execution(allow_parallel=True)`
   - `get_running_experiments()`
   - CLI flag validation

2. **Integration Tests**: Test parallel execution with 2-3 small experiments
   - Create simple test scripts that sleep and log
   - Verify parallel execution (check timestamps)
   - Verify all experiments complete

3. **Load Tests**: Test with 10+ experiments to verify stability
   - Monitor memory usage
   - Verify no deadlocks or hangs

4. **Edge Cases**: Test error scenarios
   - Ctrl+C interruption during parallel execution
   - Failed experiments don't block others
   - Mixed success/failure scenarios

## Rollout Plan

1. ✅ Create implementation plan (this document)
2. Implement Phase 1 (remove restrictions) - backward compatible
3. Implement Phase 2 (parallel execution) - feature flagged with `--parallel`
4. Implement Phase 3 (PID tracking) - debugging support
5. Comprehensive testing (Phase 4)
6. Update documentation (Phase 5)
7. Run full test suite and ruff checks
8. Commit changes
9. Release as v0.5.0 with experimental parallel execution feature
10. Gather feedback and iterate

## Estimated Effort

- **Phase 1 (Core Changes)**: 1-2 hours
- **Phase 2 (Parallel Execution)**: 3-4 hours
- **Phase 3 (PID Tracking)**: 0.5 hours
- **Phase 4 (Testing)**: 3-4 hours
- **Phase 5 (Documentation)**: 1-2 hours
- **Testing & Debugging**: 2-3 hours
- **Total**: ~11-15 hours

## Open Questions

1. **Default behavior**: Should `--parallel 0` (auto-detect) be the default when `--staged` is used, or keep sequential as default?
   - **Recommendation**: Keep sequential as default for safety and backward compatibility

2. **Progress display**: Do you want a progress bar or real-time status updates during parallel execution?
   - **Current**: Simple completion messages as experiments finish
   - **Future**: Could add rich progress bar showing N/M completed

3. **Resource limits**: Should we add warnings/limits if user requests too many parallel workers?
   - **Current**: User responsible for setting appropriate N
   - **Future**: Could warn if N > CPU count * 2

4. **Output handling**: Should we capture and display experiment output differently in parallel mode?
   - **Current**: Each experiment writes to its own stdout.txt/stderr.txt
   - **Terminal output**: Will interleave (expected for parallel execution)
   - **Future**: Add `--quiet` flag to suppress terminal output

## Success Criteria

- [ ] User can run `yanex run --staged --parallel 4` to execute 4 experiments concurrently
- [ ] Sequential execution still works (backward compatible)
- [ ] All existing tests pass
- [ ] New tests for parallel execution pass
- [ ] Documentation updated
- [ ] No ruff errors
- [ ] Test coverage maintained at 90%+

## References

- Python ProcessPoolExecutor: https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor
- Multiprocessing best practices: https://docs.python.org/3/library/multiprocessing.html
- Issue/Discussion: (add link if applicable)
