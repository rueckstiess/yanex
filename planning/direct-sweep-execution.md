# Design: Direct Parameter Sweep Execution

**Status**: Design Phase
**Created**: 2025-01-23
**Target Version**: v0.6.0
**Related**: Builds on parallel execution from v0.5.0

## Executive Summary

Enable parameter sweeps to execute immediately without requiring the `--stage` flag, supporting both sequential and parallel execution modes. This streamlines the workflow for users who want to run sweeps directly without the two-step stage-then-execute process.

## Motivation

**Current workflow (cumbersome):**
```bash
# Step 1: Stage experiments
yanex run exp.py --param "mode=list(on,off,hybrid)" --stage

# Step 2: Execute staged experiments
yanex run --staged --parallel 2
```

**Problem:**
- Two-step process is tedious for immediate execution
- Users must remember to run `--staged` after staging
- Cannot specify parallelism at sweep definition time

**Desired workflow (streamlined):**
```bash
# Run sweep sequentially, immediately
yanex run exp.py --param "mode=list(on,off,hybrid)"

# Run sweep in parallel, immediately
yanex run exp.py --param "mode=list(on,off,hybrid)" --parallel 2

# Stage for later (existing behavior preserved)
yanex run exp.py --param "mode=list(on,off,hybrid)" --stage
```

## Current Architecture Analysis

### Key Components

**1. Sweep Detection** (`yanex/core/config.py`)
```python
def has_sweep_parameters(config: dict[str, Any]) -> bool:
    """Detect if config contains sweep parameters (range, linspace, etc.)"""
```

**2. Sweep Validation** (`yanex/cli/_utils.py:102`)
```python
def validate_sweep_requirements(config: dict[str, Any], stage_flag: bool) -> None:
    """Validate that parameter sweeps are used with --stage flag."""
    if has_sweep_parameters(config) and not stage_flag:
        raise click.ClickException(
            "Parameter sweeps require --stage flag to avoid accidental batch execution."
        )
```
- **Current behavior**: BLOCKS all sweeps without `--stage`
- **Need to change**: Allow sweeps without `--stage` for direct execution

**3. Sweep Expansion** (`yanex/core/config.py`)
```python
def expand_parameter_sweeps(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Expand sweep parameters into individual configurations."""
```

**4. Staging Function** (`yanex/cli/commands/run.py:336`)
```python
def _stage_experiment(...):
    """Stage experiment(s) for later execution, expanding parameter sweeps."""
    if has_sweep_parameters(config):
        expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)
        # Create experiments with stage_only=True
```

**5. Single Experiment Execution** (`yanex/cli/commands/run.py:226`)
```python
def _execute_experiment(...):
    """Execute script as an experiment with proper lifecycle management."""
    # Create experiment (status="created")
    # Start experiment (status="running")
    # Execute script
    # Complete experiment
```

### Key Constraint

**Cannot use staging mechanism internally** because:
- There might be existing staged experiments
- Running sweep would inadvertently execute those too
- Staging is meant for deferred execution, not internal implementation detail

## Proposed Design

### Architecture Overview

```
User Command
    ↓
Sweep Detection
    ↓
    ├─ --stage? → Stage experiments (existing behavior)
    │              Status: "staged"
    │              Execute later with: yanex run --staged
    │
    └─ No --stage → Direct Execution (NEW)
                    ├─ --parallel N? → Parallel execution
                    └─ No --parallel → Sequential execution

                    Status flow: "created" → "running" → "completed"/"failed"
                    No "staged" status used
```

### Implementation Strategy

**Option 1: In-memory sweep execution (RECOMMENDED)**

Create experiments just-in-time without using "staged" status:

```python
def _execute_sweep_experiments(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
    max_workers: int | None = None,  # None=sequential, N=parallel
) -> None:
    """Execute parameter sweep directly (sequential or parallel).

    This creates and executes experiments on-the-fly without using
    the "staged" status, avoiding interference with existing staged experiments.
    """
    manager = ExperimentManager()

    # Expand parameter sweeps into individual configurations
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    click.echo(
        f"✓ Parameter sweep detected: running {len(expanded_configs)} experiments"
    )

    if max_workers is None:
        # Sequential execution
        _execute_sweep_sequential(
            script, name, tags, description, expanded_configs,
            sweep_param_paths, manager, verbose, ignore_dirty
        )
    else:
        # Parallel execution
        _execute_sweep_parallel(
            script, name, tags, description, expanded_configs,
            sweep_param_paths, manager, verbose, ignore_dirty, max_workers
        )


def _execute_sweep_sequential(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    expanded_configs: list[dict[str, Any]],
    sweep_param_paths: list[str],
    manager: ExperimentManager,
    verbose: bool,
    ignore_dirty: bool,
) -> None:
    """Execute sweep experiments sequentially."""
    for i, expanded_config in enumerate(expanded_configs):
        try:
            # Generate descriptive name for each sweep experiment
            sweep_name = _generate_sweep_experiment_name(
                name, expanded_config, sweep_param_paths
            )

            if verbose:
                click.echo(
                    f"[{i+1}/{len(expanded_configs)}] Starting: {sweep_name}"
                )

            # Create and execute immediately (NOT staged)
            experiment_id = manager.create_experiment(
                script_path=script,
                name=sweep_name,
                config=expanded_config,
                tags=tags,
                description=description,
                allow_dirty=ignore_dirty,
                stage_only=False,  # Create as "created", not "staged"
            )

            # Start experiment
            manager.start_experiment(experiment_id)

            # Execute script
            executor = ScriptExecutor(manager)
            executor.execute_script(experiment_id, script, expanded_config, verbose)

            click.echo(f"  ✓ Completed: {experiment_id}")

        except Exception as e:
            click.echo(f"  ✗ Failed: {e}", err=True)
            # Continue with next experiment

    click.echo(f"\n✓ Sweep execution completed: {len(expanded_configs)} experiments")


def _execute_sweep_parallel(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    expanded_configs: list[dict[str, Any]],
    sweep_param_paths: list[str],
    manager: ExperimentManager,
    verbose: bool,
    ignore_dirty: bool,
    max_workers: int,
) -> None:
    """Execute sweep experiments in parallel."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing

    if max_workers == 0:
        max_workers = multiprocessing.cpu_count()

    click.echo(f"  Executing with {max_workers} parallel workers")

    # Pre-generate experiment data (names, configs)
    experiment_data = []
    for expanded_config in expanded_configs:
        sweep_name = _generate_sweep_experiment_name(
            name, expanded_config, sweep_param_paths
        )
        experiment_data.append({
            "name": sweep_name,
            "config": expanded_config,
            "script": script,
            "tags": tags,
            "description": description,
            "ignore_dirty": ignore_dirty,
        })

    # Track results
    completed = 0
    failed = 0

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                _execute_single_sweep_experiment,
                exp_data["script"],
                exp_data["name"],
                exp_data["tags"],
                exp_data["description"],
                exp_data["config"],
                verbose,
                exp_data["ignore_dirty"],
            ): exp_data["name"]
            for exp_data in experiment_data
        }

        # Process results as they complete
        for future in as_completed(future_to_exp):
            exp_name = future_to_exp[future]
            try:
                success, experiment_id = future.result()
                if success:
                    completed += 1
                    click.echo(f"  ✓ Completed: {experiment_id} ({exp_name})")
                else:
                    failed += 1
                    click.echo(f"  ✗ Failed: {exp_name}")
            except Exception as e:
                failed += 1
                click.echo(f"  ✗ Error: {exp_name}: {e}")

    # Summary
    click.echo(f"\n✓ Sweep execution completed")
    click.echo(f"  Total: {len(experiment_data)}")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Failed: {failed}")


def _execute_single_sweep_experiment(
    script: Path,
    name: str,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool,
    ignore_dirty: bool,
) -> tuple[bool, str]:
    """Worker function for parallel sweep experiment execution.

    Returns:
        (success, experiment_id) tuple
    """
    manager = ExperimentManager()

    try:
        # Create experiment
        experiment_id = manager.create_experiment(
            script_path=script,
            name=name,
            config=config,
            tags=tags,
            description=description,
            allow_dirty=ignore_dirty,
            stage_only=False,  # NOT staged
        )

        # Start experiment
        manager.start_experiment(experiment_id)

        # Execute script
        executor = ScriptExecutor(manager)
        executor.execute_script(experiment_id, script, config, verbose)

        return (True, experiment_id)

    except Exception as e:
        # Try to mark as failed if experiment was created
        try:
            manager.fail_experiment(experiment_id, f"Sweep execution failed: {str(e)}")
        except Exception:
            pass

        return (False, "")
```

### Changes Required

**1. Modify validation** (`yanex/cli/_utils.py`)

```python
def validate_sweep_requirements(
    config: dict[str, Any],
    stage_flag: bool,
    parallel_flag: int | None,
) -> None:
    """
    Validate parameter sweep usage with execution flags.

    Args:
        config: Configuration dictionary to check
        stage_flag: Whether --stage flag was provided
        parallel_flag: Value of --parallel flag (None if not provided)

    Valid combinations:
        - sweep + --stage: Stage for later execution ✓
        - sweep + --parallel N: Execute N in parallel immediately ✓
        - sweep + no flags: Execute sequentially immediately ✓

    No restrictions needed - all combinations are now valid!
    """
    # No validation needed - sweeps are now allowed in all modes
    pass  # Keep function for backward compatibility, or remove entirely
```

**Actually, we can simplify this further - just remove the validation entirely!**

**2. Update run command logic** (`yanex/cli/commands/run.py`)

```python
def run(...):
    """Run a script as a tracked experiment."""

    # ... existing validation ...

    # Phase 3: Execute, stage, or execute sweep
    if resolved_stage:
        # Stage for later execution (existing behavior)
        _stage_experiment(...)
    elif has_sweep_parameters(experiment_config):
        # NEW: Direct sweep execution
        _execute_sweep_experiments(
            script=script,
            name=resolved_name,
            tags=resolved_tags,
            description=resolved_description,
            config=experiment_config,
            verbose=verbose,
            ignore_dirty=resolved_ignore_dirty,
            max_workers=parallel,  # None=sequential, N=parallel
        )
    else:
        # Single experiment execution (existing behavior)
        _execute_experiment(...)
```

**3. Add imports**

```python
from ..core.script_executor import ScriptExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
```

## Behavior Matrix

| Command | Sweep? | --stage? | --parallel? | Behavior |
|---------|--------|----------|-------------|----------|
| `yanex run exp.py` | No | No | No | Execute single experiment (existing) |
| `yanex run exp.py --parallel 2` | No | No | Yes | Error: --parallel requires sweep or --staged |
| `yanex run exp.py --param "x=list(1,2,3)"` | Yes | No | No | Execute 3 experiments sequentially (NEW) |
| `yanex run exp.py --param "x=list(1,2,3)" --parallel 2` | Yes | No | Yes | Execute 3 experiments, 2 in parallel (NEW) |
| `yanex run exp.py --param "x=list(1,2,3)" --stage` | Yes | Yes | No | Stage 3 experiments for later (existing) |
| `yanex run --staged` | N/A | N/A | No | Execute all staged sequentially (existing) |
| `yanex run --staged --parallel 2` | N/A | N/A | Yes | Execute all staged, 2 in parallel (v0.5.0) |

### Validation Rules

**New validation needed:**
- `--parallel` with single experiment (no sweep, no --staged) → Error
- `--stage` with `--parallel` → Error (doesn't make sense - staging is deferred)

```python
# In run command, after existing validations:

# Validate parallel flag with non-sweep single experiments
if parallel is not None and not staged and not has_sweep_parameters(experiment_config):
    click.echo(
        "Error: --parallel can only be used with parameter sweeps or --staged",
        err=True
    )
    raise click.Abort()

# Validate stage + parallel combination
if stage and parallel is not None:
    click.echo(
        "Error: --parallel cannot be used with --stage. "
        "Stage experiments first, then run with: yanex run --staged --parallel N",
        err=True
    )
    raise click.Abort()
```

## Benefits

### User Experience
✅ **Streamlined workflow**: One command instead of two
✅ **Intuitive**: `--parallel` flag at sweep definition time
✅ **Flexible**: Choose sequential or parallel at run time
✅ **Backward compatible**: `--stage` still works as before

### Technical
✅ **Clean implementation**: No "staged" status pollution
✅ **Reuses existing parallel infrastructure**: From v0.5.0
✅ **No interference**: Doesn't affect existing staged experiments
✅ **Consistent**: Same execution path as single experiments

## Migration Path

**No breaking changes!**

- Existing `--stage` workflow still works exactly as before
- Users can gradually adopt new direct execution style
- Documentation shows both patterns as valid

## Testing Strategy

### Unit Tests

**New tests in `tests/cli/test_direct_sweep_execution.py`:**

```python
class TestDirectSweepExecution:
    """Test direct parameter sweep execution without staging."""

    def test_sweep_sequential_without_stage(self):
        """Test that sweeps run sequentially without --stage flag."""

    def test_sweep_parallel_without_stage(self):
        """Test that sweeps run in parallel with --parallel flag."""

    def test_sweep_parallel_auto_detect_cpus(self):
        """Test --parallel 0 auto-detects CPU count."""

    def test_parallel_flag_rejects_single_experiment(self):
        """Test that --parallel errors with non-sweep single experiments."""

    def test_parallel_with_stage_rejects(self):
        """Test that --stage + --parallel is rejected."""

    def test_sweep_execution_creates_experiments_not_staged(self):
        """Test that direct sweep doesn't use 'staged' status."""

    def test_sweep_execution_doesnt_affect_existing_staged(self):
        """Test that existing staged experiments are unaffected."""
```

### Integration Tests

- Run sweep with multiple parameters
- Verify all experiments created and executed
- Verify no "staged" status used
- Verify parallel vs sequential execution
- Verify error handling for failed sweep experiments

## Implementation Phases

### Phase 1: Validation Changes (30 min)
- Remove `validate_sweep_requirements` or make it no-op
- Add new validation for `--parallel` with single experiments
- Add validation for `--stage` + `--parallel` conflict

### Phase 2: Sequential Sweep Execution (2 hours)
- Implement `_execute_sweep_experiments`
- Implement `_execute_sweep_sequential`
- Integrate into run command logic
- Test sequential execution

### Phase 3: Parallel Sweep Execution (2 hours)
- Implement `_execute_sweep_parallel`
- Implement `_execute_single_sweep_experiment`
- Reuse parallel infrastructure from v0.5.0
- Test parallel execution

### Phase 4: Testing (3 hours)
- Unit tests for new functions
- Integration tests for sweep execution
- Test edge cases and error handling
- Verify no interference with staged experiments

### Phase 5: Documentation (1 hour)
- Update CLAUDE.md
- Update user documentation
- Add examples showing both workflows

**Total Estimated Effort: 8-9 hours**

## Success Criteria

- [ ] User can run `yanex run exp.py --param "x=list(1,2,3)"` without --stage
- [ ] Experiments execute sequentially by default
- [ ] User can run with `--parallel N` for parallel execution
- [ ] `--stage` workflow still works (backward compatible)
- [ ] No "staged" status used in direct execution
- [ ] Existing staged experiments are unaffected
- [ ] All tests pass
- [ ] Documentation updated

## Open Questions

1. **Error handling**: Should failed sweep experiments abort the rest or continue?
   - **Recommendation**: Continue with remaining experiments (same as staged execution)

2. **Output verbosity**: How much detail to show during sweep execution?
   - **Recommendation**: Show progress counter [1/10], [2/10], etc.

3. **Naming conflicts**: What if sweep generates duplicate names?
   - **Current behavior**: Names are generated with parameter values, should be unique
   - **Fallback**: Append experiment ID if name collision occurs

4. **Resource management**: Should we limit parallel workers based on system resources?
   - **Recommendation**: User responsibility, but consider warning if N > CPU count * 2

## References

- v0.5.0 Parallel Execution: `planning/parallel-execution.md`
- Current sweep implementation: `yanex/cli/commands/run.py:336`
- Parallel execution functions: `yanex/cli/commands/run.py:492`
