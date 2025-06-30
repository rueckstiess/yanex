# API Cleanup Implementation Plan

## Overview

Clean up the Python API to resolve confusion between CLI-driven and explicit experiment creation patterns. Remove broken functions, add conflict detection, and update documentation to reflect the actual working design.

## Code Changes Required

### 1. Remove Broken `experiment.run()` Function

**File:** `yanex/experiment.py`
- **Action:** Remove the `run()` function entirely
- **Current:** Raises `NotImplementedError`
- **Reason:** Never implemented, causes confusion vs `create_experiment()`

### 2. Add CLI Context Detection

**File:** `yanex/experiment.py`
- **Action:** Add function to detect if running in CLI context
- **Implementation:**
  ```python
  def _is_cli_context() -> bool:
      """Check if currently running in a yanex CLI-managed experiment."""
      # Implementation options:
      # - Check environment variable set by yanex run
      # - Check if ExperimentManager has active experiment
      # - Check thread-local state
      return bool(os.environ.get('YANEX_CLI_ACTIVE'))
  ```

**File:** `yanex/cli/commands/run.py`  
- **Action:** Set environment variable when running experiments
- **Implementation:** Set `YANEX_CLI_ACTIVE=1` before executing user script

### 3. Add Conflict Detection to `create_experiment()`

**File:** `yanex/experiment.py`
- **Action:** Modify `create_experiment()` to check for CLI context
- **Implementation:**
  ```python
  def create_experiment(...):
      if _is_cli_context():
          raise ExperimentError(
              "Cannot use experiment.create_experiment() when script is run via 'yanex run'. "
              "Either:\n"
              "  - Run directly: python script.py\n" 
              "  - Or remove experiment.create_experiment() and use: yanex run script.py"
          )
      # ... existing implementation
  ```

### 4. Update Exception Classes

**File:** `yanex/utils/exceptions.py`
- **Action:** Add `ExperimentError` if it doesn't exist
- **Purpose:** Clear error type for API misuse

### 5. Update Examples

**File:** `examples/basic_usage.py`
- **Action:** Review to ensure it demonstrates the correct primary pattern
- **Check:** Uses `experiment.get_params()` and `experiment.log_results()` without explicit context manager

**File:** `examples/manual_control.py`  
- **Action:** Ensure it demonstrates `experiment.create_experiment()` pattern
- **Check:** Shows explicit experiment creation for advanced use cases

## Documentation Changes Required

### 1. Main README.md

**Current Issues:**
- Shows `with experiment.run():` pattern (broken)
- Emphasizes Python API over CLI-first approach

**Changes Needed:**
- **Replace all `experiment.run()` examples** with CLI-first pattern:
  ```python
  # OLD (broken)
  with experiment.run():
      experiment.log_results({"accuracy": accuracy})
  
  # NEW (correct)
  from yanex import experiment
  
  params = experiment.get_params()
  accuracy = train_model(params)
  experiment.log_results({"accuracy": accuracy})
  ```

- **Lead with CLI usage:**
  ```bash
  yanex run script.py --param learning_rate=0.01
  ```

- **Move Python API to advanced section:**
  - Show `experiment.create_experiment()` as secondary pattern
  - Explain when to use explicit vs CLI-driven experiments

### 2. docs/python-api.md

**Current Issues:**
- Extensive documentation of non-existent `experiment.run()` API
- Confusing mix of working and broken patterns

**Changes Needed:**
- **Remove entire `experiment.run()` section** (~200 lines)
- **Update Quick Reference:**
  ```python
  # PRIMARY PATTERN - CLI-driven
  from yanex import experiment
  
  params = experiment.get_params()
  accuracy = train_model(params)
  experiment.log_results({"accuracy": accuracy})
  
  # ADVANCED PATTERN - Explicit control  
  with experiment.create_experiment() as exp:
      exp.log_results({"accuracy": 0.95})
  ```

- **Restructure sections:**
  1. CLI-driven experiments (primary)
  2. Explicit experiment creation (advanced)
  3. Parameter management (unchanged)
  4. Integration patterns (updated)

### 3. docs/cli-commands.md

**Changes Needed:**
- **Update `yanex run` examples** to show the correct script pattern
- **Remove references to `experiment.run()` context manager**
- **Emphasize that scripts work both standalone and via CLI**

### 4. docs/commands/run.md

**Changes Needed:**
- **Fix script integration section** - remove `experiment.run()` examples
- **Show correct pattern:**
  ```python
  # train.py - works both ways
  from yanex import experiment
  
  params = experiment.get_params()
  
  # Your training code here
  accuracy = train_model(params)
  
  # Logging works in both contexts
  experiment.log_results({"accuracy": accuracy})
  ```

### 5. docs/best-practices.md

**Changes Needed:**
- **Update all code examples** to use correct API
- **Add section on when to use each approach:**
  - CLI-driven: Most experiments, production, team workflows
  - Explicit: Notebooks, parameter sweeps, fine control
- **Add anti-pattern:** Don't mix `yanex run` with `experiment.create_experiment()`

### 6. docs/configuration.md

**Changes Needed:**
- **Update code examples** to remove `experiment.run()` references
- **Keep parameter management examples** (these are correct)

## Testing Changes Required

### 1. Update Existing Tests

**Files:** `tests/test_experiment.py`, etc.
- **Action:** Remove tests for `experiment.run()` function
- **Action:** Add tests for CLI context detection
- **Action:** Add tests for conflict detection in `create_experiment()`

### 2. Add New Tests

**Test CLI Context Detection:**
```python
def test_cli_context_detection():
    # Test without CLI context
    assert not _is_cli_context()
    
    # Test with CLI context
    os.environ['YANEX_CLI_ACTIVE'] = '1'
    assert _is_cli_context()
```

**Test Conflict Detection:**
```python
def test_create_experiment_conflict():
    os.environ['YANEX_CLI_ACTIVE'] = '1'
    
    with pytest.raises(ExperimentError, match="Cannot use experiment.create_experiment"):
        experiment.create_experiment()
```

## Implementation Priority

### Phase 1 - Core API Fixes
1. Remove `experiment.run()` function
2. Add CLI context detection
3. Add conflict detection to `create_experiment()`
4. Update exception classes

### Phase 2 - Examples and Tests  
1. Fix examples to use correct patterns
2. Add tests for new functionality
3. Update existing tests

### Phase 3 - Documentation
1. Update README.md (main entry point)
2. Fix docs/python-api.md (major rewrite needed)
3. Update other documentation files
4. Verify all examples work

## Validation Checklist

After implementation, verify:

- ✅ `python script.py` works (no experiment created)
- ✅ `yanex run script.py` works (creates experiment)  
- ✅ `experiment.create_experiment()` works in standalone context
- ✅ `experiment.create_experiment()` raises clear error in CLI context
- ✅ All documentation examples work as shown
- ✅ No references to broken `experiment.run()` remain
- ✅ Examples demonstrate correct patterns

## Files Summary

**Code Changes:**
- `yanex/experiment.py` - Remove run(), add detection, add conflict check
- `yanex/cli/commands/run.py` - Set CLI context flag
- `yanex/utils/exceptions.py` - Add ExperimentError
- `examples/*.py` - Fix to use correct patterns

**Documentation Changes:**
- `README.md` - Major rewrite of examples
- `docs/python-api.md` - Remove run() section, restructure
- `docs/cli-commands.md` - Fix run examples
- `docs/commands/run.md` - Fix script integration
- `docs/best-practices.md` - Update examples, add guidance
- `docs/configuration.md` - Fix code examples

**Test Changes:**
- Remove tests for broken functionality
- Add tests for new functionality
- Verify integration works correctly