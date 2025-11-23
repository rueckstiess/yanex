# Parameter Access Tracking

**Status:** Planning - Implementation Ready
**Date:** 2025-11-23
**Context:** Track which parameters scripts actually use, store only accessed params instead of full config

## Background

### Current Limitation

Currently, yanex stores the entire config file for each experiment, even if the script only uses a subset of parameters. This creates several problems:

1. **Bloated storage** - Experiments store many unused parameters from shared configs
2. **Unclear dependencies** - Can't tell which parameters actually affected the experiment
3. **Conflict noise** - When using experiment dependencies, all params conflict even if not used by both scripts
4. **Poor introspection** - Results API returns all params, not just the ones that mattered

### Use Case

Typical shared config workflow with nested parameters:

```yaml
# shared_config.yaml
model:
  architecture:
    n_layers: 5
    n_hidden: 128
    activation: relu
  train:
    epochs: 20
    learning_rate: 0.001
    batch_size: 32
data:
  filepath: "dataset.json"
  train_split: 0.8
  val_split: 0.1
seed: 42
logging:
  verbose: true
  log_dir: "./logs"
```

**Problem:** `train.py` uses only `model.train.*` and `data.filepath`, but stores entire config.

**Goal:** Store only accessed parameters:
```yaml
# params.yaml (saved after execution)
model:
  train:
    epochs: 20
    learning_rate: 0.001
data:
  filepath: "dataset.json"
seed: 42
```

## Design Decisions

### 1. Tracked Dictionary Wrapper

Parameters are wrapped in a `TrackedDict` class that monitors access:

```python
class TrackedDict(dict):
    """Dictionary wrapper that tracks which keys are accessed during script execution"""

    def __init__(self, data, path=""):
        super().__init__(data)
        self._accessed_paths = set()  # Full paths: "model.train.learning_rate"
        self._path = path

    def __getitem__(self, key):
        # Mark as accessed
        full_path = f"{self._path}.{key}" if self._path else key
        self._accessed_paths.add(full_path)

        value = super().__getitem__(key)

        # Wrap nested dicts recursively
        if isinstance(value, dict) and not isinstance(value, TrackedDict):
            value = TrackedDict(value, path=full_path)
            super().__setitem__(key, value)

        return value
```

**Rationale:**
- Automatic tracking - no code changes required in existing scripts
- Preserves nested structure - tracks full paths like `model.train.epochs`
- Transparent - behaves exactly like a normal dict
- Recursive - handles arbitrary nesting depth

### 2. Dot-Notation Parameter Access

Support nested parameter access via dot notation:

```python
# Access nested config sections
arch = yanex.get_param("model.architecture")
# Returns: TrackedDict({"n_layers": 5, "n_hidden": 128, "activation": "relu"})
# Does NOT mark as accessed yet

# Access individual values
n_layers = arch["n_layers"]
# NOW marks "model.architecture.n_layers" as accessed

# Direct access also works
n_layers = yanex.get_param("model.architecture.n_layers")
# Marks "model.architecture.n_layers" as accessed
```

**Implementation:**
- `get_param(key)` splits on `.` and traverses nested dicts
- Returns `TrackedDict` for nested objects (allows further access tracking)
- Returns raw values for leaf nodes
- Marks each segment of the path as accessed

**Rationale:**
- Ergonomic for deeply nested configs
- Consistent with common config libraries (OmegaConf, hydra, etc.)
- Allows section-level access without tracking individual keys
- Fine-grained tracking only when values are actually used

### 3. Access Tracking Rules

**What counts as "accessed":**
- ✅ `params["key"]` - Direct item access
- ✅ `params.get("key")` - Get with optional default
- ✅ `params.get("key", default)` - Get with explicit default
- ✅ `for key in params.keys()` - Iterating keys marks all keys
- ✅ `for value in params.values()` - Iterating values marks all keys
- ✅ `for key, value in params.items()` - Iterating items marks all keys
- ✅ `yanex.get_param("model.train.epochs")` - Explicit get_param call

**What does NOT count as accessed:**
- ❌ `"key" in params` - Existence check (exploration, not usage)
- ❌ `len(params)` - Size check

**Rationale for tracking iterations:**
- Conservative approach - better to track too much than too little
- Iteration typically means the values are being used
- Can't track individual accesses within loop body after iteration started
- Edge case: user iterates but doesn't use values - acceptable over-tracking

### 4. Nested Path Tracking

Track full paths, preserve nested structure in storage:

```python
params = yanex.get_params()
# Config: {"model": {"train": {"lr": 0.01, "epochs": 20}}, "seed": 42}

lr = params["model"]["train"]["lr"]
# Marks accessed: "model", "model.train", "model.train.lr"

# Stored in params.yaml:
# model:
#   train:
#     lr: 0.01
```

**Implementation:**
- Each access adds full path to `_accessed_paths` set
- On save, reconstruct nested dict containing only accessed paths
- Preserve original structure (nested dicts, not flattened)

**Rationale:**
- Preserves human-readable YAML structure
- Compatible with existing config parsing
- Easy to understand what was accessed
- Supports tools expecting nested config format

### 5. Storage: params.yaml (rename from config.yaml)

**Before:**
- `~/.yanex/experiments/{exp_id}/config.yaml` - Full config (all params)
- Saved at experiment creation time

**After:**
- `~/.yanex/experiments/{exp_id}/params.yaml` - Only accessed params
- Saved at script end (normal exit, exception, or cancellation)

**File format - same as before, just filtered:**
```yaml
# Only parameters actually accessed during execution
model:
  train:
    learning_rate: 0.001
    epochs: 20
data:
  filepath: "dataset.json"
seed: 42
```

**Rationale:**
- Clear naming: `params.yaml` = actual parameters used
- YAML format maintained for human readability
- Saved at end allows tracking throughout script lifecycle
- Smaller files, clearer experiment documentation

### 6. Save Timing: Script Exit Handler

Parameters saved at script end to capture all accesses:

```python
# yanex/api.py or runner
import atexit

_tracked_params = None

def get_params():
    global _tracked_params
    if _tracked_params is None:
        raw_params = _load_params_from_config()
        _tracked_params = TrackedDict(raw_params)

        # Register cleanup on normal exit
        atexit.register(_save_accessed_params)

    return _tracked_params

def _save_accessed_params():
    """Extract and save only accessed parameters"""
    if _tracked_params is not None:
        accessed = _extract_accessed_params(_tracked_params)
        storage.save_params(experiment_id, accessed)
```

**CLI execution wrapper:**
```python
# In experiment runner
try:
    exec(script_code)
    # Normal exit - atexit will trigger
except KeyboardInterrupt:
    _save_accessed_params()  # Save before exit
    raise
except Exception as e:
    _save_accessed_params()  # Save even on failure
    raise
```

**Rationale:**
- `atexit` handles normal script completion
- Explicit calls in exception handlers catch all exit scenarios
- Ensures params saved even if script fails
- No changes required to script code

### 7. Conflict Detection with Dependencies

When experiment depends on other experiments, check for parameter conflicts:

```python
# At get_dependencies() call time
deps = yanex.get_dependencies()  # Returns list of Experiment objects

current_params = _tracked_params  # Current TrackedDict
for dep in deps:
    dep_params = dep.get_params()  # Load dependency's params.yaml

    # Find conflicts (same key path, different value)
    conflicts = _find_param_conflicts(current_params, dep_params)

    if conflicts:
        for path, (current_val, dep_val) in conflicts.items():
            warnings.warn(
                f"Parameter conflict with dependency {dep.id}: "
                f"{path} = {current_val} (current) vs {dep_val} (dependency)"
            )
```

**Behavior:**
- Check at `get_dependencies()` call (early warning)
- Warn but allow execution (non-blocking)
- Only checks accessed params from both experiments
- Ignores params accessed by only one experiment

**Rationale:**
- Early detection helps users catch configuration mistakes
- Non-blocking allows legitimate cases (e.g., different learning rates)
- Only checking accessed params avoids noise from unused config sections
- Clear warnings help debug unexpected results

### 8. Mixed API Access Pattern

Both `get_param()` and `get_params()` track in shared set:

```python
# All mark as accessed, stored in same _accessed_paths set
lr = yanex.get_param("learning_rate")  # Track: "learning_rate"
params = yanex.get_params()             # Get TrackedDict
bs = params["batch_size"]               # Track: "batch_size"
epochs = yanex.get_param("model.train.epochs")  # Track: "model.train.epochs"

# Final params.yaml contains all three
```

**Implementation:**
- Single global `_tracked_params` instance
- `get_param(key)` internally uses `_tracked_params[key]` (triggers tracking)
- `get_params()` returns `_tracked_params` directly
- Both APIs share same tracking state

**Rationale:**
- Unified tracking regardless of access method
- No confusion about which API to use
- Consistent behavior

### 9. Edge Cases and Limitations

**Copying creates untracked dict:**
```python
params = yanex.get_params()
my_copy = dict(params)  # Untracked copy
my_copy["key"]  # Not tracked ❌
```

**Acceptance:** Cannot prevent this, but rare in practice. Document as limitation.

**Workaround:** Use `params` directly, don't copy.

---

**Attribute access not supported:**
```python
params = yanex.get_params()
lr = params.learning_rate  # ❌ AttributeError
```

**Rationale:** Python dict doesn't support attribute access natively. Keeping `TrackedDict` as pure dict wrapper avoids complexity.

**Workaround:** Use bracket notation or `get_param("learning_rate")`.

---

**Serialization may break tracking:**
```python
import json
params = yanex.get_params()
json_str = json.dumps(params)  # Serializes but doesn't track
```

**Acceptance:** Serialization inherently creates copy. Acceptable limitation.

## Implementation Overview

### Core Modules to Create/Modify

**New modules:**
1. `yanex/core/tracked_dict.py` - TrackedDict implementation
2. `yanex/core/param_tracking.py` - Access tracking logic, save handler

**Modified modules:**
1. `yanex/api.py` - Update `get_params()`, `get_param()` to use TrackedDict
2. `yanex/core/manager.py` - Remove config save at creation, add atexit handler
3. `yanex/core/storage_*.py` - Rename config.yaml → params.yaml
4. `yanex/results/experiment.py` - Update `get_params()` to load params.yaml
5. `yanex/cli/commands/run.py` - Add exception handlers for saving params

### Storage Changes

**Experiment directory structure:**
```
~/.yanex/experiments/{exp_id}/
├── metadata.json
├── params.yaml           # NEW NAME (was config.yaml), only accessed params
├── dependencies.json
├── metrics.json
└── artifacts/
```

**Migration:**
- For backward compatibility, check for both `params.yaml` and `config.yaml`
- Prefer `params.yaml` if exists, fall back to `config.yaml`
- No automatic migration (users can manually rename if desired)

## API Changes

### Run API (yanex.*)

**Getting parameters (updated behavior):**

```python
# Get single parameter (marks as accessed)
lr = yanex.get_param("learning_rate")
# Returns: 0.001

# Get nested parameter with dot notation (new feature)
arch = yanex.get_param("model.architecture")
# Returns: TrackedDict({"n_layers": 5, "n_hidden": 128, ...})

n_layers = arch["n_layers"]
# Marks "model.architecture.n_layers" as accessed

# Direct nested access (also new)
n_layers = yanex.get_param("model.architecture.n_layers")
# Returns: 5, marks "model.architecture.n_layers" as accessed

# Get all parameters (marks all accessed if iterated)
params = yanex.get_params()
# Returns: TrackedDict

for key, value in params.items():
    # This iteration marks ALL keys as accessed
    print(f"{key}: {value}")
```

**Standalone mode:** Same behavior as before - loads from config file, no tracking.

### Results API (yanex.results.*)

**Loading parameters (unchanged API, updated storage):**

```python
from yanex.results import ResultsManager

manager = ResultsManager()
exp = manager.get_experiment("exp_id")

# Get parameters (loads from params.yaml)
params = exp.get_params()
# Returns: dict with only accessed parameters (not TrackedDict)

param_val = exp.get_param("learning_rate")
# Returns: value or None
```

**Note:** Results API returns plain dicts (no tracking needed - experiment already completed).

## Scope Clarifications

**Included in this feature:**
- ✅ TrackedDict wrapper for automatic access tracking
- ✅ Dot notation for nested parameter access
- ✅ Track iterations over .keys(), .values(), .items()
- ✅ Save only accessed parameters to params.yaml
- ✅ Save at script end (normal, exception, cancellation)
- ✅ Conflict detection with dependency experiments
- ✅ Backward compatibility (read config.yaml if params.yaml doesn't exist)
- ✅ Works with both get_param() and get_params() APIs

**Explicitly OUT OF SCOPE** (future work):
- ❌ Attribute access (`params.learning_rate`) - use bracket notation
- ❌ Tracking access frequency or timing - only binary accessed/not accessed
- ❌ Automatic migration of existing config.yaml to params.yaml
- ❌ Parameter provenance tracking (which line of code accessed)
- ❌ Required vs optional parameter validation

## References

- **Similar systems:** OmegaConf (structured configs), Hydra (config composition), MLflow (parameter logging)
- **Python stdlib:** `atexit` module for cleanup handlers, `signal` for interrupt handling
