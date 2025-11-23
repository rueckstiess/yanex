# Adding Lightweight Dependency Tracking to Yanex

## Context: What is Yanex

Yanex is a lightweight experiment tracking system for Python ML/research workflows. It's currently in alpha (v0.6.0) and provides:

**Current features:**
- Experiment tracking via `yanex run script.py` (CLI) or `yanex.create_experiment()` (Python API)
- Parameter management: Config files + CLI overrides, accessed via `yanex.get_params()`
- Metrics logging: `yanex.log_metrics({"loss": 0.5})`
- Artifact management: NEW unified API with `yanex.save_artifact()`, `yanex.load_artifact()`, `yanex.copy_artifact()`
  - Auto-detects format from extension (.json, .pt, .pkl, .csv, etc.)
  - Works in standalone mode (saves to `./artifacts/`) or tracked mode (saves to `~/.yanex/experiments/{id}/artifacts/`)
  - Results API: `exp.load_artifact("model.pt")` auto-loads artifacts from any experiment

**Storage:**
- Each experiment stored in `~/.yanex/experiments/{8-char-hex-id}/`
- Contains: `metadata.json`, `params.yaml`, `metrics.json`, `artifacts/` directory
- Experiments have states: created, running, completed, failed, cancelled, staged

**Experiment identification:**
- Each experiment gets unique 8-character hex ID (e.g., `abc12345`)
- Short ID resolution supported (4+ chars, e.g., `abc1` resolves to `abc12345`)

## The Problem

Currently, yanex experiments are completely isolated. There's no way to:

1. **Express relationships** between experiments in a multi-stage ML workflow
2. **Load artifacts from previous experiments** in a structured way
3. **Track which experiments depend on which** for reproducibility
4. **Validate that prerequisites completed** before running dependent experiments

**Typical ML workflow that's currently awkward:**

```bash
# Step 1: Prepare data
yanex run prepare_data.py
# Returns: abc12345 (creates train_data.jsonl, test_data.jsonl)

# Step 2: Train model (needs data from step 1)
yanex run train_model.py
# Problem: How does train_model.py know to use abc12345's artifacts?
# Currently: Manual - user copies ID, hardcodes it, or passes as script arg
```

**Current workaround (manual, error-prone):**

```python
# train_model.py
import yanex
from yanex.results import ResultsManager

# User manually provides data experiment ID (hardcoded or via script arg)
data_exp_id = "abc12345"  # Hardcoded - breaks reproducibility!

manager = ResultsManager()
data_exp = manager.get_experiment(data_exp_id)
train_data_path = data_exp.get_artifact("train_data.jsonl")

# Load and use data...
```

This is fragile, not reproducible, and loses the dependency relationship.

## Desired User Experience

### CLI Workflow

**Declare dependencies when running experiments:**

```bash
# Step 1: Prepare data (no dependencies)
yanex run prepare_data.py
# → abc12345

# Step 2: Train model (depends on data preparation)
yanex run train_model.py -D abc12345
# → def67890
# Stores that def67890 depends on abc12345

# Step 3: Evaluate (depends on training, which depends on data prep)
yanex run evaluate.py -D def67890
# → ghi23456
# Can access artifacts from def67890 AND abc12345 (transitive)
```

**Dependency sweeps (create multiple experiments, one per dependency):**

```bash
# Train on two different datasets
yanex run prepare_data.py --param "source=dataset_a"  # → prep1
yanex run prepare_data.py --param "source=dataset_b"  # → prep2

# Creates 2 training experiments (one per dataset)
yanex run train_model.py -D prep1,prep2
# → train1 (depends on prep1), train2 (depends on prep2)
```

### Python API (Within Scripts)

**Access artifacts from dependencies automatically:**

```python
# train_model.py (run with: yanex run train_model.py -D abc12345)
import yanex

# NEW: Load artifact - automatically searches current experiment AND dependencies
train_data = yanex.load_artifact("train_data.jsonl")
test_data = yanex.load_artifact("test_data.jsonl")

# If ambiguous (multiple experiments have same filename), error with helpful message
# If not found anywhere, returns None
```

**Access dependency metadata:**

```python
# evaluate.py (run with: yanex run evaluate.py -D train_id)
import yanex

# Load model from dependency
model = yanex.load_artifact("model_best.pt")

# Get dependency experiments (list of Experiment objects)
deps = yanex.get_dependencies()
# Returns: [Experiment(train_id)]

# Access dependency metadata
for dep in deps:
    print(f"Depends on {dep.id}: {dep.script_path}")
    print(f"Training params: {dep.get_params()}")
    print(f"Best accuracy: {dep.get_metric('best_accuracy')}")
```

## Design Decisions We've Made

After exploring different approaches, here's what we want:

### 1. **Lightweight CLI Declaration**
- Simple flag: `-D <experiment_id>` (or `--depends-on <experiment_id>`)
- Multiple dependencies: `-D id1 -D id2 -D id3`
- NO config file declaration required (current main branch has no dependency config syntax)
- NO named slots (e.g., `-D data=id1 -D model=id2`) - just a list of dependency IDs

### 2. **Automatic Artifact Resolution**
- `yanex.load_artifact("file.json")` searches: current experiment → direct dependencies → transitive dependencies
- If unique filename found: return loaded object
- If not found: return `None` (allows optional artifacts)
- If ambiguous (multiple deps have same filename): raise error with helpful message

### 3. **Transitive Dependencies (Automatic)**
- If `eval` depends on `train`, and `train` depends on `prep`, then `eval` can access artifacts from both `train` and `prep`
- Transitive resolution happens automatically (full depth, no limits)
- Prevents redundant dependency declarations

### 4. **Dependency Sweeps**
- `-D id1,id2,id3` creates 3 experiments (one per dependency)
- Cartesian product with parameter sweeps: `-D prep1,prep2 --param "lr=0.01,0.1"` creates 4 experiments (2 deps × 2 lr)
- Each experiment stores its single dependency

### 5. **Parameter Conflict Detection**
- After implementing parameter tracking (separate feature), warn if dependency and current experiment use conflicting parameter values
- Example: `prep` used `batch_size=32`, `train` uses `batch_size=64` → warning
- Non-blocking (warn but allow execution)

### 6. **Validation**
- Dependency experiment must exist
- Dependency experiment must have status="completed" (not running, failed, staged)
- Validation happens at experiment execution time (not at staging time)

### 7. **Storage**
- Store dependencies in `~/.yanex/experiments/{exp_id}/dependencies.json`
- Format: `{"dependency_ids": ["id1", "id2"], "metadata": {...}}`
- Store only forward links (what this experiment depends on)
- Backward links (what depends on this) computed on-demand via scan

### 8. **Out of Scope (Explicitly NOT Needed)**
- ❌ Fan-in experiments (load artifacts from multiple deps via iterator) - can be done with Results API
- ❌ Named dependency slots (e.g., `data=id1, model=id2`)
- ❌ Config file dependency declarations
- ❌ Artifact-level dependencies (just experiment-level)

## Your Task

Design and document the dependency tracking system for yanex. Specifically:

### 1. Storage Design
- What should `dependencies.json` contain?
- Should we track additional metadata (script names, timestamps)?
- How do we handle backward links (querying "what depends on this experiment")?

### 2. API Design (Run API - `yanex.*`)
- `get_dependencies()` - return what? List of Experiment objects? Dict?
- How should `load_artifact()` integrate with dependency resolution?
- Any additional helper methods needed?

### 3. Artifact Resolution Algorithm
- How to search across current + direct deps + transitive deps?
- How to detect ambiguities?
- What error messages for common cases?
- Performance considerations (caching, avoiding repeated scans)?

### 4. CLI Integration
- How to parse `-D id1,id2` syntax?
- How to resolve short IDs (4+ chars)?
- Validation: when and what to check?
- How to expand dependency sweeps?

### 5. Integration Points
- `yanex/core/manager.py` - creating experiments with dependencies
- `yanex/api.py` - `get_dependencies()`, `load_artifact()` integration
- `yanex/cli/commands/run.py` - CLI flag parsing
- New modules needed?

### 6. Edge Cases
- Circular dependencies: how to detect and handle?
- Missing dependencies: when to error vs warn?
- Archived experiments: can they be dependencies?
- Transitive resolution: max depth? cycle detection?

### 7. Testing Strategy
- Key scenarios to test
- Integration tests needed

## Constraints

- Keep it simple - this is a lightweight experiment tracker, not Airflow
- Reuse existing patterns (e.g., short ID resolution, experiment status checking)
- The new artifacts API is already implemented (`load_artifact()`, `save_artifact()`, etc.)
- Parameters tracking will be separate feature (just design how conflict detection hooks in)
- Must work with existing features: staging, parallel execution, parameter sweeps

## Deliverable

Create a detailed design document (markdown format) covering:

1. **Overview** - Summary of the feature
2. **User workflows** - CLI and API examples
3. **Storage format** - JSON schemas
4. **API specifications** - Function signatures, return types, behavior
5. **Implementation plan** - Modules to create/modify, key algorithms
6. **Edge cases** - How to handle circular deps, ambiguities, errors
7. **Testing approach** - What to test

Focus on designing a system that feels natural to yanex users and integrates cleanly with existing features. For
small use cases (single scripts) and for inexperienced users, the dependency system should become effectively invisible.
Only advanced users running multi-stage workflows should need to think about it.
