# Yanex Dependency Tracking System - Design Document

## 1. Overview

This document specifies the design for lightweight dependency tracking in Yanex, enabling multi-stage ML workflows where experiments can declare dependencies on prior experiments and automatically access their artifacts.

### Goals
- **Simple declaration**: Single CLI flag (`-D exp_id`) to declare dependencies
- **Automatic artifact resolution**: `yanex.load_artifact()` searches current experiment and all dependencies
- **Transitive dependencies**: Access artifacts from entire dependency chain automatically
- **Dependency sweeps**: Create multiple experiments, one per dependency
- **Validation**: Ensure dependencies exist and completed successfully
- **Minimal complexity**: No named slots, config file syntax, or heavyweight orchestration

### Non-Goals (Explicitly Out of Scope)
- Named dependency slots (e.g., `data=id1, model=id2`)
- Config file dependency declarations
- Artifact-level dependencies (only experiment-level)
- Fan-in artifact iteration (use Results API instead)
- Dependency execution orchestration (use external tools)

---

## 2. User Workflows

### 2.1 CLI Workflow - Basic Chain

```bash
# Step 1: Prepare data (no dependencies)
yanex run prepare_data.py --param "source=dataset_v2"
# → abc12345

# Step 2: Train model (depends on data)
yanex run train_model.py -D abc12345
# → def67890
# Can load artifacts from abc12345

# Step 3: Evaluate (depends on training)
yanex run evaluate.py -D def67890
# → ghi23456
# Can load artifacts from def67890 AND abc12345 (transitive)
```

### 2.2 CLI Workflow - Dependency Sweeps

```bash
# Prepare two datasets
yanex run prepare_data.py --param "source=dataset_a"  # → prep_a
yanex run prepare_data.py --param "source=dataset_b"  # → prep_b

# Train on both datasets (creates 2 experiments)
yanex run train_model.py -D prep_a,prep_b
# → train_1 (depends on prep_a)
# → train_2 (depends on prep_b)
```

### 2.3 CLI Workflow - Cartesian Product with Parameter Sweeps

```bash
# Dependency sweep × parameter sweep = Cartesian product
yanex run train.py -D prep_a,prep_b --param "lr=0.01,0.1"
# Creates 4 experiments:
# → exp1 (deps: prep_a, params: lr=0.01)
# → exp2 (deps: prep_a, params: lr=0.1)
# → exp3 (deps: prep_b, params: lr=0.01)
# → exp4 (deps: prep_b, params: lr=0.1)
```

### 2.4 CLI Workflow - Short ID Support

```bash
# Can use short IDs (4+ characters minimum)
yanex run train.py -D abc1
# Resolves abc1 → abc12345 (if unique)
# Error if ambiguous or not found
```

### 2.5 Python API - Automatic Artifact Resolution

```python
# train_model.py (run with: yanex run train_model.py -D abc12345)
import yanex

# Load artifacts - searches current experiment, then dependencies (transitive)
train_data = yanex.load_artifact("train_data.jsonl")
test_data = yanex.load_artifact("test_data.jsonl")
# Returns None if not found anywhere
# Raises error if found in multiple dependency experiments (ambiguous)

# Work with data
model = train_model(train_data)
yanex.save_artifact(model, "model.pt")
```

### 2.6 Python API - Access Dependency Metadata

```python
# evaluate.py (run with: yanex run evaluate.py -D train_id)
import yanex

# Get list of direct dependencies
deps = yanex.get_dependencies()
# Returns: [Experiment(train_id)]

# Get all dependencies (including transitive)
all_deps = yanex.get_dependencies(transitive=True)
# Returns: [Experiment(prep_id), Experiment(train_id)] (topological order)

# Access dependency metadata
for dep in deps:
    print(f"Training experiment: {dep.id}")
    print(f"Script: {dep.script_path}")
    print(f"Learning rate: {dep.get_param('lr')}")
    print(f"Best accuracy: {dep.get_metric('best_accuracy')}")

# Load model from dependency
model = yanex.load_artifact("model_best.pt")
```

### 2.7 Python API - Explicit Dependency Artifact Loading

```python
import yanex

# If artifact resolution is ambiguous, load from specific dependency
deps = yanex.get_dependencies()
train_exp = deps[0]
model = train_exp.load_artifact("model.pt")

# Or get all dependencies (including transitive) in topological order
all_deps = yanex.get_dependencies(transitive=True)
# Returns: [Experiment(prep_id), Experiment(train_id)]
```

---

## 3. Storage Format

### 3.1 Directory Structure

```
~/.yanex/experiments/
├── abc12345/                   # Data preparation experiment
│   ├── metadata.json
│   ├── params.yaml
│   ├── metrics.json
│   └── artifacts/
│       ├── train_data.jsonl
│       └── test_data.jsonl
│
├── def67890/                   # Training experiment (depends on abc12345)
│   ├── metadata.json
│   ├── params.yaml
│   ├── metrics.json
│   ├── dependencies.json       # NEW: Dependency tracking
│   └── artifacts/
│       └── model.pt
│
└── ghi23456/                   # Evaluation (depends on def67890)
    ├── metadata.json
    ├── params.yaml
    ├── dependencies.json       # NEW: Dependency tracking
    └── artifacts/
        └── results.json
```

### 3.2 dependencies.json Schema

```json
{
  "dependency_ids": ["abc12345"],
  "created_at": "2025-11-23T10:30:00.000000",
  "metadata": {
    "abc12345": {
      "short_id_used": "abc1",
      "resolved_at": "2025-11-23T10:30:00.000000",
      "status_at_resolution": "completed",
      "script_path": "/path/to/prepare_data.py",
      "name": "data-prep-v2"
    }
  }
}
```

**Field Descriptions:**

- `dependency_ids` (required): List of full 8-character experiment IDs this experiment depends on (direct dependencies only)
- `created_at` (required): ISO 8601 timestamp when dependencies were resolved
- `metadata` (optional): Additional context about each dependency
  - `short_id_used`: The short ID provided by user (if applicable)
  - `resolved_at`: When the short ID was resolved to full ID
  - `status_at_resolution`: Dependency status when validated ("completed" required)
  - `script_path`: Dependency's script path (for debugging)
  - `name`: Dependency's name (for debugging)

**Design Rationale:**
- Store only direct dependencies (not transitive) to avoid duplication and staleness
- Compute transitive dependencies on-demand via recursive resolution
- Include metadata for debugging and reproducibility context
- Keep schema simple and extensible

### 3.3 metadata.json Updates

No changes needed to `metadata.json` - dependencies are stored in separate file for modularity.

Optional: Track CLI-provided dependency syntax in `metadata.json` for full reproducibility:

```json
{
  "id": "def67890",
  "status": "completed",
  "cli_args": {
    "script": "train_model.py",
    "dependencies": ["abc1"]  // Short IDs as provided by user
  },
  ...
}
```

---

## 4. API Specifications

### 4.1 Run API (`yanex/api.py`)

#### `get_dependencies()` - Get Dependencies

```python
def get_dependencies(
    transitive: bool = False,
    include_self: bool = False,
) -> list[Experiment]:
    """Get dependency experiments for current experiment.

    Args:
        transitive: If True, return all transitive dependencies (full graph).
                   If False, return only direct dependencies.
        include_self: If True, include current experiment in result.
                     Only applicable when transitive=True.

    Returns:
        List of Experiment objects in topological order (dependencies before dependents).
        Empty list if no dependencies or not in experiment context.

    Raises:
        CircularDependencyError: If circular dependency detected (only when transitive=True).

    Example:
        # Get direct dependencies
        deps = yanex.get_dependencies()
        for dep in deps:
            print(f"Depends on: {dep.id} ({dep.name})")
            print(f"  Script: {dep.script_path}")

        # Get all dependencies (transitive)
        # Current experiment ghi23456 depends on def67890
        # def67890 depends on abc12345
        all_deps = yanex.get_dependencies(transitive=True)
        # Returns: [Experiment(abc12345), Experiment(def67890)]

        all_deps = yanex.get_dependencies(transitive=True, include_self=True)
        # Returns: [Experiment(abc12345), Experiment(def67890), Experiment(ghi23456)]
    """
```

#### `load_artifact()` - Enhanced with Dependency Resolution

```python
def load_artifact(filename: str, loader: Any | None = None) -> Any | None:
    """Load artifact with automatic dependency resolution.

    Works in two modes:
    - Standalone mode (no experiment context): Load from ./artifacts/
    - Experiment mode (with context): Search current experiment + dependencies

    Search order (experiment mode):
    1. Current experiment's artifacts directory
    2. Direct dependencies (in order declared)
    3. Transitive dependencies (topological order)

    Args:
        filename: Name of artifact file to load.
        loader: Optional custom loader function.

    Returns:
        Loaded artifact object, or None if not found anywhere.

    Raises:
        AmbiguousArtifactError: If artifact found in multiple experiments.

    Example:
        # Standalone mode
        data = yanex.load_artifact("data.json")  # Loads from ./artifacts/data.json

        # Experiment mode (searches current exp + dependencies)
        data = yanex.load_artifact("train_data.jsonl")

        # If found in multiple places, raises AmbiguousArtifactError with:
        # "Artifact 'data.json' found in multiple experiments:
        #  - abc12345 (data-prep-v1)
        #  - def67890 (data-prep-v2)
        # Load explicitly: yanex.get_dependencies()[0].load_artifact('data.json')"
    """
```

### 4.2 Results API (`yanex/results/experiment.py`)

#### `Experiment.get_dependencies()` - Get Dependencies of Any Experiment

```python
class Experiment:
    def get_dependencies(
        self,
        transitive: bool = False,
        include_self: bool = False,
    ) -> list[Experiment]:
        """Get dependencies for this experiment.

        Same API as yanex.get_dependencies() but works on any experiment.

        Args:
            transitive: If True, return all transitive dependencies.
                       If False, return only direct dependencies.
            include_self: If True, include this experiment in result.
                         Only applicable when transitive=True.

        Returns:
            List of Experiment objects in topological order.
            Empty list if no dependencies.

        Example:
            exp = yanex.results.get_experiment("ghi23456")
            deps = exp.get_dependencies()  # Direct only
            all_deps = exp.get_dependencies(transitive=True)  # All transitive
            with_self = exp.get_dependencies(transitive=True, include_self=True)
        """
```

#### `Experiment.get_dependents()` - Get Experiments That Depend On This One

```python
class Experiment:
    def get_dependents(self, transitive: bool = False) -> list[Experiment]:
        """Get experiments that depend on this one (backward links).

        Computed by scanning all experiments for dependency references.
        May be slow for large experiment databases.

        Args:
            transitive: If True, return all transitive dependents.

        Returns:
            List of Experiment objects.
            Empty list if nothing depends on this experiment.

        Example:
            data_exp = yanex.results.get_experiment("abc12345")
            dependents = data_exp.get_dependents()
            # Returns: [Experiment(def67890), Experiment(xyz99999), ...]
        """
```

### 4.3 Core Module (`yanex/core/dependencies.py`) - NEW

```python
"""Dependency resolution and validation for experiment workflows."""

from pathlib import Path
from typing import Any
from graphlib import TopologicalSorter
from ..utils.exceptions import (
    ExperimentNotFoundError,
    AmbiguousIDError,
    CircularDependencyError,
    InvalidDependencyError,
)
from ..utils.id_resolution import resolve_experiment_id  # Use existing utility


class DependencyResolver:
    """Handles dependency resolution, validation, and graph operations."""

    def __init__(self, manager: ExperimentManager):
        self.manager = manager

    def resolve_short_id(self, short_id: str) -> str:
        """Resolve short experiment ID to full ID using existing utilities.

        Args:
            short_id: Partial experiment ID (4+ characters).

        Returns:
            Full 8-character experiment ID.

        Raises:
            ExperimentNotFoundError: No matching experiment.
            AmbiguousIDError: Multiple matches found.

        Note:
            Uses yanex.utils.id_resolution.resolve_experiment_id()
        """

    def validate_dependency(self, experiment_id: str, for_staging: bool = False) -> None:
        """Validate that experiment can be used as dependency.

        Args:
            experiment_id: Full experiment ID to validate.
            for_staging: If True, allow dependencies with status="staged".

        Raises:
            ExperimentNotFoundError: Experiment doesn't exist.
            InvalidDependencyError: Experiment status is not "completed".
        """

    def resolve_and_validate_dependencies(
        self,
        dependency_ids: list[str],
        for_staging: bool = False,
    ) -> list[str]:
        """Resolve short IDs and validate all dependencies.

        Args:
            dependency_ids: List of experiment IDs (may be short).
            for_staging: If True, allow dependencies with status="staged".

        Returns:
            List of full experiment IDs.

        Raises:
            ExperimentNotFoundError: Dependency doesn't exist.
            AmbiguousIDError: Short ID matches multiple experiments.
            InvalidDependencyError: Dependency has invalid status.
        """

    def get_transitive_dependencies(
        self,
        experiment_id: str,
        include_self: bool = False,
    ) -> list[str]:
        """Get all dependencies (direct + transitive) in topological order.

        Uses graphlib.TopologicalSorter from Python standard library for
        graph traversal and cycle detection.

        Args:
            experiment_id: Experiment ID to get dependencies for.
            include_self: If True, include experiment_id in result.

        Returns:
            List of experiment IDs in topological order (dependencies before dependents).

        Raises:
            CircularDependencyError: Circular dependency detected.

        Note:
            Uses graphlib.TopologicalSorter for efficient topological sorting
            and automatic cycle detection.
        """

    def detect_circular_dependency(
        self,
        experiment_id: str,
        new_dependency_id: str,
    ) -> bool:
        """Check if adding dependency would create a cycle.

        Args:
            experiment_id: Experiment that would get new dependency.
            new_dependency_id: Dependency to be added.

        Returns:
            True if circular dependency would be created.
        """

    def find_artifact_in_dependencies(
        self,
        experiment_id: str,
        artifact_filename: str,
    ) -> tuple[str | None, list[str]]:
        """Search for artifact in experiment and all dependencies.

        Search order:
        1. Current experiment
        2. Direct dependencies (in declaration order)
        3. Transitive dependencies (depth-first order)

        Args:
            experiment_id: Current experiment ID.
            artifact_filename: Name of artifact to find.

        Returns:
            Tuple of (experiment_id_with_artifact, all_experiment_ids_with_artifact)
            - If found uniquely: ("abc12345", ["abc12345"])
            - If found in multiple: (None, ["abc12345", "def67890"])
            - If not found: (None, [])
        """


class DependencyStorage:
    """Handles persistence of dependency data."""

    def __init__(self, directory_manager: FileSystemDirectoryManager):
        self.directory_manager = directory_manager

    def save_dependencies(
        self,
        experiment_id: str,
        dependency_ids: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save dependencies.json file."""

    def load_dependencies(self, experiment_id: str) -> dict[str, Any]:
        """Load dependencies.json file.

        Returns:
            Dict with keys: dependency_ids, created_at, metadata.
            Returns {"dependency_ids": []} if file doesn't exist.
        """

    def dependency_file_exists(self, experiment_id: str) -> bool:
        """Check if dependencies.json exists."""
```

### 4.4 New Exception Classes

```python
# yanex/utils/exceptions.py

class AmbiguousIDError(YanexError):
    """Raised when short ID matches multiple experiments."""

class CircularDependencyError(YanexError):
    """Raised when circular dependency detected."""

class InvalidDependencyError(YanexError):
    """Raised when dependency validation fails."""

class AmbiguousArtifactError(YanexError):
    """Raised when artifact found in multiple dependency experiments."""
```

---

## 5. Implementation Plan

### 5.1 New Modules

#### `yanex/core/dependencies.py`
- `DependencyResolver` class (resolution, validation, graph operations)
- `DependencyStorage` class (persistence)
- Helper functions for common operations

### 5.2 Modified Modules

#### `yanex/core/manager.py`
```python
def create_experiment(
    self,
    script_path: Path,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    stage_only: bool = False,
    script_args: list[str] | None = None,
    cli_args: dict[str, Any] | None = None,
    dependency_ids: list[str] | None = None,  # NEW
) -> str:
    """Create experiment with optional dependencies.

    New behavior:
    1. Resolve and validate dependency IDs
    2. Detect circular dependencies
    3. Save dependencies.json
    4. Store dependency metadata in metadata.json
    """
```

#### `yanex/cli/commands/run.py`
```python
@click.option(
    "-D",
    "--depends-on",
    "dependencies",
    multiple=True,
    help="Experiment ID this run depends on (repeatable). Supports short IDs.",
)
def run(
    script: str,
    config: tuple[str],
    param: tuple[str],
    dependencies: tuple[str],  # NEW
    stage: bool,
    parallel: int | None,
    ...
):
    """Enhanced to handle dependency sweeps and validation."""
```

**Dependency Sweep Logic:**
```python
# Parse dependencies: "id1,id2,id3" or multiple -D flags
dependency_list = _parse_dependencies(dependencies)

# Expand sweeps: Cartesian product of dependencies × parameters
if dependency_list:
    # For each dependency, create separate experiment
    for dep_id in dependency_list:
        # Resolve short ID
        full_id = resolver.resolve_short_id(dep_id)

        # Validate dependency
        resolver.validate_dependency(full_id, for_staging=stage)

        # Create experiment with single dependency
        create_experiment(..., dependency_ids=[full_id])
```

#### `yanex/api.py`
```python
def get_dependencies() -> list[Experiment]:
    """Implementation using DependencyResolver."""

def get_all_dependencies(include_self: bool = False) -> list[Experiment]:
    """Implementation using DependencyResolver."""

def load_artifact(filename: str, loader: Any | None = None) -> Any | None:
    """Enhanced with dependency resolution logic."""
```

#### `yanex/results/experiment.py`
```python
class Experiment:
    def get_dependencies(self, transitive: bool = False) -> list[Experiment]:
        """Use DependencyResolver for lookup."""

    def get_dependents(self, transitive: bool = False) -> list[Experiment]:
        """Scan all experiments for backward links."""
```

#### `yanex/core/storage_composition.py`
```python
class CompositeExperimentStorage:
    def __init__(self, experiments_dir: Path):
        ...
        self.dependency_storage = DependencyStorage(self.directory_manager)  # NEW
```

#### `yanex/utils/exceptions.py`
Add new exception classes (see section 4.4).

### 5.3 Implementation Phases

**Phase 1: Core Infrastructure**
1. Create `yanex/core/dependencies.py` with `DependencyStorage` class
2. Add exception classes to `yanex/utils/exceptions.py`
3. Add `DependencyStorage` to `CompositeExperimentStorage`
4. Verify existing `yanex.utils.id_resolution` utilities work for dependency resolution
5. Write unit tests for storage layer

**Phase 2: Resolution and Validation**
1. Implement `DependencyResolver` class using:
   - `graphlib.TopologicalSorter` for transitive resolution and cycle detection
   - Existing `yanex.utils.id_resolution` for short ID resolution
2. Add dependency validation (status checks)
3. Write unit tests for resolver

**Phase 3: Experiment Manager Integration**
1. Extend `create_experiment()` to accept `dependency_ids`
2. Save dependencies during experiment creation
3. Update metadata to include dependency info
4. Write integration tests

**Phase 4: CLI Integration**
1. Add `-D`/`--depends-on` flag to `run` command
2. Implement dependency sweep expansion
3. Add validation error messages
4. Write CLI integration tests

**Phase 5: API Integration**
1. Implement `get_dependencies(transitive=False, include_self=False)` in `yanex/api.py`
2. Enhance `load_artifact()` with dependency resolution (preserves standalone mode)
3. Add `get_dependencies()` and `get_dependents()` methods to `Experiment` class
4. Ensure consistent API between Run API and Results API
5. Write API integration tests

**Phase 6: Testing and Documentation**
1. End-to-end workflow tests
2. Edge case tests (circular deps, ambiguous artifacts, etc.)
3. Update documentation and examples
4. Add example workflows to examples directory

---

## 6. Edge Cases and Error Handling

### 6.1 Circular Dependencies

**Detection:**
```python
# When creating experiment B with dependency on A:
# 1. Check if A depends on B (direct)
# 2. Check if any of A's dependencies depend on B (transitive)

def detect_circular_dependency(experiment_id: str, new_dependency_id: str) -> bool:
    # Get all dependencies of new_dependency_id
    dep_chain = get_transitive_dependencies(new_dependency_id)
    # Check if experiment_id is in the chain
    return experiment_id in dep_chain
```

**Error Message:**
```
CircularDependencyError: Cannot add dependency on 'abc12345'.
This would create a circular dependency:
  ghi23456 (current) → def67890 → abc12345 → ghi23456

Dependency chain:
  1. ghi23456 (evaluate.py)
  2. def67890 (train.py)
  3. abc12345 (prepare.py)
```

### 6.2 Missing Dependencies

**Scenario 1: Dependency doesn't exist**
```bash
yanex run train.py -D nonexistent
```

**Error:**
```
ExperimentNotFoundError: No experiment found matching 'nonexistent'.

Suggestions:
- Check experiment ID is correct
- Use 'yanex list' to see available experiments
- Ensure experiment wasn't deleted or archived
```

**Scenario 2: Ambiguous short ID**
```bash
yanex run train.py -D abc
```

**Error:**
```
AmbiguousIDError: Short ID 'abc' matches multiple experiments:
  - abc12345 (data-prep-v1, completed)
  - abc98765 (data-prep-v2, completed)

Use a longer ID to disambiguate (minimum 4 characters):
  yanex run train.py -D abc12  # For abc12345
  yanex run train.py -D abc98  # For abc98765
```

### 6.3 Invalid Dependency Status

**Scenario: Dependency not completed**
```bash
# Create experiment but don't run it
yanex run prepare.py --stage  # → abc12345 (status=staged)

# Try to depend on it
yanex run train.py -D abc12345
```

**Error:**
```
InvalidDependencyError: Dependency 'abc12345' has invalid status 'staged'.

Dependencies must have status 'completed'.

Current status: staged
Experiment: abc12345 (prepare.py)

To fix:
  yanex run --staged  # Execute staged experiment first
```

**Valid statuses for dependencies:** `completed` only
**Invalid statuses:** `created`, `running`, `failed`, `cancelled`, `staged`

**Special case for staging:**
```bash
# Allow staged dependencies when creating staged experiments
yanex run prepare.py --stage  # → abc (status=staged)
yanex run train.py -D abc --stage  # OK: both staged
```

### 6.4 Ambiguous Artifacts

**Policy:** Error on ANY artifact ambiguity, including transitive dependencies.

**Scenario 1: Same filename in multiple direct dependencies**
```python
# Current experiment ghi23456 depends on:
#   - def67890 (has data.json)
#   - abc12345 (has data.json)

data = yanex.load_artifact("data.json")
```

**Error:**
```
AmbiguousArtifactError: Artifact 'data.json' found in multiple experiments:
  1. def67890 (train-model-v1)
  2. abc12345 (data-prep-v2)

Load explicitly from specific dependency:
  deps = yanex.get_dependencies()
  data = deps[0].load_artifact("data.json")  # From def67890

Or get all dependencies:
  all_deps = yanex.get_dependencies(transitive=True)
  for dep in all_deps:
      if dep.id == "abc12345":
          data = dep.load_artifact("data.json")
```

**Scenario 2: Same filename in transitive dependencies**
```python
# Dependency chain: ghi23456 → def67890 → abc12345
# Both def67890 and abc12345 have config.yaml

config = yanex.load_artifact("config.yaml")
```

**Error:**
```
AmbiguousArtifactError: Artifact 'config.yaml' found in multiple experiments:
  1. def67890 (train-model-v1) [direct dependency]
  2. abc12345 (data-prep-v2) [transitive dependency]

This includes transitive dependencies. Load explicitly:
  all_deps = yanex.get_dependencies(transitive=True)
  config = all_deps[0].load_artifact("config.yaml")  # From abc12345
```

**Rationale:** Strict checking prevents subtle bugs from loading the wrong artifact.

### 6.5 Archived Dependencies

**Policy:** Archived experiments CAN be used as dependencies.

**Rationale:**
- Archiving is organizational, not a status change
- Experiment data remains accessible
- Reproducibility requires access to old experiments

**Implementation:**
```python
def validate_dependency(experiment_id: str) -> None:
    metadata = storage.load_metadata(experiment_id, include_archived=True)
    if metadata["status"] != "completed":
        raise InvalidDependencyError(...)
```

### 6.6 Transitive Resolution Depth

**Policy:** No depth limit, full transitive resolution.

**Rationale:**
- ML workflows rarely exceed 5-10 stages
- Graph traversal is fast (< 1ms for 100 experiments)
- Users shouldn't worry about depth limits

**Cycle Detection:** Required to prevent infinite loops.

### 6.7 Deleted Dependencies

**Scenario: Dependency deleted after experiment created**
```bash
yanex run prepare.py  # → abc12345
yanex run train.py -D abc12345  # → def67890
yanex delete abc12345  # Delete dependency

# Later, try to load artifact from train experiment
yanex results get def67890
```

**Behavior:**
- `get_dependencies()` raises `ExperimentNotFoundError` with helpful message
- `load_artifact()` skips missing dependencies, continues searching
- Error only if artifact not found anywhere

**Error Message:**
```
Warning: Dependency 'abc12345' not found (may have been deleted).
Experiment 'def67890' depends on:
  - abc12345 (MISSING)

Artifacts from missing dependencies cannot be loaded.
```

### 6.8 Dependency Sweeps with Failures

**Scenario: Some dependencies invalid**
```bash
yanex run train.py -D abc1,def2,ghi3
# abc1 → completed ✓
# def2 → failed ✗
# ghi3 → running ✗
```

**Behavior:**
- Validate ALL dependencies before creating ANY experiments
- If any validation fails, create NONE
- Show all errors together

**Error Message:**
```
InvalidDependencyError: Cannot create experiments. 2 dependencies invalid:

1. def67890 (def2) - Status 'failed' (must be 'completed')
2. ghi23456 (ghi3) - Status 'running' (must be 'completed')

Valid dependency:
  ✓ abc12345 (abc1) - Status 'completed'

Fix dependency issues and try again.
```

### 6.9 Mixed Dependency and Parameter Sweeps

**Scenario: Cartesian product expansion**
```bash
yanex run train.py -D prep1,prep2 --param "lr=0.01,0.1"
```

**Expansion:**
```
4 experiments created:
  1. exp_001: deps=[prep1], params={lr: 0.01}
  2. exp_002: deps=[prep1], params={lr: 0.1}
  3. exp_003: deps=[prep2], params={lr: 0.01}
  4. exp_004: deps=[prep2], params={lr: 0.1}
```

**Algorithm:**
```python
# 1. Resolve and validate dependencies
resolved_deps = [resolve_short_id(d) for d in dependency_list]

# 2. Expand parameter sweeps
param_configs = expand_parameter_sweeps(config)

# 3. Cartesian product
experiments = []
for dep_id in resolved_deps:
    for param_config in param_configs:
        experiments.append(
            ExperimentSpec(
                script_path=script,
                config=param_config,
                dependency_ids=[dep_id],  # Single dependency
                ...
            )
        )
```

### 6.10 Parameter Conflict Detection (Future Feature)

**Not implemented in initial version, but designed for future extension.**

**Scenario:**
```bash
yanex run prepare.py --param "batch_size=32"  # → abc
yanex run train.py -D abc --param "batch_size=64"  # → def
```

**Desired behavior (future):**
```
Warning: Parameter conflict detected with dependency abc12345:
  - batch_size: 64 (current) vs 32 (dependency abc12345)

This may cause unexpected behavior if the script assumes consistent batch_size.
Continuing execution...
```

**Hook for future implementation:**
```python
def _check_parameter_conflicts(
    current_params: dict,
    dependency_params: dict,
) -> list[str]:
    """Compare parameters and return list of conflicts."""
    conflicts = []
    for key, value in current_params.items():
        if key in dependency_params:
            if value != dependency_params[key]:
                conflicts.append(
                    f"  - {key}: {value} (current) vs {dependency_params[key]} (dependency)"
                )
    return conflicts
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**`tests/core/test_dependencies.py`**
- DependencyStorage: save, load, exists
- DependencyResolver: short ID resolution
- DependencyResolver: validation (status checks)
- DependencyResolver: circular dependency detection
- DependencyResolver: transitive resolution
- DependencyResolver: artifact search in dependencies

**`tests/utils/test_exceptions.py`**
- New exception classes

### 7.2 Integration Tests

**`tests/core/test_manager.py`**
- Create experiment with dependencies
- Validate dependency IDs during creation
- Save dependencies.json correctly
- Circular dependency prevention

**`tests/cli/commands/test_run.py`**
- CLI flag parsing: `-D id1 -D id2`
- CLI flag parsing: `-D id1,id2,id3`
- Short ID resolution from CLI
- Dependency validation errors
- Dependency sweep expansion
- Cartesian product: dependencies × parameters

**`tests/test_api.py`**
- `get_dependencies()` returns correct list
- `get_all_dependencies()` includes transitive
- `load_artifact()` searches dependencies
- `load_artifact()` detects ambiguous artifacts
- `load_artifact()` handles missing artifacts

**`tests/results/test_experiment.py`**
- `Experiment.get_dependencies()`
- `Experiment.get_dependents()`
- Transitive dependency traversal

### 7.3 End-to-End Workflow Tests

**`tests/integration/test_dependency_workflows.py`**

**Test: Linear dependency chain**
```python
def test_linear_dependency_chain(temp_dir):
    # Create 3 experiments: prep → train → eval
    prep_id = create_experiment("prepare.py", {})
    train_id = create_experiment("train.py", {}, deps=[prep_id])
    eval_id = create_experiment("evaluate.py", {}, deps=[train_id])

    # Verify transitive access
    exp = get_experiment(eval_id)
    all_deps = exp.get_dependencies(transitive=True)
    assert len(all_deps) == 2
    assert prep_id in [d.id for d in all_deps]
```

**Test: Dependency sweep**
```python
def test_dependency_sweep(temp_dir):
    # Create 2 data prep experiments
    prep1 = create_experiment("prepare.py", {"source": "a"})
    prep2 = create_experiment("prepare.py", {"source": "b"})

    # Run with dependency sweep
    results = cli_run("train.py", dependencies=[prep1, prep2])

    # Should create 2 training experiments
    assert len(results) == 2
```

**Test: Artifact resolution**
```python
def test_artifact_resolution_from_dependencies(temp_dir):
    # Create prep experiment with artifact
    prep_id = create_experiment("prepare.py", {})
    save_artifact_to_experiment(prep_id, {"data": [1, 2, 3]}, "data.json")

    # Create train experiment depending on prep
    train_id = create_experiment("train.py", {}, deps=[prep_id])

    # Load artifact from within train context
    with experiment_context(train_id):
        data = yanex.load_artifact("data.json")
        assert data == {"data": [1, 2, 3]}
```

**Test: Circular dependency detection**
```python
def test_circular_dependency_detection(temp_dir):
    exp_a = create_experiment("a.py", {})
    exp_b = create_experiment("b.py", {}, deps=[exp_a])

    # Try to create circular dependency
    with pytest.raises(CircularDependencyError):
        create_experiment("a_updated.py", {}, deps=[exp_b], experiment_id=exp_a)
```

**Test: Ambiguous artifact error**
```python
def test_ambiguous_artifact_error(temp_dir):
    # Create 2 dependencies with same artifact filename
    dep1 = create_experiment("prep1.py", {})
    dep2 = create_experiment("prep2.py", {})
    save_artifact_to_experiment(dep1, {"v": 1}, "data.json")
    save_artifact_to_experiment(dep2, {"v": 2}, "data.json")

    # Create experiment depending on both
    train_id = create_experiment("train.py", {}, deps=[dep1, dep2])

    # Loading artifact should raise error
    with experiment_context(train_id):
        with pytest.raises(AmbiguousArtifactError):
            yanex.load_artifact("data.json")
```

### 7.4 Performance Tests

**`tests/performance/test_dependency_performance.py`**

**Test: Large dependency graph traversal**
```python
def test_large_dependency_graph_traversal():
    # Create chain of 100 experiments
    exp_ids = []
    for i in range(100):
        deps = [exp_ids[-1]] if exp_ids else []
        exp_id = create_experiment(f"exp_{i}.py", {}, deps=deps)
        exp_ids.append(exp_id)

    # Measure transitive resolution time
    start = time.time()
    all_deps = get_all_dependencies(exp_ids[-1])
    duration = time.time() - start

    assert len(all_deps) == 99
    assert duration < 0.1  # Should be < 100ms
```

### 7.5 Edge Case Tests

**`tests/edge_cases/test_dependency_edge_cases.py`**
- Deleted dependencies
- Archived dependencies
- Missing dependency files
- Malformed dependencies.json
- Empty dependency list
- Self-dependency prevention

---

## 8. Algorithm Details

### 8.1 Short ID Resolution

**Implementation:** Use existing `yanex.utils.id_resolution.resolve_experiment_id()` utility.

The dependency tracking system reuses the existing ID resolution infrastructure rather than reimplementing it.

```python
from ..utils.id_resolution import resolve_experiment_id

def resolve_short_id(self, short_id: str) -> str:
    """Resolve short ID to full experiment ID.

    Delegates to existing yanex.utils.id_resolution.resolve_experiment_id().
    """
    return resolve_experiment_id(short_id, self.manager)
```

**Key features:**
- Supports partial IDs (minimum 4 characters, configurable)
- Searches both active and archived experiments
- Provides helpful error messages for not found / ambiguous cases
- Consistent behavior across all yanex commands

### 8.2 Transitive Dependency Resolution

**Implementation:** Use `graphlib.TopologicalSorter` from Python standard library.

```python
from graphlib import TopologicalSorter, CycleError

def get_transitive_dependencies(
    experiment_id: str,
    include_self: bool = False,
) -> list[str]:
    """Get all dependencies in topological order using graphlib.

    TopologicalSorter provides:
    - Automatic cycle detection
    - Efficient graph traversal
    - Standard library implementation (no custom graph algorithms)
    """
    # Build dependency graph
    graph = {}
    to_visit = [experiment_id]
    visited = set()

    while to_visit:
        exp_id = to_visit.pop()
        if exp_id in visited:
            continue
        visited.add(exp_id)

        # Load dependencies for this experiment
        dep_data = storage.load_dependencies(exp_id)
        dep_ids = dep_data.get("dependency_ids", [])
        graph[exp_id] = dep_ids

        # Add dependencies to visit queue
        to_visit.extend(dep_ids)

    # Use TopologicalSorter for ordering and cycle detection
    try:
        sorter = TopologicalSorter(graph)
        sorted_ids = list(sorter.static_order())
    except CycleError as e:
        raise CircularDependencyError(
            f"Circular dependency detected in experiment '{experiment_id}': {e}"
        ) from e

    # Filter to only include dependencies (not the experiment itself unless requested)
    if include_self:
        return sorted_ids
    else:
        return [exp_id for exp_id in sorted_ids if exp_id != experiment_id]
```

**Benefits:**
- Uses standard library (no custom graph traversal code)
- Automatic cycle detection via `CycleError`
- Correct topological ordering guaranteed
- Well-tested implementation

### 8.3 Artifact Search in Dependencies

```python
def find_artifact_in_dependencies(
    experiment_id: str,
    artifact_filename: str,
) -> tuple[str | None, list[str]]:
    """Search for artifact in experiment and all dependencies.

    Returns:
        (unique_experiment_id, all_matching_experiment_ids)
    """
    # Get search order: current + transitive deps
    search_order = [experiment_id] + get_transitive_dependencies(experiment_id)

    # Search for artifact
    found_in = []
    for exp_id in search_order:
        if storage.artifact_exists(exp_id, artifact_filename, include_archived=True):
            found_in.append(exp_id)

    if len(found_in) == 0:
        return (None, [])
    elif len(found_in) == 1:
        return (found_in[0], found_in)
    else:
        return (None, found_in)  # Ambiguous
```

### 8.4 Circular Dependency Detection

**Implementation:** Leverage `graphlib.TopologicalSorter` for automatic cycle detection.

```python
def detect_circular_dependency(
    experiment_id: str,
    new_dependency_id: str,
) -> bool:
    """Check if adding dependency would create cycle.

    Uses graphlib.TopologicalSorter which automatically detects cycles
    via CycleError exception.
    """
    try:
        # Get all dependencies of the new dependency
        dep_chain = get_transitive_dependencies(new_dependency_id, include_self=True)

        # If current experiment is in that chain, it's circular
        return experiment_id in dep_chain
    except CircularDependencyError:
        # If the new dependency itself has a cycle, that's also a problem
        # (caught by TopologicalSorter.static_order())
        return True
```

**Benefits:**
- No manual cycle tracking needed
- Automatic detection via standard library
- Consistent with transitive resolution algorithm

---

## 9. Migration and Backward Compatibility

### 9.1 Backward Compatibility

**Existing experiments (no dependencies):**
- No `dependencies.json` file → treated as having zero dependencies
- All APIs return empty lists/None gracefully
- `load_artifact()` works exactly as before (no dependency search)

**Existing code:**
- All existing APIs continue to work unchanged
- New APIs are purely additive
- No breaking changes to storage format

### 9.2 Graceful Degradation

```python
# If dependencies.json missing or malformed
def load_dependencies(experiment_id: str) -> dict:
    try:
        with open(dep_file) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        # No dependencies - return empty
        return {"dependency_ids": [], "created_at": None, "metadata": {}}
    except json.JSONDecodeError:
        # Malformed file - log warning and return empty
        logger.warning(f"Malformed dependencies.json for {experiment_id}")
        return {"dependency_ids": [], "created_at": None, "metadata": {}}
```

---

## 10. Future Enhancements (Out of Scope)

These are explicitly NOT part of the initial implementation but documented for future consideration:

### 10.1 Parameter Conflict Detection
- Warn when current experiment params differ from dependency params
- Non-blocking warnings
- Configurable rules for which params to check

### 10.2 Dependency Visualization
- `yanex graph` command to visualize dependency DAG
- ASCII art or DOT format output
- Integration with web UI

### 10.3 Dependency Metadata Queries
- Filter experiments by dependency relationships
- Find all experiments in a dependency subtree
- Compare dependency graphs

### 10.4 Automatic Dependency Inference
- Analyze script imports and artifact loads
- Suggest dependencies based on usage patterns
- Warn about missing dependencies

---

## 11. Design Decisions (Resolved)

All design questions have been resolved:

1. **Short ID resolution**: ✅ Use existing `yanex.utils.id_resolution` utilities - no reimplementation needed

2. **Dependency metadata storage**: ✅ Keep minimal - no additional information beyond current design (script_path, name, status_at_resolution)

3. **Archived dependency policy**: ✅ Archived experiments CAN be targeted with `-D` flag
   - Archiving is organizational, not a status change
   - Reproducibility requires access to old experiments

4. **Error message verbosity**: ✅ Detailed error messages with suggestions
   - Helpful guidance for users
   - Follows existing Yanex error handling patterns

5. **Dependency sweep naming**: ✅ Use existing sweep naming convention
   - No special naming for dependency sweeps
   - Consistent with parameter sweep behavior

6. **Artifact ambiguity handling**: ✅ Error on ANY artifact ambiguity (including transitive dependencies)
   - Strict checking prevents subtle bugs
   - User must explicitly choose which experiment's artifact to use

---

## 12. Summary

This design provides a **lightweight, intuitive dependency tracking system** that:

✅ Enables multi-stage ML workflows with minimal overhead
✅ Integrates seamlessly with existing Yanex features (sweeps, staging, parallel execution)
✅ Provides automatic artifact resolution across dependency chains
✅ Validates dependencies to prevent common errors
✅ Handles edge cases gracefully with helpful error messages
✅ Maintains backward compatibility with existing experiments
✅ Follows established Yanex patterns (composition, modular storage, rich errors)
✅ Uses Python standard library (`graphlib.TopologicalSorter`) instead of custom graph algorithms
✅ Reuses existing utilities (`id_resolution`) for consistency
✅ Preserves standalone mode - scripts work with `python script.py` without yanex tracking

The system is designed to be **invisible for simple use cases** (single scripts) while providing **powerful workflow orchestration** for advanced users running complex multi-stage experiments.
