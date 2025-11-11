# Experiment Dependencies

**Status:** Planning - Architecture Finalized
**Date:** 2025-11-11
**Context:** Add dependency tracking between experiments to support ML workflow orchestration and DAG visualization

## Background

### Current Limitation

Currently, yanex experiments are completely isolated - each experiment has no knowledge of other experiments it depends on. This makes it difficult to:

1. **Track workflow pipelines** - Cannot represent multi-stage ML workflows (dataprep → train → evaluate)
2. **Reuse artifacts** - No formal way to reference artifacts from previous experiments
3. **Validate prerequisites** - Cannot ensure required experiments completed before running dependent ones
4. **Visualize workflows** - Cannot build DAG visualizations of experiment relationships

### Use Case

Typical ML workflow with dependencies:

```bash
# 1. Data preparation
yanex run dataprep.py
# → dp1 (creates train_data.parquet, test_data.parquet)

# 2. Training (depends on dataprep)
yanex run train.py --depends-on dataprep=dp1
# → tr1, tr2, tr3, tr4 (with parameter sweep)

# 3. Evaluation (depends on both dataprep and training)
yanex run evaluate.py --depends-on dataprep=dp1 --depends-on training=tr1
# → eval1
```

More advanced: Evaluate all 4 trained models:
```bash
yanex run evaluate.py \
  --depends-on dataprep=dp1 \
  --depends-on training=$(yanex id --script train.py --limit 4)
# → eval1, eval2, eval3, eval4 (one per training experiment)
```

### Goal

Enable yanex to:
1. **Declare** experiment dependencies in config files
2. **Validate** dependencies before execution (experiment exists, correct script, completed status)
3. **Track** dependency relationships bidirectionally for fast queries
4. **Visualize** experiment workflows as DAG in web UI
5. **Query** experiments by their dependency relationships

## Design Decisions

### 1. Dependency Model: Named Slots

Dependencies are represented as **named slots** that must be fulfilled with experiment IDs:

```yaml
# evaluate.py config
yanex:
  dependencies:
    dataprep: dataprep.py
    training: train.py
```

**Rationale:**
- **Named slots** are more explicit than positional dependencies
- **Script validation** ensures correct experiment types fill slots
- **Multiple slots** can reference the same script type (e.g., model1, model2 both from train.py)
- **Extensible** - can add optional dependencies, artifact requirements later

**Alternative considered:** Artifact-level dependencies (e.g., `--depends-on dp1:train_data.parquet`)
- **Rejected** because experiments are the natural boundary in ML workflows, not individual files
- Artifact-level validation can be added later as enhancement

### 2. Declaration Location: Config File `yanex:` Section

Dependencies are declared in the experiment's config file under the existing `yanex:` section:

```yaml
# config.yaml
learning_rate: 0.001
batch_size: 32

yanex:
  name: "Model Evaluation"
  tags: ["evaluation", "prod"]
  dependencies:
    dataprep: dataprep.py
    training: train.py
```

**Rationale:**
- Reuses existing `yanex:` section pattern (already used for CLI defaults)
- Single source of truth for dependency requirements
- Can be validated before script execution
- UI can read config to show required slots
- Clean separation: config = how to run, parameters = what values to use

**Alternative considered:** Python API declaration (`yanex.require_dependencies(...)`)
- **Rejected** to avoid multiple sources of truth and enable pre-execution validation

### 3. CLI Syntax: `--depends-on slot=experiment_id`

```bash
# Single dependency
yanex run train.py --depends-on dataprep=dp1

# Multiple dependencies
yanex run evaluate.py --depends-on dataprep=dp1 --depends-on training=tr1

# Multiple values (creates sweep - one experiment per combination)
yanex run evaluate.py --depends-on dataprep=dp1 --depends-on training=tr1,tr2,tr3

# Nested query with yanex id
yanex run evaluate.py \
  --depends-on dataprep=dp1 \
  --depends-on training=$(yanex id --script train.py --status completed --limit 4)
```

**Rationale:**
- Explicit slot names prevent confusion
- Consistent with parameter syntax (`--param key=value`)
- Composable with shell substitution (`$(yanex id ...)`)
- Multiple values enable dependency sweeps (like parameter sweeps)

### 4. Storage: Separate `dependencies.json` File

Each experiment directory gets an optional `dependencies.json` file:

```
~/.yanex/experiments/eval1/
├── metadata.json
├── dependencies.json      # NEW: Only exists if experiment has dependencies
├── config.yaml
├── metrics.json
└── artifacts/
```

**Rationale:**
- Keeps metadata.json immutable after creation (current design pattern)
- Only created when needed (doesn't clutter experiments without dependencies)
- Follows existing modular storage pattern (metadata.json, metrics.json, config.yaml all separate)
- Can be updated independently (e.g., for validation status)

**Alternative considered:** Embed in metadata.json
- **Rejected** because it breaks the immutability pattern and mixes concerns

### 5. Bidirectional Tracking: Reverse Index

Dependencies are tracked in **both directions**:

**Forward:** `eval1/dependencies.json` records what eval1 depends on:
```json
{
  "resolved_dependencies": {
    "dataprep": "dp1",
    "training": "tr1"
  }
}
```

**Reverse:** `dp1/dependencies.json` records what depends on dp1:
```json
{
  "depended_by": [
    {"experiment_id": "eval1", "slot_name": "dataprep", "created_at": "..."},
    {"experiment_id": "tr1", "slot_name": "dataprep", "created_at": "..."}
  ]
}
```

**Rationale:**
- Enables fast queries in both directions
- Supports cascade operations (delete all dependents)
- Allows warning before deleting depended-on experiments
- Critical for UI "what depends on this?" feature

**Trade-off:** Requires updating multiple files per operation, but enables O(1) queries instead of O(N) scans

### 6. Validation: Three-Level Progressive

**Level 1: Existence & Status (MVP)**
- Does dependency experiment exist?
- Is it completed? (not running, failed, or staged)

**Level 2: Script Matching**
- Does dependency experiment's script match declared slot requirement?
- Example: `dataprep` slot requires `dataprep.py`, verify dep experiment actually ran that script

**Level 3: Artifact Validation (Future)**
- Do required artifacts exist in dependency experiment?
- Pattern matching for globs: `*.parquet`, `model_*.pkl`
- Content validation: schemas, file signatures

**Implementation order:** Level 1 & 2 for MVP, Level 3 as enhancement

### 7. Dependency Sweeps: Cartesian Product

Multiple values for a slot create multiple experiments (like parameter sweeps):

```bash
# Creates 2 experiments
yanex run evaluate.py --depends-on dataprep=dp1 --depends-on training=tr1,tr2
# eval1: dataprep=dp1, training=tr1
# eval2: dataprep=dp1, training=tr2

# Creates 4 experiments (cartesian product)
yanex run compare.py --depends-on model1=tr1,tr2 --depends-on model2=tr3,tr4
# cmp1: model1=tr1, model2=tr3
# cmp2: model1=tr1, model2=tr4
# cmp3: model1=tr2, model2=tr3
# cmp4: model1=tr2, model2=tr4
```

**Rationale:**
- Natural extension of existing parameter sweep mechanism
- Solves "evaluate all trained models" use case
- Consistent behavior with `--param key=value1,value2,value3`

## Implementation Details

### dependencies.json Schema

**For experiments WITH dependencies (consumer):**

```json
{
  "version": "1.0",

  "declared_slots": {
    "dataprep": {
      "script": "dataprep.py",
      "required": true
    },
    "training": {
      "script": "train.py",
      "required": true
    }
  },

  "resolved_dependencies": {
    "dataprep": "dp1",
    "training": "tr1"
  },

  "validation": {
    "validated_at": "2025-11-11T10:30:00Z",
    "status": "valid",
    "checks": [
      {
        "slot": "dataprep",
        "experiment_id": "dp1",
        "script_expected": "dataprep.py",
        "script_actual": "dataprep.py",
        "script_match": true,
        "experiment_status": "completed",
        "valid": true
      },
      {
        "slot": "training",
        "experiment_id": "tr1",
        "script_expected": "train.py",
        "script_actual": "train.py",
        "script_match": true,
        "experiment_status": "completed",
        "valid": true
      }
    ]
  },

  "depended_by": []
}
```

**For experiments that ARE dependencies (provider):**

```json
{
  "version": "1.0",
  "declared_slots": {},
  "resolved_dependencies": {},
  "validation": null,

  "depended_by": [
    {
      "experiment_id": "tr1",
      "slot_name": "dataprep",
      "created_at": "2025-11-11T09:00:00Z"
    },
    {
      "experiment_id": "eval1",
      "slot_name": "dataprep",
      "created_at": "2025-11-11T10:30:00Z"
    }
  ]
}
```

### metadata.json Enhancement (Lightweight Summary)

Add minimal fields for fast filtering without reading `dependencies.json`:

```json
{
  "id": "eval1",
  "name": "Model Evaluation",
  "script_path": "/path/to/evaluate.py",
  "status": "completed",
  ...existing fields...,

  "dependencies_summary": {
    "has_dependencies": true,
    "dependency_count": 2,
    "dependency_slots": ["dataprep", "training"],
    "is_depended_by": false,
    "depended_by_count": 0
  }
}
```

**Purpose:** Enable fast queries like:
- "Show all experiments with dependencies" → filter `has_dependencies: true`
- "Show leaf nodes" → filter `depended_by_count: 0`
- "Show root nodes" → filter `dependency_count: 0`

### Storage Module: `storage_dependencies.py`

Create new storage module following existing composition pattern:

```python
# yanex/core/storage_dependencies.py
from pathlib import Path
from typing import Any
import json

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager

class FileSystemDependencyStorage:
    """File system-based experiment dependency storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        self.directory_manager = directory_manager

    def save_dependencies(
        self,
        experiment_id: str,
        dependencies: dict[str, Any],
        include_archived: bool = False
    ) -> None:
        """Save dependency information."""
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        deps_path = exp_dir / "dependencies.json"

        try:
            with deps_path.open("w", encoding="utf-8") as f:
                json.dump(dependencies, f, indent=2, sort_keys=True)
        except Exception as e:
            raise StorageError(f"Failed to save dependencies: {e}") from e

    def load_dependencies(
        self,
        experiment_id: str,
        include_archived: bool = False
    ) -> dict[str, Any] | None:
        """Load dependency information. Returns None if no dependencies."""
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        deps_path = exp_dir / "dependencies.json"

        if not deps_path.exists():
            return None

        try:
            with deps_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load dependencies: {e}") from e

    def add_dependent(
        self,
        experiment_id: str,
        dependent_id: str,
        slot_name: str,
        include_archived: bool = False
    ) -> None:
        """Add a dependent to this experiment's reverse index."""
        from datetime import datetime

        # Load existing dependencies or create empty structure
        deps = self.load_dependencies(experiment_id, include_archived) or {
            "version": "1.0",
            "declared_slots": {},
            "resolved_dependencies": {},
            "validation": None,
            "depended_by": []
        }

        # Add new dependent
        deps["depended_by"].append({
            "experiment_id": dependent_id,
            "slot_name": slot_name,
            "created_at": datetime.utcnow().isoformat()
        })

        # Save back
        self.save_dependencies(experiment_id, deps, include_archived)

    def remove_dependent(
        self,
        experiment_id: str,
        dependent_id: str,
        include_archived: bool = False
    ) -> None:
        """Remove a dependent from this experiment's reverse index."""
        deps = self.load_dependencies(experiment_id, include_archived)
        if not deps:
            return

        # Filter out the dependent
        deps["depended_by"] = [
            d for d in deps["depended_by"]
            if d["experiment_id"] != dependent_id
        ]

        # Save back
        self.save_dependencies(experiment_id, deps, include_archived)
```

**Integration:** Add to `storage_composition.py` similar to existing storage modules.

### Validation Logic: `DependencyValidator`

Create validator class in new module:

```python
# yanex/core/dependency_validator.py
from typing import Any
from ..utils.exceptions import DependencyError

class DependencyValidator:
    """Validates experiment dependencies."""

    def __init__(self, storage, manager):
        self.storage = storage
        self.manager = manager

    def validate_dependencies(
        self,
        experiment_id: str,
        declared_slots: dict[str, dict[str, Any]],
        resolved_deps: dict[str, str]
    ) -> dict[str, Any]:
        """
        Validate all dependencies for an experiment.

        Returns validation result structure for dependencies.json.
        Raises DependencyError if validation fails.
        """
        from datetime import datetime

        checks = []
        all_valid = True

        for slot_name, slot_config in declared_slots.items():
            # Check slot is provided
            if slot_config.get("required", True):
                if slot_name not in resolved_deps:
                    raise DependencyError(
                        f"Missing required dependency '{slot_name}'"
                    )

            if slot_name not in resolved_deps:
                continue  # Optional and not provided

            dep_id = resolved_deps[slot_name]
            expected_script = slot_config["script"]

            # Validate experiment exists
            if not self.storage.experiment_exists(dep_id):
                raise DependencyError(
                    f"Dependency experiment '{dep_id}' not found"
                )

            # Load dependency metadata
            dep_metadata = self.storage.load_metadata(dep_id)

            # Validate script matches
            actual_script = Path(dep_metadata["script_path"]).name
            script_match = actual_script == expected_script

            if not script_match:
                raise DependencyError(
                    f"Dependency '{slot_name}' requires script '{expected_script}', "
                    f"but experiment {dep_id} ran '{actual_script}'"
                )

            # Validate status
            dep_status = dep_metadata.get("status")
            if dep_status not in ["completed"]:
                raise DependencyError(
                    f"Dependency '{slot_name}' ({dep_id}) has status '{dep_status}', "
                    f"expected 'completed'"
                )

            # Record check
            check = {
                "slot": slot_name,
                "experiment_id": dep_id,
                "script_expected": expected_script,
                "script_actual": actual_script,
                "script_match": script_match,
                "experiment_status": dep_status,
                "valid": script_match and dep_status == "completed"
            }
            checks.append(check)

            if not check["valid"]:
                all_valid = False

        return {
            "validated_at": datetime.utcnow().isoformat(),
            "status": "valid" if all_valid else "invalid",
            "checks": checks
        }
```

### CLI Changes: `yanex run` Command

Modify `yanex/cli/commands/run.py`:

```python
# Add new option
@click.option(
    "--depends-on",
    multiple=True,
    help="Dependency on another experiment (format: slot=experiment_id or slot=id1,id2,id3)"
)
def run(..., depends_on, ...):
    """Run an experiment with dependencies."""

    # Parse dependencies
    parsed_deps = parse_dependency_args(depends_on)
    # Returns: {"dataprep": ["dp1"], "training": ["tr1", "tr2"]}

    # Load config to get declared slots
    if config_path:
        config_data = load_config(config_path)
        declared_slots = config_data.get("yanex", {}).get("dependencies", {})
    else:
        declared_slots = {}

    # Check if we need to create multiple experiments (dependency sweep)
    if should_create_sweep(parsed_deps):
        # Multiple values for at least one slot - create cartesian product
        experiments = create_dependency_sweep(
            script_path,
            parsed_deps,
            declared_slots,
            config_data,
            ...
        )
        # Use existing run_multiple() machinery
        results = yanex.run_multiple(experiments, parallel=parallel)
        display_sweep_results(results)
        return

    # Single experiment - flatten to single values
    resolved_deps = {slot: ids[0] for slot, ids in parsed_deps.items()}

    # Validate dependencies
    validator = DependencyValidator(manager.storage, manager)
    try:
        validation = validator.validate_dependencies(
            experiment_id=None,  # Not created yet
            declared_slots=declared_slots,
            resolved_deps=resolved_deps
        )
    except DependencyError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Create experiment (existing logic)
    metadata = manager.create_experiment(...)

    # Save dependencies
    if resolved_deps:
        dependencies = {
            "version": "1.0",
            "declared_slots": declared_slots,
            "resolved_dependencies": resolved_deps,
            "validation": validation,
            "depended_by": []
        }
        manager.storage.save_dependencies(experiment_id, dependencies)

        # Update reverse index for each dependency
        for slot_name, dep_id in resolved_deps.items():
            manager.storage.add_dependent(dep_id, experiment_id, slot_name)

        # Update metadata summary
        metadata["dependencies_summary"] = {
            "has_dependencies": True,
            "dependency_count": len(resolved_deps),
            "dependency_slots": list(resolved_deps.keys()),
            "is_depended_by": False,
            "depended_by_count": 0
        }
        manager.storage.save_metadata(experiment_id, metadata)

    # Continue with normal execution...
```

**Helper function:**

```python
def parse_dependency_args(depends_on: tuple[str]) -> dict[str, list[str]]:
    """
    Parse --depends-on arguments.

    Input: ("dataprep=dp1", "training=tr1,tr2,tr3")
    Output: {"dataprep": ["dp1"], "training": ["tr1", "tr2", "tr3"]}
    """
    result = {}
    for dep_arg in depends_on:
        if "=" not in dep_arg:
            raise click.UsageError(
                f"Invalid dependency format: '{dep_arg}'. "
                f"Expected format: slot=experiment_id"
            )

        slot, ids_str = dep_arg.split("=", 1)
        ids = [id.strip() for id in ids_str.split(",")]

        if slot in result:
            # Multiple --depends-on for same slot
            result[slot].extend(ids)
        else:
            result[slot] = ids

    return result
```

### Python API: Creating Experiments with Dependencies

Both the context manager API (`create_experiment()`) and batch execution API (`ExperimentSpec`) need to support dependency declaration and fulfillment.

#### `create_experiment()` Enhancement

Add two new parameters to `yanex/api.py`:

```python
def create_experiment(
    script_path: Path,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    dependencies: dict[str, str] | None = None,  # NEW: Declaration (slot -> script)
    depends_on: dict[str, str] | None = None,     # NEW: Fulfillment (slot -> experiment_id)
) -> ExperimentContext:
    """
    Create experiment with optional dependencies.

    Args:
        script_path: Path to script to execute
        name: Optional experiment name
        config: Dictionary of parameters
        config_path: Path to config file (YAML)
        tags: List of tags
        description: Experiment description
        dependencies: Declare required dependencies (slot -> script name).
                     If None, reads from config file yanex.dependencies section.
                     If provided, overrides config file declaration.
        depends_on: Fulfill dependencies (slot -> experiment_id).
                   Must match declared dependencies (from dependencies param or config).

    Raises:
        ValueError: If depends_on provided without dependencies declaration
        DependencyError: If dependency validation fails

    Example:
        # Declare inline (no config file)
        with yanex.create_experiment(
            script_path=Path("evaluate.py"),
            dependencies={"dataprep": "dataprep.py", "train": "train.py"},
            depends_on={"dataprep": "dp1", "train": "tr1"}
        ) as exp:
            exp.start()
            # ... evaluation code ...

        # Declare in config, fulfill in API
        with yanex.create_experiment(
            script_path=Path("evaluate.py"),
            config_path=Path("config.yaml"),  # Has yanex.dependencies
            depends_on={"dataprep": "dp1", "train": "tr1"}
        ) as exp:
            exp.start()
            # ... evaluation code ...
    """
    manager = ExperimentManager()

    # Resolve dependency declaration - hierarchy:
    # 1. dependencies parameter (highest priority)
    # 2. config file yanex.dependencies section
    # 3. None (no dependencies)

    declared_slots = dependencies
    if declared_slots is None and config_path:
        # Try to load from config file
        config_data = load_yaml_config(config_path)
        yanex_section = config_data.get("yanex", {})
        config_dependencies = yanex_section.get("dependencies", {})

        if config_dependencies:
            # Normalize to full schema
            declared_slots = {
                slot: {"script": script, "required": True}
                for slot, script in config_dependencies.items()
            }

    # Normalize inline declaration to full schema
    if declared_slots and isinstance(next(iter(declared_slots.values())), str):
        # Simple format: {"dataprep": "dataprep.py"}
        declared_slots = {
            slot: {"script": script, "required": True}
            for slot, script in declared_slots.items()
        }

    # Validate dependencies if fulfillment provided
    if depends_on:
        if not declared_slots:
            raise ValueError(
                "Cannot provide depends_on without declaring dependencies. "
                "Either add 'dependencies' parameter or declare in config file "
                "under 'yanex.dependencies' section."
            )

        validator = DependencyValidator(manager.storage, manager)
        validation = validator.validate_dependencies(
            experiment_id=None,
            declared_slots=declared_slots,
            resolved_deps=depends_on
        )
    else:
        validation = None

    # Create experiment (existing logic)
    metadata = manager.create_experiment(
        script_path=script_path,
        name=name,
        config=config,
        tags=tags,
        description=description,
        ...
    )
    experiment_id = metadata["id"]

    # Store dependencies if provided
    if depends_on:
        dependencies_data = {
            "version": "1.0",
            "declared_slots": declared_slots,
            "resolved_dependencies": depends_on,
            "validation": validation,
            "depended_by": []
        }
        manager.storage.save_dependencies(experiment_id, dependencies_data)

        # Update reverse index for each dependency
        for slot_name, dep_id in depends_on.items():
            manager.storage.add_dependent(dep_id, experiment_id, slot_name)

        # Update metadata summary
        metadata["dependencies_summary"] = {
            "has_dependencies": True,
            "dependency_count": len(depends_on),
            "dependency_slots": list(depends_on.keys()),
            "is_depended_by": False,
            "depended_by_count": 0
        }
        manager.storage.save_metadata(experiment_id, metadata)

    return ExperimentContext(experiment_id, manager)
```

**Usage Examples:**

```python
# Example 1: Simple inline declaration
import yanex
from pathlib import Path

with yanex.create_experiment(
    script_path=Path("train.py"),
    dependencies={"dataprep": "dataprep.py"},
    depends_on={"dataprep": "dp1"}
) as exp:
    exp.start()

    # Access dependency artifacts
    deps = yanex.get_dependencies()
    data_dir = deps["dataprep"].artifacts_dir
    train_data = pd.read_parquet(data_dir / "train_data.parquet")

    # ... training code ...
    yanex.log_metrics({"accuracy": 0.95})

# Example 2: Config file declaration
# config.yaml:
# yanex:
#   dependencies:
#     dataprep: dataprep.py
#     baseline: train.py

with yanex.create_experiment(
    script_path=Path("compare.py"),
    config_path=Path("config.yaml"),
    depends_on={"dataprep": "dp1", "baseline": "tr1"}
) as exp:
    exp.start()
    # ... comparison code ...

# Example 3: Override config declaration
with yanex.create_experiment(
    script_path=Path("evaluate.py"),
    config_path=Path("config.yaml"),  # Declares 2 dependencies
    dependencies={"train": "train.py"},  # Override: only need 1
    depends_on={"train": "tr1"}
) as exp:
    exp.start()
    # ... evaluation code ...
```

#### `ExperimentSpec` Enhancement

Update `yanex/executor.py` to support dependencies:

```python
@dataclass
class ExperimentSpec:
    """Specification for a single experiment execution."""

    script_path: Path
    config: dict[str, Any] | None = None
    script_args: list[str] | None = None
    name: str | None = None
    tags: list[str] | None = None
    description: str | None = None
    dependencies: dict[str, str] | None = None  # NEW: Declaration
    depends_on: dict[str, str] | None = None     # NEW: Fulfillment
```

**Usage with `run_multiple()`:**

```python
import yanex
from pathlib import Path

# Create dataprep experiment (no dependencies)
dataprep_spec = yanex.ExperimentSpec(
    script_path=Path("dataprep.py"),
    config={"dataset": "yelp"},
    name="YELP Data Prep"
)

# Run dataprep first
dataprep_results = yanex.run_multiple([dataprep_spec])
dp_id = dataprep_results[0].experiment_id

# Create training experiments (depend on dataprep)
training_specs = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": lr},
        name=f"Train LR={lr}",
        tags=["training", "sweep"],
        dependencies={"dataprep": "dataprep.py"},  # Declare
        depends_on={"dataprep": dp_id}              # Fulfill
    )
    for lr in [0.001, 0.005, 0.01]
]

# Run training in parallel
training_results = yanex.run_multiple(training_specs, parallel=3)

# Create evaluation experiments (depend on both dataprep and training)
eval_specs = [
    yanex.ExperimentSpec(
        script_path=Path("evaluate.py"),
        name=f"Evaluate {result.name}",
        tags=["evaluation"],
        dependencies={
            "dataprep": "dataprep.py",
            "train": "train.py"
        },
        depends_on={
            "dataprep": dp_id,
            "train": result.experiment_id
        }
    )
    for result in training_results
    if result.status == "completed"
]

# Run evaluations
eval_results = yanex.run_multiple(eval_specs, parallel=3)
```

#### Dependency Resolution Hierarchy

The API follows this hierarchy for dependency declaration:

1. **`dependencies` parameter** (highest priority) - overrides everything
2. **Config file `yanex.dependencies`** - if `config_path` provided
3. **None** - no dependencies

**Examples:**

```python
# Scenario 1: Both inline and config - inline wins
# config.yaml has: yanex.dependencies = {"dataprep": "dataprep.py", "train": "train.py"}
with yanex.create_experiment(
    config_path=Path("config.yaml"),
    dependencies={"train": "train.py"},  # OVERRIDES config
    depends_on={"train": "tr1"}
) as exp:
    # Only "train" dependency required, "dataprep" ignored
    pass

# Scenario 2: Only config
with yanex.create_experiment(
    config_path=Path("config.yaml"),
    depends_on={"dataprep": "dp1", "train": "tr1"}
) as exp:
    # Uses dependencies from config.yaml
    pass

# Scenario 3: Only inline
with yanex.create_experiment(
    dependencies={"dataprep": "dataprep.py"},
    depends_on={"dataprep": "dp1"}
) as exp:
    # Uses inline declaration
    pass

# Scenario 4: Error - fulfillment without declaration
with yanex.create_experiment(
    depends_on={"dataprep": "dp1"}  # No declaration!
) as exp:
    # Raises ValueError
    pass
```

#### Validation Behavior

**Requirement:** Must declare dependencies before fulfilling them.

```python
# ✅ Valid: Declaration + Fulfillment
yanex.create_experiment(
    dependencies={"dataprep": "dataprep.py"},
    depends_on={"dataprep": "dp1"}
)

# ✅ Valid: Declaration in config + Fulfillment in API
yanex.create_experiment(
    config_path=Path("config.yaml"),  # Has dependencies
    depends_on={"dataprep": "dp1"}
)

# ✅ Valid: Declaration only (no fulfillment = no dependencies)
yanex.create_experiment(
    dependencies={"dataprep": "dataprep.py"}
    # depends_on not provided = experiment has no dependencies
)

# ❌ Invalid: Fulfillment without declaration
yanex.create_experiment(
    depends_on={"dataprep": "dp1"}  # Where's the declaration?
)
# Raises: ValueError("Cannot provide depends_on without declaring dependencies...")

# ❌ Invalid: Wrong slot name
yanex.create_experiment(
    dependencies={"dataprep": "dataprep.py"},
    depends_on={"data": "dp1"}  # Slot mismatch!
)
# Raises: DependencyError("Missing required dependency 'dataprep'")
```

### Python API: `yanex.get_dependencies()`

Add to `yanex/api.py`:

```python
def get_dependencies() -> dict[str, "DependencyReference"]:
    """
    Get dependencies for the current experiment.

    Returns:
        Dictionary mapping slot names to DependencyReference objects

    Example:
        deps = yanex.get_dependencies()
        dataprep_dir = deps["dataprep"].artifacts_dir
        train_data = pd.read_parquet(dataprep_dir / "train_data.parquet")
    """
    experiment_id = os.getenv("YANEX_EXPERIMENT_ID")
    if not experiment_id:
        return {}  # Not in yanex context

    manager = ExperimentManager()
    deps_data = manager.storage.load_dependencies(experiment_id)

    if not deps_data:
        return {}

    resolved = deps_data.get("resolved_dependencies", {})
    return {
        slot: DependencyReference(dep_id, slot, manager)
        for slot, dep_id in resolved.items()
    }


class DependencyReference:
    """Reference to a dependency experiment."""

    def __init__(self, experiment_id: str, slot_name: str, manager):
        self.experiment_id = experiment_id
        self.slot_name = slot_name
        self._manager = manager

    @property
    def artifacts_dir(self) -> Path:
        """Get path to dependency's artifacts directory."""
        exp_dir = self._manager.storage.get_experiment_directory(
            self.experiment_id
        )
        return exp_dir / "artifacts"

    @property
    def artifacts(self) -> list[Path]:
        """List all artifacts in dependency."""
        artifacts_dir = self.artifacts_dir
        if not artifacts_dir.exists():
            return []
        return sorted(artifacts_dir.iterdir())

    def artifact_path(self, name: str) -> Path:
        """Get path to specific artifact."""
        return self.artifacts_dir / name

    def load_artifact(self, name: str) -> Path:
        """Get path to artifact, raising error if not found."""
        path = self.artifact_path(name)
        if not path.exists():
            raise FileNotFoundError(
                f"Artifact '{name}' not found in dependency {self.experiment_id}"
            )
        return path
```

### Results API: Querying Dependency Graphs

The `Experiment` class in `yanex/results/experiment.py` should provide methods for querying dependency relationships post-hoc. This enables analysis notebooks, pipeline debugging, and reporting.

#### `Experiment.get_dependencies()` - Query Upstream

```python
def get_dependencies(
    self,
    recursive: bool = False,
    max_depth: int | None = None
) -> list["Experiment"]:
    """
    Get experiments this one depends on (upstream/parents).

    Args:
        recursive: Include all ancestors (default: direct only)
        max_depth: Limit recursion depth (None = unlimited)

    Returns:
        List of Experiment objects

    Example:
        exp = Experiment("eval1")
        deps = exp.get_dependencies()
        # [Experiment("dp1"), Experiment("tr1")]

        all_deps = exp.get_dependencies(recursive=True)
        # [Experiment("dp1"), Experiment("prep0")] if dp1 depends on prep0
    """
    result = []

    if not recursive:
        # Direct dependencies only
        deps_data = self._manager.storage.load_dependencies(
            self.id, include_archived=self.archived
        )
        if deps_data and deps_data.get("resolved_dependencies"):
            for dep_id in deps_data["resolved_dependencies"].values():
                result.append(Experiment(dep_id, self._manager))
        return result

    # Recursive traversal
    visited = set()

    def traverse(exp_id: str, depth: int = 0):
        if exp_id in visited:
            return
        if max_depth is not None and depth >= max_depth:
            return

        visited.add(exp_id)

        deps_data = self._manager.storage.load_dependencies(exp_id)
        if deps_data and deps_data.get("resolved_dependencies"):
            for dep_id in deps_data["resolved_dependencies"].values():
                result.append(Experiment(dep_id, self._manager))
                traverse(dep_id, depth + 1)

    traverse(self.id)
    return result
```

#### `Experiment.get_dependents()` - Query Downstream

```python
def get_dependents(
    self,
    recursive: bool = False,
    max_depth: int | None = None
) -> list["Experiment"]:
    """
    Get experiments that depend on this one (downstream/children).

    Args:
        recursive: Include all descendants (default: direct only)
        max_depth: Limit recursion depth (None = unlimited)

    Returns:
        List of Experiment objects

    Example:
        exp = Experiment("dp1")
        deps = exp.get_dependents()
        # [Experiment("tr1"), Experiment("tr2"), Experiment("eval1")]

        all_deps = exp.get_dependents(recursive=True)
        # Includes children of tr1, tr2, eval1 recursively
    """
    result = []

    if not recursive:
        # Direct dependents only
        deps_data = self._manager.storage.load_dependencies(
            self.id, include_archived=self.archived
        )
        if deps_data and deps_data.get("depended_by"):
            for dep_info in deps_data["depended_by"]:
                result.append(Experiment(dep_info["experiment_id"], self._manager))
        return result

    # Recursive traversal
    visited = set()

    def traverse(exp_id: str, depth: int = 0):
        if exp_id in visited:
            return
        if max_depth is not None and depth >= max_depth:
            return

        visited.add(exp_id)

        deps_data = self._manager.storage.load_dependencies(exp_id)
        if deps_data and deps_data.get("depended_by"):
            for dep_info in deps_data["depended_by"]:
                dep_id = dep_info["experiment_id"]
                result.append(Experiment(dep_id, self._manager))
                traverse(dep_id, depth + 1)

    traverse(self.id)
    return result
```

#### `Experiment.get_dependency_info()` - Detailed Metadata

```python
def get_dependency_info(self) -> dict[str, dict]:
    """
    Get detailed dependency information with slot names and metadata.

    Returns:
        Dictionary mapping slot names to dependency details

    Example:
        exp = Experiment("eval1")
        info = exp.get_dependency_info()
        # {
        #   "dataprep": {
        #       "experiment_id": "dp1",
        #       "experiment": Experiment("dp1"),
        #       "slot": "dataprep",
        #       "script": "dataprep.py",
        #       "status": "completed",
        #       "validated": True
        #   },
        #   "training": {...}
        # }
    """
    deps_data = self._manager.storage.load_dependencies(
        self.id, include_archived=self.archived
    )

    if not deps_data or not deps_data.get("resolved_dependencies"):
        return {}

    result = {}
    declared_slots = deps_data.get("declared_slots", {})
    validation_checks = deps_data.get("validation", {}).get("checks", [])

    for slot, dep_id in deps_data["resolved_dependencies"].items():
        dep_exp = Experiment(dep_id, self._manager)

        # Find validation info for this slot
        validation_info = None
        for check in validation_checks:
            if check["slot"] == slot:
                validation_info = check
                break

        result[slot] = {
            "experiment_id": dep_id,
            "experiment": dep_exp,
            "slot": slot,
            "script": declared_slots.get(slot, {}).get("script"),
            "status": dep_exp.status,
            "validated": validation_info["valid"] if validation_info else None
        }

    return result
```

#### `Experiment.get_dependency_graph()` / `Experiment.get_pipeline()` - Full Subgraph

```python
def get_dependency_graph(self) -> dict:
    """
    Get full connected subgraph (pipeline) containing this experiment.

    Traverses both upstream and downstream to find all connected experiments.
    Returns DAG as edge list with rich node objects.

    Returns:
        {
            "nodes": {experiment_id: Experiment, ...},
            "edges": [{"source": id, "target": id, "slot": name}, ...],
            "root_nodes": [experiment_id, ...],
            "leaf_nodes": [experiment_id, ...]
        }

    Example:
        exp = Experiment("tr1")  # Any experiment in pipeline
        graph = exp.get_dependency_graph()

        # Access nodes
        dataprep = graph["nodes"]["dp1"]
        print(dataprep.status)

        # Find entry/exit points
        roots = [graph["nodes"][id] for id in graph["root_nodes"]]
        leaves = [graph["nodes"][id] for id in graph["leaf_nodes"]]

        # Iterate edges
        for edge in graph["edges"]:
            print(f"{edge['source']} -> {edge['target']} (slot: {edge['slot']})")
    """
    visited = set()
    nodes = {}
    edges = []

    def traverse(exp_id: str):
        """Recursively traverse both directions."""
        if exp_id in visited:
            return
        visited.add(exp_id)

        # Add node
        exp = Experiment(exp_id, self._manager)
        nodes[exp_id] = exp

        # Traverse upstream (dependencies)
        deps_data = self._manager.storage.load_dependencies(exp_id)
        if deps_data and deps_data.get("resolved_dependencies"):
            for slot, dep_id in deps_data["resolved_dependencies"].items():
                edges.append({"source": dep_id, "target": exp_id, "slot": slot})
                traverse(dep_id)

        # Traverse downstream (dependents)
        if deps_data and deps_data.get("depended_by"):
            for dep_info in deps_data["depended_by"]:
                dep_id = dep_info["experiment_id"]
                slot = dep_info["slot_name"]
                edges.append({"source": exp_id, "target": dep_id, "slot": slot})
                traverse(dep_id)

    # Start traversal from this experiment
    traverse(self.id)

    # Find root and leaf nodes
    all_sources = {e["source"] for e in edges}
    all_targets = {e["target"] for e in edges}
    root_nodes = list(all_sources - all_targets)
    leaf_nodes = list(all_targets - all_sources)

    # Handle single-node graph (no dependencies or dependents)
    if not edges:
        root_nodes = [self.id]
        leaf_nodes = [self.id]

    return {
        "nodes": nodes,
        "edges": edges,
        "root_nodes": root_nodes,
        "leaf_nodes": leaf_nodes
    }

def get_pipeline(self) -> dict:
    """
    Alias for get_dependency_graph().

    More intuitive name for getting the full experiment pipeline.
    """
    return self.get_dependency_graph()
```

#### Module-Level `get_pipeline()` Function

Add to `yanex/results/__init__.py`:

```python
def get_pipeline(experiment_id: str) -> dict:
    """
    Get full pipeline (dependency graph) for any experiment.

    Convenience function - equivalent to Experiment(id).get_pipeline()

    Args:
        experiment_id: Any experiment in the pipeline

    Returns:
        Full connected subgraph with nodes, edges, roots, and leaves

    Example:
        import yanex.results as yr

        # Get pipeline from any experiment in it
        pipeline = yr.get_pipeline("tr1")

        # All experiments in pipeline
        all_exps = list(pipeline["nodes"].values())

        # Entry points
        roots = [pipeline["nodes"][id] for id in pipeline["root_nodes"]]

        # Exit points
        leaves = [pipeline["nodes"][id] for id in pipeline["leaf_nodes"]]

        # Count experiments by script
        from collections import Counter
        scripts = Counter(exp.script_path.name for exp in all_exps)
        # Counter({'train.py': 3, 'dataprep.py': 1, 'evaluate.py': 2})
    """
    from .experiment import Experiment
    exp = Experiment(experiment_id)
    return exp.get_pipeline()
```

#### Usage Examples

```python
# Example 1: Analyze entire pipeline
import yanex.results as yr

pipeline = yr.get_pipeline("tr1")  # Any experiment in the pipeline

print(f"Pipeline has {len(pipeline['nodes'])} experiments")
print(f"Entry points: {pipeline['root_nodes']}")
print(f"Final outputs: {pipeline['leaf_nodes']}")

# Example 2: Find all failed experiments in pipeline
failed = [
    exp for exp in pipeline["nodes"].values()
    if exp.status == "failed"
]
if failed:
    print(f"Failed experiments: {[e.id for e in failed]}")

# Example 3: Compare all training runs that used same dataprep
from yanex.results import Experiment

dataprep = Experiment("dp1")
training_runs = [
    exp for exp in dataprep.get_dependents()
    if "train.py" in str(exp.script_path)
]

for run in training_runs:
    metrics = run.get_metrics()
    print(f"{run.id}: accuracy={metrics.get('accuracy', 'N/A')}")

# Example 4: Get dependency info with slot names
exp = Experiment("eval1")
dep_info = exp.get_dependency_info()

for slot, info in dep_info.items():
    print(f"Slot '{slot}':")
    print(f"  Experiment: {info['experiment_id']}")
    print(f"  Script: {info['script']}")
    print(f"  Status: {info['status']}")
    print(f"  Validated: {info['validated']}")

# Example 5: Export for UI visualization (React-Flow format)
import json

pipeline = yr.get_pipeline("eval1")

ui_format = {
    "nodes": [
        {
            "id": exp.id,
            "data": {
                "name": exp.name or exp.id,
                "script": exp.script_path.name if exp.script_path else "unknown",
                "status": exp.status,
                "created_at": exp.created_at.isoformat() if exp.created_at else None
            },
            "position": {"x": 0, "y": 0}  # Layout done client-side
        }
        for exp in pipeline["nodes"].values()
    ],
    "edges": [
        {
            "id": f"{e['source']}-{e['target']}",
            "source": e["source"],
            "target": e["target"],
            "label": e["slot"]
        }
        for e in pipeline["edges"]
    ]
}

# Can be sent to UI
json_str = json.dumps(ui_format, indent=2)

# Example 6: Delete entire pipeline
pipeline = yr.get_pipeline("dp1")
root = pipeline["nodes"][pipeline["root_nodes"][0]]

# Delete from root will cascade to all dependents
# (requires --cascade flag implemented in Phase 2)
```

#### JSON Export Format

The edge list format is **JSON-serializable** with minor transformation:

```python
import json
from datetime import datetime

def serialize_pipeline(pipeline: dict) -> str:
    """Convert pipeline to JSON (Experiment objects → dicts)."""

    serializable = {
        "nodes": [
            {
                "id": exp.id,
                "name": exp.name,
                "script": str(exp.script_path) if exp.script_path else None,
                "status": exp.status,
                "tags": exp.tags,
                "created_at": exp.created_at.isoformat() if exp.created_at else None
            }
            for exp in pipeline["nodes"].values()
        ],
        "edges": pipeline["edges"],  # Already JSON-serializable
        "root_nodes": pipeline["root_nodes"],
        "leaf_nodes": pipeline["leaf_nodes"]
    }

    return json.dumps(serializable, indent=2)
```

### New Command: `yanex id`

Create `yanex/cli/commands/id.py`:

```python
"""Command to output experiment IDs matching filters."""
import click
from ..formatters.experiment_list import get_experiment_ids

@click.command()
@click.option("--script", help="Filter by script name/pattern")
@click.option("--status", help="Filter by status")
@click.option("--tags", multiple=True, help="Filter by tags")
@click.option("--since", help="Filter by date (e.g., '2 days ago')")
@click.option("--limit", type=int, help="Limit number of results")
@click.option("--format", type=click.Choice(["line", "csv", "json"]), default="line")
@click.option("--depends-on", help="Filter by dependency on experiment ID")
@click.option("--depends-on-script", help="Filter by dependency on script type")
@click.option("--root", is_flag=True, help="Only root experiments (no dependencies)")
@click.option("--leaf", is_flag=True, help="Only leaf experiments (nothing depends on them)")
def id_command(script, status, tags, since, limit, format, depends_on,
               depends_on_script, root, leaf):
    """Output experiment IDs matching filters (for composition with other commands)."""

    # Get filtered experiments (reuse existing filter logic)
    experiments = get_filtered_experiments(
        script=script,
        status=status,
        tags=tags,
        since=since,
        depends_on=depends_on,
        depends_on_script=depends_on_script,
        root=root,
        leaf=leaf
    )

    # Apply limit
    if limit:
        experiments = experiments[:limit]

    # Extract IDs
    ids = [exp.id for exp in experiments]

    # Format output
    if format == "csv":
        click.echo(",".join(ids))
    elif format == "json":
        import json
        click.echo(json.dumps(ids))
    else:  # line
        for id in ids:
            click.echo(id)
```

**Usage:**
```bash
# Basic
yanex id --script train.py --status completed
# tr1
# tr2
# tr3

# Composition
yanex run evaluate.py --depends-on training=$(yanex id --script train.py --limit 1 --format csv)
```

### Command Enhancements

#### `yanex list`

**Add dependency column:**

```python
def format_experiment_list(experiments):
    """Format experiments with dependency information."""
    headers = ["ID", "Script", "Status", "Created", "Dependencies"]

    rows = []
    for exp in experiments:
        metadata = load_metadata(exp.id)
        deps_summary = metadata.get("dependencies_summary", {})

        # Format dependency column
        if deps_summary.get("has_dependencies"):
            deps = load_dependencies(exp.id)
            dep_ids = list(deps["resolved_dependencies"].values())
            dep_str = f"→ {', '.join(dep_ids)}"
        else:
            dep_str = "-"

        rows.append([
            exp.id,
            Path(metadata["script_path"]).name,
            metadata["status"],
            format_date(metadata["created_at"]),
            dep_str
        ])

    return tabulate(rows, headers=headers)
```

**Add tree mode:**

```python
@click.option("--tree", is_flag=True, help="Show as dependency tree")
def list_command(..., tree):
    if tree:
        display_tree_view(experiments)
    else:
        display_flat_view(experiments)
```

#### `yanex show`

**Add dependencies section:**

```python
def show_experiment(experiment_id):
    """Show experiment details including dependencies."""

    # ... existing metadata display ...

    # Show dependencies if any
    deps = load_dependencies(experiment_id)
    if deps and deps.get("resolved_dependencies"):
        click.echo("\nDependencies:")
        for slot, dep_id in deps["resolved_dependencies"].items():
            dep_meta = load_metadata(dep_id)
            status_icon = "✓" if deps["validation"]["status"] == "valid" else "✗"
            click.echo(f"  {status_icon} {slot} ({dep_id}):")
            click.echo(f"      Script: {Path(dep_meta['script_path']).name}")
            click.echo(f"      Status: {dep_meta['status']}")

    # Show what depends on this
    if deps and deps.get("depended_by"):
        click.echo("\nDepended By:")
        for dep in deps["depended_by"]:
            click.echo(f"  {dep['experiment_id']} (slot: {dep['slot_name']})")
```

#### `yanex delete`

**Add cascade options:**

```python
@click.option("--cascade", is_flag=True, help="Delete dependent experiments too")
@click.option("--force", is_flag=True, help="Force delete even if depended on")
def delete_command(experiment_id, cascade, force):
    """Delete experiment with dependency checks."""

    # Check if anything depends on this
    deps = load_dependencies(experiment_id)
    if deps and deps.get("depended_by"):
        dependents = deps["depended_by"]

        if cascade:
            # Show what will be deleted
            click.echo(f"Warning: This will delete {experiment_id} AND {len(dependents)} dependents:")
            for dep in dependents:
                click.echo(f"  - {dep['experiment_id']}")

            if not click.confirm("Continue?"):
                return

            # Delete all dependents recursively
            for dep in dependents:
                delete_experiment_cascade(dep["experiment_id"])

        elif not force:
            # Prevent deletion
            click.echo(f"Error: Cannot delete {experiment_id} - depended on by:", err=True)
            for dep in dependents:
                click.echo(f"  - {dep['experiment_id']}", err=True)
            click.echo("Use --force to delete anyway, or --cascade to delete dependents", err=True)
            sys.exit(1)

        else:
            # Force - delete but leave orphans
            click.echo(f"Warning: Deleting {experiment_id} will orphan {len(dependents)} experiments")

    # Proceed with deletion
    delete_experiment(experiment_id)

    # Update reverse indexes
    if has_dependencies(experiment_id):
        deps = load_dependencies(experiment_id)
        for dep_id in deps["resolved_dependencies"].values():
            remove_dependent(dep_id, experiment_id)
```

### New Filters

Add to existing filter infrastructure:

```python
# In filter utilities
def filter_by_dependencies(experiments, **kwargs):
    """Filter experiments by dependency relationships."""

    depends_on = kwargs.get("depends_on")
    depends_on_script = kwargs.get("depends_on_script")
    root = kwargs.get("root")
    leaf = kwargs.get("leaf")

    filtered = []
    for exp in experiments:
        metadata = load_metadata(exp.id)
        deps_summary = metadata.get("dependencies_summary", {})

        # Root filter
        if root and deps_summary.get("dependency_count", 0) > 0:
            continue

        # Leaf filter
        if leaf and deps_summary.get("depended_by_count", 0) > 0:
            continue

        # Depends on specific experiment
        if depends_on:
            deps = load_dependencies(exp.id)
            if not deps or depends_on not in deps["resolved_dependencies"].values():
                continue

        # Depends on script type
        if depends_on_script:
            deps = load_dependencies(exp.id)
            if not deps:
                continue

            has_script = False
            for dep_id in deps["resolved_dependencies"].values():
                dep_meta = load_metadata(dep_id)
                if Path(dep_meta["script_path"]).name == depends_on_script:
                    has_script = True
                    break

            if not has_script:
                continue

        filtered.append(exp)

    return filtered
```

### UI Integration: DAG Endpoint

Add to `yanex/web/api.py`:

```python
@app.get("/api/experiments/dag")
def get_experiment_dag(
    script: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[List[str]] = Query(None)
):
    """Get experiment DAG for visualization."""

    # Get filtered experiments
    experiments = get_experiments_filtered(script, status, tags)

    nodes = []
    edges = []

    for exp in experiments:
        metadata = manager.storage.load_metadata(exp.id)

        # Create node
        nodes.append({
            "id": exp.id,
            "type": "experiment",
            "data": {
                "name": metadata.get("name") or f"Experiment {exp.id}",
                "script": Path(metadata["script_path"]).name,
                "status": metadata["status"],
                "created_at": metadata["created_at"],
                "tags": metadata.get("tags", []),
                "has_dependencies": metadata.get("dependencies_summary", {}).get("has_dependencies", False)
            },
            "position": {"x": 0, "y": 0}  # Layout done client-side
        })

        # Create edges from dependencies
        deps = manager.storage.load_dependencies(exp.id)
        if deps and deps.get("resolved_dependencies"):
            for slot_name, dep_id in deps["resolved_dependencies"].items():
                edges.append({
                    "id": f"{dep_id}-{exp.id}-{slot_name}",
                    "source": dep_id,
                    "target": exp.id,
                    "label": slot_name,
                    "data": {
                        "slot": slot_name,
                        "validated": deps["validation"]["status"] == "valid"
                    }
                })

    return {"nodes": nodes, "edges": edges}
```

**React-Flow component (yanex/web/src/components/ExperimentDAG.tsx):**

```typescript
import ReactFlow, { Node, Edge } from 'reactflow';
import 'reactflow/dist/style.css';

interface ExperimentNode {
  id: string;
  type: string;
  data: {
    name: string;
    script: string;
    status: string;
    created_at: string;
    tags: string[];
  };
  position: { x: number; y: number };
}

const nodeColor = {
  'dataprep.py': '#3b82f6',
  'train.py': '#10b981',
  'evaluate.py': '#f59e0b',
  'default': '#6b7280'
};

function ExperimentNode({ data }: { data: ExperimentNode['data'] }) {
  const color = nodeColor[data.script] || nodeColor.default;

  return (
    <div className="px-4 py-2 rounded shadow" style={{ backgroundColor: color }}>
      <div className="font-bold">{data.name}</div>
      <div className="text-sm">{data.script}</div>
      <div className="text-xs">{data.status}</div>
    </div>
  );
}

export function ExperimentDAG() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    fetch('/api/experiments/dag')
      .then(res => res.json())
      .then(data => {
        setNodes(data.nodes);
        setEdges(data.edges);
      });
  }, []);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={{ experiment: ExperimentNode }}
      fitView
    />
  );
}
```

## Error Handling

### Comprehensive Error Messages

All validation errors should provide:
1. What went wrong
2. How to fix it
3. Helpful commands to find solutions

**Example error messages:**

```
Error: Missing required dependency 'train'

Expected dependencies (from config):
  ✓ dataprep: dataprep.py (provided: dp1)
  ✗ train: train.py (missing)

To fix, add: --depends-on train=<experiment-id>

Find training experiments:
  yanex id --script train.py --status completed
```

```
Error: Dependency 'dataprep' requires script 'dataprep.py', but experiment tr1 ran 'train.py'

Looks like you may have swapped dependencies. Did you mean:
  --depends-on dataprep=dp1 --depends-on training=tr1
```

```
Error: Dependency experiment 'xyz999' not found

Recent experiments:
  dp1 (dataprep.py, completed, 1 hour ago)
  tr1 (train.py, completed, 30 min ago)

List all experiments: yanex list
```

### Exception Classes

```python
# yanex/utils/exceptions.py

class DependencyError(Exception):
    """Raised when dependency validation fails."""
    pass

class MissingDependencyError(DependencyError):
    """Raised when required dependency not provided."""
    pass

class InvalidDependencyError(DependencyError):
    """Raised when dependency experiment invalid."""
    pass

class CircularDependencyError(DependencyError):
    """Raised when circular dependency detected."""
    pass
```

## Testing Strategy

### Unit Tests

**Test dependency parsing:**
```python
# tests/cli/test_dependency_parsing.py
def test_parse_single_dependency():
    result = parse_dependency_args(("dataprep=dp1",))
    assert result == {"dataprep": ["dp1"]}

def test_parse_multiple_dependencies():
    result = parse_dependency_args(("dataprep=dp1", "training=tr1"))
    assert result == {"dataprep": ["dp1"], "training": ["tr1"]}

def test_parse_sweep_dependencies():
    result = parse_dependency_args(("training=tr1,tr2,tr3",))
    assert result == {"training": ["tr1", "tr2", "tr3"]}
```

**Test validation:**
```python
# tests/core/test_dependency_validation.py
def test_validate_dependencies_success(temp_dir, git_repo):
    # Create dependency experiment
    manager = ExperimentManager(temp_dir)
    dep_metadata = manager.create_experiment(
        script_path=Path("dataprep.py"),
        ...
    )
    manager.complete_experiment(dep_metadata["id"])

    # Validate
    validator = DependencyValidator(manager.storage, manager)
    result = validator.validate_dependencies(
        experiment_id=None,
        declared_slots={"dataprep": {"script": "dataprep.py", "required": True}},
        resolved_deps={"dataprep": dep_metadata["id"]}
    )

    assert result["status"] == "valid"

def test_validate_missing_dependency():
    validator = DependencyValidator(manager.storage, manager)
    with pytest.raises(MissingDependencyError):
        validator.validate_dependencies(
            experiment_id=None,
            declared_slots={"dataprep": {"script": "dataprep.py", "required": True}},
            resolved_deps={}  # Missing!
        )

def test_validate_wrong_script():
    # Create experiment with wrong script
    dep_metadata = manager.create_experiment(
        script_path=Path("train.py"),  # Wrong!
        ...
    )

    validator = DependencyValidator(manager.storage, manager)
    with pytest.raises(InvalidDependencyError):
        validator.validate_dependencies(
            experiment_id=None,
            declared_slots={"dataprep": {"script": "dataprep.py", "required": True}},
            resolved_deps={"dataprep": dep_metadata["id"]}
        )
```

**Test storage:**
```python
# tests/core/test_dependency_storage.py
def test_save_and_load_dependencies(temp_dir):
    storage = create_storage(temp_dir)

    deps = {
        "version": "1.0",
        "declared_slots": {"dataprep": {"script": "dataprep.py", "required": True}},
        "resolved_dependencies": {"dataprep": "dp1"},
        "depended_by": []
    }

    storage.save_dependencies("exp1", deps)
    loaded = storage.load_dependencies("exp1")

    assert loaded == deps

def test_add_dependent(temp_dir):
    storage = create_storage(temp_dir)

    # Create initial empty dependencies
    storage.save_dependencies("dp1", {
        "version": "1.0",
        "declared_slots": {},
        "resolved_dependencies": {},
        "depended_by": []
    })

    # Add dependent
    storage.add_dependent("dp1", "tr1", "dataprep")

    # Verify
    deps = storage.load_dependencies("dp1")
    assert len(deps["depended_by"]) == 1
    assert deps["depended_by"][0]["experiment_id"] == "tr1"
    assert deps["depended_by"][0]["slot_name"] == "dataprep"
```

**Test Python API:**
```python
# tests/api/test_dependency_api.py
def test_create_experiment_with_inline_dependencies(temp_dir, git_repo):
    """Test create_experiment with inline dependency declaration."""
    # Create dependency experiment
    with yanex.create_experiment(
        script_path=Path("dataprep.py"),
        name="Data Prep"
    ) as dep_exp:
        dep_exp.start()
        dep_id = dep_exp.experiment_id

    manager = ExperimentManager(temp_dir)
    manager.complete_experiment(dep_id)

    # Create experiment with inline dependency declaration
    with yanex.create_experiment(
        script_path=Path("train.py"),
        dependencies={"dataprep": "dataprep.py"},
        depends_on={"dataprep": dep_id}
    ) as exp:
        exp.start()
        exp_id = exp.experiment_id

    # Verify dependencies stored
    deps = manager.storage.load_dependencies(exp_id)
    assert deps["resolved_dependencies"]["dataprep"] == dep_id
    assert deps["declared_slots"]["dataprep"]["script"] == "dataprep.py"

def test_create_experiment_with_config_dependencies(temp_dir, git_repo):
    """Test create_experiment reading dependencies from config file."""
    # Create config with dependency declaration
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
yanex:
  dependencies:
    dataprep: dataprep.py
    """)

    # Create dependency
    dep_id = create_completed_experiment("dataprep.py")

    # Create experiment with config
    with yanex.create_experiment(
        script_path=Path("train.py"),
        config_path=config_path,
        depends_on={"dataprep": dep_id}
    ) as exp:
        exp_id = exp.experiment_id

    # Verify
    manager = ExperimentManager(temp_dir)
    deps = manager.storage.load_dependencies(exp_id)
    assert deps["resolved_dependencies"]["dataprep"] == dep_id

def test_create_experiment_override_config_dependencies(temp_dir, git_repo):
    """Test inline dependencies override config file."""
    # Config declares 2 dependencies
    config_path = temp_dir / "config.yaml"
    config_path.write_text("""
yanex:
  dependencies:
    dataprep: dataprep.py
    baseline: train.py
    """)

    # Create only train dependency
    tr_id = create_completed_experiment("train.py")

    # Override config - only need train
    with yanex.create_experiment(
        script_path=Path("evaluate.py"),
        config_path=config_path,
        dependencies={"train": "train.py"},  # Override
        depends_on={"train": tr_id}
    ) as exp:
        exp_id = exp.experiment_id

    # Verify only train dependency
    manager = ExperimentManager(temp_dir)
    deps = manager.storage.load_dependencies(exp_id)
    assert list(deps["resolved_dependencies"].keys()) == ["train"]

def test_create_experiment_without_declaration_raises(temp_dir, git_repo):
    """Test that depends_on without dependencies raises error."""
    with pytest.raises(ValueError, match="Cannot provide depends_on without declaring"):
        with yanex.create_experiment(
            script_path=Path("train.py"),
            depends_on={"dataprep": "dp1"}  # No declaration!
        ) as exp:
            pass

def test_experiment_spec_with_dependencies(temp_dir, git_repo):
    """Test ExperimentSpec with dependencies."""
    # Create dependency
    dp_id = create_completed_experiment("dataprep.py")

    # Create spec with dependencies
    spec = yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": 0.01},
        dependencies={"dataprep": "dataprep.py"},
        depends_on={"dataprep": dp_id}
    )

    # Run
    results = yanex.run_multiple([spec])
    assert len(results) == 1
    assert results[0].status == "completed"

    # Verify dependencies stored
    manager = ExperimentManager(temp_dir)
    deps = manager.storage.load_dependencies(results[0].experiment_id)
    assert deps["resolved_dependencies"]["dataprep"] == dp_id

def test_get_dependencies(temp_dir, git_repo):
    """Test yanex.get_dependencies() API."""
    # Create dependency with artifacts
    dep_id = create_completed_experiment("dataprep.py")
    manager = ExperimentManager(temp_dir)

    # Add artifacts to dependency
    artifacts_dir = manager.storage.get_experiment_directory(dep_id) / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "train_data.parquet").write_text("data")

    # Create experiment with dependency
    with yanex.create_experiment(
        script_path=Path("train.py"),
        dependencies={"dataprep": "dataprep.py"},
        depends_on={"dataprep": dep_id}
    ) as exp:
        exp.start()

        # Get dependencies
        deps = yanex.get_dependencies()

        # Verify
        assert "dataprep" in deps
        assert deps["dataprep"].experiment_id == dep_id
        assert deps["dataprep"].slot_name == "dataprep"

        # Check artifacts
        artifacts = deps["dataprep"].artifacts
        assert len(artifacts) == 1
        assert artifacts[0].name == "train_data.parquet"

        # Check artifact_path
        path = deps["dataprep"].artifact_path("train_data.parquet")
        assert path.exists()

        # Check load_artifact
        loaded_path = deps["dataprep"].load_artifact("train_data.parquet")
        assert loaded_path.exists()

        # Check missing artifact raises
        with pytest.raises(FileNotFoundError):
            deps["dataprep"].load_artifact("missing.parquet")
```

**Test Results API:**
```python
# tests/results/test_dependency_results_api.py
from yanex.results import Experiment, get_pipeline

def test_experiment_get_dependencies(temp_dir, git_repo):
    """Test Experiment.get_dependencies() returns direct dependencies."""
    # Create pipeline: dp1 -> tr1 -> eval1
    dp_id = create_completed_experiment("dataprep.py")
    tr_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    eval_id = create_completed_experiment_with_deps("evaluate.py", {"train": tr_id, "dataprep": dp_id})

    # Test direct dependencies
    exp = Experiment(eval_id)
    deps = exp.get_dependencies()

    assert len(deps) == 2
    assert any(d.id == dp_id for d in deps)
    assert any(d.id == tr_id for d in deps)

def test_experiment_get_dependencies_recursive(temp_dir, git_repo):
    """Test Experiment.get_dependencies(recursive=True) returns all ancestors."""
    # Create chain: prep0 -> dp1 -> tr1
    prep0_id = create_completed_experiment("prep.py")
    dp_id = create_completed_experiment_with_deps("dataprep.py", {"prep": prep0_id})
    tr_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})

    # Test recursive
    exp = Experiment(tr_id)
    all_deps = exp.get_dependencies(recursive=True)

    assert len(all_deps) == 2  # dp1 and prep0
    assert any(d.id == dp_id for d in all_deps)
    assert any(d.id == prep0_id for d in all_deps)

def test_experiment_get_dependents(temp_dir, git_repo):
    """Test Experiment.get_dependents() returns direct dependents."""
    # Create pipeline: dp1 -> [tr1, tr2, eval1]
    dp_id = create_completed_experiment("dataprep.py")
    tr1_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    tr2_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    eval_id = create_completed_experiment_with_deps("evaluate.py", {"dataprep": dp_id})

    # Test direct dependents
    exp = Experiment(dp_id)
    dependents = exp.get_dependents()

    assert len(dependents) == 3
    dep_ids = {d.id for d in dependents}
    assert tr1_id in dep_ids
    assert tr2_id in dep_ids
    assert eval_id in dep_ids

def test_experiment_get_dependents_recursive(temp_dir, git_repo):
    """Test Experiment.get_dependents(recursive=True) returns all descendants."""
    # Create chain: dp1 -> tr1 -> eval1
    dp_id = create_completed_experiment("dataprep.py")
    tr_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    eval_id = create_completed_experiment_with_deps("evaluate.py", {"train": tr_id})

    # Test recursive from root
    exp = Experiment(dp_id)
    all_dependents = exp.get_dependents(recursive=True)

    assert len(all_dependents) == 2  # tr1 and eval1
    assert any(d.id == tr_id for d in all_dependents)
    assert any(d.id == eval_id for d in all_dependents)

def test_experiment_get_dependency_info(temp_dir, git_repo):
    """Test Experiment.get_dependency_info() returns slot metadata."""
    dp_id = create_completed_experiment("dataprep.py")
    tr_id = create_completed_experiment("train.py")
    eval_id = create_completed_experiment_with_deps(
        "evaluate.py",
        {"dataprep": dp_id, "training": tr_id}
    )

    exp = Experiment(eval_id)
    info = exp.get_dependency_info()

    assert "dataprep" in info
    assert "training" in info

    # Check dataprep info
    assert info["dataprep"]["experiment_id"] == dp_id
    assert info["dataprep"]["experiment"].id == dp_id
    assert info["dataprep"]["slot"] == "dataprep"
    assert info["dataprep"]["script"] == "dataprep.py"
    assert info["dataprep"]["status"] == "completed"

    # Check training info
    assert info["training"]["experiment_id"] == tr_id
    assert info["training"]["slot"] == "training"

def test_experiment_get_pipeline(temp_dir, git_repo):
    """Test Experiment.get_pipeline() returns full connected subgraph."""
    # Create DAG:
    #     dp1
    #    /   \
    #  tr1   tr2
    #    \   /
    #    eval1

    dp_id = create_completed_experiment("dataprep.py")
    tr1_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    tr2_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})
    eval_id = create_completed_experiment_with_deps(
        "evaluate.py",
        {"train1": tr1_id, "train2": tr2_id, "dataprep": dp_id}
    )

    # Get pipeline from middle node
    exp = Experiment(tr1_id)
    graph = exp.get_pipeline()

    # Verify nodes
    assert len(graph["nodes"]) == 4
    assert dp_id in graph["nodes"]
    assert tr1_id in graph["nodes"]
    assert tr2_id in graph["nodes"]
    assert eval_id in graph["nodes"]

    # Verify edges
    assert len(graph["edges"]) == 5
    edge_pairs = {(e["source"], e["target"]) for e in graph["edges"]}
    assert (dp_id, tr1_id) in edge_pairs
    assert (dp_id, tr2_id) in edge_pairs
    assert (tr1_id, eval_id) in edge_pairs
    assert (tr2_id, eval_id) in edge_pairs
    assert (dp_id, eval_id) in edge_pairs

    # Verify roots and leaves
    assert graph["root_nodes"] == [dp_id]
    assert graph["leaf_nodes"] == [eval_id]

def test_get_pipeline_module_function(temp_dir, git_repo):
    """Test module-level yr.get_pipeline() function."""
    import yanex.results as yr

    dp_id = create_completed_experiment("dataprep.py")
    tr_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})

    # Call module function
    pipeline = yr.get_pipeline(tr_id)

    assert len(pipeline["nodes"]) == 2
    assert dp_id in pipeline["nodes"]
    assert tr_id in pipeline["nodes"]
    assert pipeline["root_nodes"] == [dp_id]
    assert pipeline["leaf_nodes"] == [tr_id]

def test_pipeline_edge_list_format(temp_dir, git_repo):
    """Test that pipeline format is correct (nodes, edges, roots, leaves)."""
    dp_id = create_completed_experiment("dataprep.py")
    tr_id = create_completed_experiment_with_deps("train.py", {"dataprep": dp_id})

    pipeline = get_pipeline(tr_id)

    # Verify structure
    assert "nodes" in pipeline
    assert "edges" in pipeline
    assert "root_nodes" in pipeline
    assert "leaf_nodes" in pipeline

    # Verify nodes are Experiment objects
    assert isinstance(pipeline["nodes"][dp_id], Experiment)
    assert isinstance(pipeline["nodes"][tr_id], Experiment)

    # Verify edges have correct structure
    assert len(pipeline["edges"]) == 1
    edge = pipeline["edges"][0]
    assert "source" in edge
    assert "target" in edge
    assert "slot" in edge
    assert edge["source"] == dp_id
    assert edge["target"] == tr_id
    assert edge["slot"] == "dataprep"

def test_pipeline_single_node(temp_dir, git_repo):
    """Test pipeline with single node (no dependencies)."""
    dp_id = create_completed_experiment("dataprep.py")

    exp = Experiment(dp_id)
    graph = exp.get_pipeline()

    assert len(graph["nodes"]) == 1
    assert len(graph["edges"]) == 0
    assert graph["root_nodes"] == [dp_id]
    assert graph["leaf_nodes"] == [dp_id]
```

### Integration Tests

```python
# tests/integration/test_dependency_workflow.py
def test_full_dependency_workflow(temp_dir, git_repo):
    """Test complete workflow: dataprep -> train -> evaluate."""

    # 1. Run dataprep
    result = runner.invoke(cli, [
        "run", "dataprep.py",
        "--name", "Data Preparation"
    ])
    assert result.exit_code == 0
    dp_id = extract_experiment_id(result.output)

    # 2. Run training with dependency
    result = runner.invoke(cli, [
        "run", "train.py",
        "--depends-on", f"dataprep={dp_id}",
        "--name", "Training"
    ])
    assert result.exit_code == 0
    tr_id = extract_experiment_id(result.output)

    # 3. Verify dependency stored
    manager = ExperimentManager(temp_dir)
    deps = manager.storage.load_dependencies(tr_id)
    assert deps["resolved_dependencies"]["dataprep"] == dp_id

    # 4. Verify reverse index
    dp_deps = manager.storage.load_dependencies(dp_id)
    assert any(d["experiment_id"] == tr_id for d in dp_deps["depended_by"])

    # 5. Run evaluation with multiple dependencies
    result = runner.invoke(cli, [
        "run", "evaluate.py",
        "--depends-on", f"dataprep={dp_id}",
        "--depends-on", f"training={tr_id}",
        "--name", "Evaluation"
    ])
    assert result.exit_code == 0

def test_dependency_sweep(temp_dir, git_repo):
    """Test creating multiple experiments via dependency sweep."""

    # Create dataprep
    dp_id = create_experiment("dataprep.py")

    # Create multiple training runs
    tr_ids = [
        create_experiment("train.py", depends_on={"dataprep": dp_id}),
        create_experiment("train.py", depends_on={"dataprep": dp_id}),
        create_experiment("train.py", depends_on={"dataprep": dp_id})
    ]

    # Sweep over training dependencies
    result = runner.invoke(cli, [
        "run", "evaluate.py",
        "--depends-on", f"dataprep={dp_id}",
        "--depends-on", f"training={','.join(tr_ids)}"
    ])
    assert result.exit_code == 0

    # Should create 3 evaluation experiments
    experiments = list_experiments()
    eval_exps = [e for e in experiments if e.script == "evaluate.py"]
    assert len(eval_exps) == 3
```

### CLI Tests

```python
# tests/cli/test_dependency_commands.py
def test_yanex_id_command(temp_dir):
    """Test yanex id command."""
    # Create experiments
    dp1 = create_experiment("dataprep.py")
    tr1 = create_experiment("train.py")

    # Test basic filter
    result = runner.invoke(cli, ["id", "--script", "train.py"])
    assert tr1 in result.output
    assert dp1 not in result.output

def test_delete_with_cascade(temp_dir):
    """Test deleting experiment with dependents."""
    dp1 = create_experiment("dataprep.py")
    tr1 = create_experiment("train.py", depends_on={"dataprep": dp1})

    # Try delete without cascade - should fail
    result = runner.invoke(cli, ["delete", dp1])
    assert result.exit_code != 0
    assert "depended on by" in result.output

    # Delete with cascade
    result = runner.invoke(cli, ["delete", dp1, "--cascade"], input="y\n")
    assert result.exit_code == 0

    # Both should be deleted
    assert not experiment_exists(dp1)
    assert not experiment_exists(tr1)
```

## Implementation Phases

### Phase 1: Foundation (MVP)

**Goal:** Basic dependency tracking and validation

**Tasks:**
1. Create `storage_dependencies.py` module
2. Create `dependency_validator.py` module
3. Update `metadata.json` schema with `dependencies_summary`
4. Implement `--depends-on` flag parsing in `yanex run`
5. Implement dependency validation before experiment creation
6. Store `dependencies.json` with forward and reverse links
7. Update `yanex show` to display dependencies
8. Update `yanex list` to show dependency indicator
9. Prevent `yanex delete` if depended on (without --force)
10. Add basic filters: `--depends-on`, `--root`, `--leaf`
11. **Python API: Add `dependencies` and `depends_on` parameters to `create_experiment()`**
12. **Python API: Add `dependencies` and `depends_on` fields to `ExperimentSpec`**
13. **Python API: Implement dependency resolution hierarchy (inline → config → none)**
14. Python API: `yanex.get_dependencies()` and `DependencyReference` class
15. **Results API: Add `Experiment.get_dependencies(recursive, max_depth)` method**
16. **Results API: Add `Experiment.get_dependents(recursive, max_depth)` method**
17. **Results API: Add `Experiment.get_dependency_info()` method**
18. **Results API: Add `Experiment.get_dependency_graph()` and `get_pipeline()` methods**
19. **Results API: Add module-level `yanex.results.get_pipeline(experiment_id)` function**
20. Write comprehensive tests (CLI, API, Results API, integration)

**Deliverables:**
- Users can declare dependencies in config
- Users can provide dependencies via CLI
- **Users can create experiments with dependencies via Python API**
- **Both inline and config-based dependency declaration work**
- Validation prevents invalid dependencies
- Dependencies are tracked bidirectionally
- Basic queries work (show, list with filters)
- Scripts can access dependency artifacts via `get_dependencies()`
- **Results API enables post-hoc dependency graph analysis**
- **Users can query upstream/downstream dependencies programmatically**
- **Full pipeline extraction via `yr.get_pipeline(experiment_id)`**

**Testing checklist:**
- [ ] Parse single/multiple/sweep dependencies
- [ ] Validate experiment exists
- [ ] Validate script matches
- [ ] Validate status is completed
- [ ] Store dependencies.json correctly
- [ ] Update reverse index correctly
- [ ] **Python API: create_experiment with inline dependencies**
- [ ] **Python API: create_experiment with config dependencies**
- [ ] **Python API: create_experiment override config with inline**
- [ ] **Python API: create_experiment without declaration raises error**
- [ ] **Python API: ExperimentSpec with dependencies**
- [ ] **Python API: get_dependencies() returns correct references**
- [ ] **Python API: DependencyReference.artifacts works**
- [ ] **Results API: get_dependencies() returns direct dependencies**
- [ ] **Results API: get_dependencies(recursive=True) returns all ancestors**
- [ ] **Results API: get_dependents() returns direct dependents**
- [ ] **Results API: get_dependents(recursive=True) returns all descendants**
- [ ] **Results API: get_dependency_info() returns slot metadata**
- [ ] **Results API: get_pipeline() returns full connected subgraph**
- [ ] **Results API: yr.get_pipeline() module function works**
- [ ] **Results API: Edge list format is correct (nodes, edges, roots, leaves)**
- [ ] Show command displays dependencies
- [ ] List command shows dependency indicator
- [ ] Delete command prevents deletion if depended on

### Phase 2: Enhanced CLI

**Goal:** Full CLI feature set

**Tasks:**
1. Implement `yanex id` command with all filters
2. Implement dependency sweeps (cartesian product)
3. Add `--cascade` and `--force` to `yanex delete`
4. Add `--cascade-up` to delete upstream too
5. Implement `yanex archive --with-dependencies/--with-dependents`
6. Add advanced filters: `--depends-on-script`, `--orphaned`, `--has-slot`
7. Implement `yanex list --tree` mode
8. Implement `yanex list --by-depth` mode
9. Implement `yanex validate` command
10. Implement `yanex deps` command (show graph)
11. Implement `yanex dependents` command
12. Improve error messages with suggestions

**Deliverables:**
- Full CLI feature parity
- Composable queries with `yanex id`
- Safe cascade operations
- Multiple visualization modes
- Helpful error messages

**Testing checklist:**
- [ ] yanex id with filters works
- [ ] Dependency sweeps create correct experiments
- [ ] Cascade delete works correctly
- [ ] Archive with dependencies works
- [ ] Tree/depth views render correctly
- [ ] Validate command finds broken dependencies
- [ ] Error messages are helpful

### Phase 3: UI Integration

**Goal:** Visual DAG in web interface

**Tasks:**
1. Add `/api/experiments/dag` endpoint
2. Add React-Flow component
3. Implement node coloring by script type
4. Implement interactive graph (zoom, pan, select)
5. Add dependency filtering in UI
6. Add "show subgraph" for selected experiment
7. Add dependency creation UI (future)
8. Style nodes based on status (running, completed, failed)

**Deliverables:**
- Beautiful DAG visualization in web UI
- Interactive exploration of experiment workflows
- Filter experiments by dependencies in UI

**Testing checklist:**
- [ ] API returns correct nodes and edges
- [ ] Graph renders correctly
- [ ] Filtering works
- [ ] Subgraph view works
- [ ] Node colors/styles correct

### Phase 4: Advanced Features (Future)

**Goal:** Advanced dependency features

**Tasks:**
1. Optional dependencies (`required: false`)
2. Artifact-level validation (specific files required)
3. Circular dependency detection
4. Auto-execution of dependency chains
5. Dependency versioning/invalidation
6. Soft links to dependency artifacts
7. Dependency templates/macros
8. Import DAG from config file

**Deliverables:**
- Production-ready dependency system
- Advanced workflow orchestration

## Migration Strategy

**This feature is fully backward compatible:**

1. **No changes to existing experiments** - experiments without dependencies work identically
2. **New files only created when needed** - `dependencies.json` only for experiments with dependencies
3. **metadata.json enhancement is optional** - `dependencies_summary` absent on old experiments is fine
4. **CLI changes are additive** - all existing commands work as before

**No migration script needed.**

## Documentation Updates

### User-facing docs to create/update:

1. **docs/dependencies.md** (new)
   - Concept explanation
   - Declaration in config
   - CLI usage examples
   - Python API examples
   - Common patterns (dataprep → train → evaluate)

2. **docs/configuration.md** (update)
   - Add `dependencies` section to `yanex:` docs

3. **docs/cli-reference.md** (update)
   - Document `--depends-on` flag
   - Document `yanex id` command
   - Document `yanex validate`, `yanex deps`, `yanex dependents`
   - Document new filters

4. **README.md** (update)
   - Add dependency workflow example

5. **examples/** (new files)
   - `examples/dependencies/dataprep.py`
   - `examples/dependencies/train.py`
   - `examples/dependencies/evaluate.py`
   - `examples/dependencies/config.yaml`

## Open Questions

1. **Should failed experiments be allowed as dependencies?**
   - Use case: Evaluate model even if training failed
   - Recommendation: Default reject, add `--allow-failed` flag

2. **Should running experiments be allowed as dependencies?**
   - Use case: Start evaluation while training is ongoing (read partial checkpoints)
   - Recommendation: Default reject, could add `--allow-running` flag later

3. **Circular dependency detection:**
   - When to check? At declaration time or execution time?
   - Recommendation: Phase 4 feature - check at creation time

4. **Artifact validation:**
   - Should we validate artifacts exist even in MVP?
   - Recommendation: No, add in Phase 2/3

5. **Dependency on archived experiments:**
   - Should dependencies search archived experiments too?
   - Recommendation: Yes, with `include_archived=True` in lookups

## Success Metrics

**Phase 1 complete when:**
- [ ] Users can create experiments with dependencies via CLI
- [ ] Dependencies are validated before execution
- [ ] Dependencies are tracked bidirectionally
- [ ] Basic queries work (show, list, delete protection)
- [ ] 90%+ test coverage on new code
- [ ] Documentation complete

**Phase 2 complete when:**
- [ ] All CLI commands support dependencies
- [ ] Dependency sweeps work
- [ ] Cascade operations safe and tested
- [ ] Error messages helpful and actionable

**Phase 3 complete when:**
- [ ] DAG visualized in web UI
- [ ] Users can explore dependency graphs interactively
- [ ] UI filtering by dependencies works

## References

- **Similar systems:** MLflow (experiments), Airflow (DAGs), Prefect (workflows), DVC (pipeline stages)
- **React-Flow docs:** https://reactflow.dev/
- **DAG algorithms:** Topological sort, cycle detection, graph traversal
