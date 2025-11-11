# Experiment Dependencies - Implementation Addendum

**Status:** Ready for Implementation
**Date:** 2025-11-11
**Purpose:** Fill technical gaps and provide detailed implementation guidance for Phase 1

---

## Design Decisions (FINALIZED)

Based on review, the following decisions have been made for Phase 1:

1. **Circular dependency detection:** ✅ Phase 1 - Check at creation time and block
2. **File locking:** ❌ Phase 2 - Defer to avoid cross-platform complexity
3. **Parameter sweep syntax migration:** ❌ Phase 2 - Keep `list()` wrapper for now
4. **Extra slots behavior:** ✅ Strict - Reject with error (prevents misconfigurations)
5. **Artifact validation:** ❌ Phase 2 - Defer to keep MVP focused

---

## 1. Config File Schema Reference

**INSERT AFTER:** Design Decision #2 (line 117 in main document)

### Complete YAML Schema for Dependencies

```yaml
# shared_config.yaml - Complete example

# Regular parameters (shared by all scripts)
learning_rate: 0.001
batch_size: 32
epochs: 100

# Yanex-specific configuration
yanex:
  scripts:
    # Root node - no dependencies
    - name: "dataprep.py"
      # No dependencies section = root node

    # Single dependency
    - name: "train.py"
      dependencies:
        dataprep: dataprep.py

    # Multiple dependencies
    - name: "evaluate.py"
      dependencies:
        dataprep: dataprep.py
        training: train.py

    # Multiple slots, same script type (A/B testing, ensemble)
    - name: "compare.py"
      dependencies:
        baseline: train.py
        variant: train.py
```

### Schema Specification

**`yanex.scripts[]` array:**
- Type: List of script definition objects
- Optional (if absent, no dependencies are declared)

**Script definition object:**
```yaml
name: string              # REQUIRED: Exact script filename (e.g., "train.py")
dependencies: object      # OPTIONAL: Dependency declarations
```

**Dependencies object:**
```yaml
slot_name:                # REQUIRED: Arbitrary identifier for this dependency
  <script>                # Value can be string (shorthand) or object (full)
```

**Shorthand syntax (Phase 1):**
```yaml
dependencies:
  dataprep: dataprep.py   # String = script path
  training: train.py
```

**Full syntax (Phase 4 - forward compatibility):**
```yaml
dependencies:
  dataprep:
    script: dataprep.py   # Required script
    required: true        # Optional (default: true) - Phase 4
    artifacts:            # Optional - Phase 3
      - "*.parquet"
      - "model.pkl"
```

**Phase 1 implementation:** Parser should accept both formats but only uses `script` field.

### Validation Rules

1. **Slot names:**
   - Must be non-empty strings
   - Must be unique within a script's dependencies
   - Can be different from script names (enables multiple slots for same script)
   - Arbitrary identifiers (e.g., `baseline_model`, `data_v2`)

2. **Script paths:**
   - Must be exact filenames (e.g., `train.py`, not `./train.py`)
   - Should match experiment's `script_path` basename
   - No directory separators allowed

3. **Config lookup:**
   - When running `yanex run train.py --config shared.yaml`
   - Match on `name` field exactly (case-sensitive)
   - If multiple matches, use first match
   - If no match, dependencies = empty (ad-hoc allowed)

---

## 2. Validation and Creation Sequence

**INSERT AFTER:** Line 981 (Implementation Details section)

### Complete Execution Sequence

When user runs: `yanex run evaluate.py -d dataprep=dp1 -d training=tr1 --config shared.yaml`

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Pre-Validation (Before experiment creation)       │
└─────────────────────────────────────────────────────────────┘

1.1 Parse CLI arguments
    ├─ script_path = Path("evaluate.py")
    ├─ depends_on = ("dataprep=dp1", "training=tr1")
    └─ config_path = Path("shared.yaml")

1.2 Load configuration file
    ├─ config_data = load_yaml_config(config_path)
    ├─ yanex_section = config_data.get("yanex", {})
    └─ Find "evaluate.py" in scripts array

1.3 Extract declared slots
    ├─ declared_slots = {
    │     "dataprep": {"script": "dataprep.py", "required": true},
    │     "training": {"script": "train.py", "required": true}
    │  }
    └─ Remove 'yanex' section from config_data

1.4 Parse dependency arguments
    ├─ parsed_deps = parse_dependency_args(depends_on)
    ├─ Result: {"dataprep": ["dp1"], "training": ["tr1"]}
    └─ Validate experiment ID format (8-char hex)

1.5 Validate all required slots provided
    ├─ Check: All required slots in declared_slots are in parsed_deps
    ├─ Check: No extra slots (STRICT MODE - Error if extras found)
    ├─ ✓ Pass: All slots match
    └─ ✗ Fail: Raise DependencyError with helpful message

┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Dependency Validation (Before experiment creation)│
└─────────────────────────────────────────────────────────────┘

2.1 For each dependency (dp1, tr1):

    2.1.1 Check experiment exists
        ├─ storage.experiment_exists(dep_id, include_archived=True)
        ├─ ✓ Pass: Continue
        └─ ✗ Fail: Raise DependencyError("Experiment 'dp1' not found")

    2.1.2 Load dependency metadata
        ├─ dep_metadata = storage.load_metadata(dep_id, include_archived=True)
        └─ Extract: script_path, status, created_at

    2.1.3 Validate script match
        ├─ actual_script = Path(dep_metadata["script_path"]).name
        ├─ expected_script = declared_slots[slot]["script"]
        ├─ ✓ Pass: actual == expected
        └─ ✗ Fail: Raise DependencyError("Script mismatch: expected X, got Y")

    2.1.4 Validate status is "completed"
        ├─ dep_status = dep_metadata["status"]
        ├─ ✓ Pass: status == "completed"
        └─ ✗ Fail: Raise DependencyError("Dependency not completed")

    2.1.5 Record validation check
        └─ checks.append({
              "slot": slot_name,
              "experiment_id": dep_id,
              "script_expected": expected_script,
              "script_actual": actual_script,
              "script_match": True,
              "experiment_status": "completed",
              "valid": True
            })

2.2 Check for circular dependencies (NEW - Phase 1)
    ├─ Run DFS from each dependency
    ├─ ✓ Pass: No cycles detected
    └─ ✗ Fail: Raise CircularDependencyError("Cycle: dp1 -> tr1 -> eval1 -> dp1")

2.3 Build validation result
    └─ validation = {
          "validated_at": datetime.utcnow().isoformat(),
          "status": "valid",
          "checks": checks
        }

┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: Git Checks                                        │
└─────────────────────────────────────────────────────────────┘

3.1 Check git repository state
    ├─ Get current commit, branch, dirty status
    └─ Generate patch if working directory is dirty

┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: Experiment Creation (Atomic operations)           │
└─────────────────────────────────────────────────────────────┘

4.1 Generate experiment ID
    └─ experiment_id = secrets.token_hex(4)  # "eval1234"

4.2 Create directory structure
    └─ exp_dir = storage.create_experiment_directory(experiment_id)

4.3 Save metadata.json
    ├─ metadata = {
    │     "id": experiment_id,
    │     "script_path": str(script_path),
    │     "status": "created",
    │     "created_at": timestamp,
    │     ... (git info, name, tags, etc.)
    │  }
    └─ storage.save_metadata(experiment_id, metadata)

4.4 Save config.yaml
    └─ storage.save_config(experiment_id, config_data)

┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: Dependency Storage (Multiple file writes)         │
└─────────────────────────────────────────────────────────────┘

5.1 Save dependencies.json for current experiment
    ├─ dependencies_data = {
    │     "version": "1.0",
    │     "declared_slots": declared_slots,
    │     "resolved_dependencies": {"dataprep": "dp1", "training": "tr1"},
    │     "validation": validation,
    │     "depended_by": []
    │  }
    └─ storage.save_dependencies(experiment_id, dependencies_data)

5.2 Update reverse indexes (for each dependency)
    ├─ For dp1:
    │   ├─ Load dp1/dependencies.json (or create empty structure)
    │   ├─ Append to depended_by: {"experiment_id": "eval1234", "slot_name": "dataprep", "created_at": "..."}
    │   └─ Save dp1/dependencies.json
    │
    └─ For tr1:
        ├─ Load tr1/dependencies.json
        ├─ Append to depended_by: {"experiment_id": "eval1234", "slot_name": "training", "created_at": "..."}
        └─ Save tr1/dependencies.json

    ⚠️ NOTE: No file locking in Phase 1
        - Risk: If two experiments depend on dp1 simultaneously, one update might be lost
        - Mitigation: Rare case, low impact (missing one reverse index entry)
        - Resolution: Add file locking in Phase 2

5.3 Update metadata.json with summary
    ├─ dependencies_summary = {
    │     "has_dependencies": True,
    │     "dependency_count": 2,
    │     "dependency_slots": ["dataprep", "training"],
    │     "is_depended_by": False,
    │     "depended_by_count": 0
    │  }
    └─ storage.update_experiment_metadata(experiment_id, {"dependencies_summary": summary})

┌─────────────────────────────────────────────────────────────┐
│ PHASE 6: Script Execution                                  │
└─────────────────────────────────────────────────────────────┘

6.1 Set environment variables
    ├─ YANEX_EXPERIMENT_ID = experiment_id
    ├─ YANEX_CLI_ACTIVE = "1"
    └─ YANEX_PARAM_* = parameter values

6.2 Update status to "running"
    └─ storage.update_experiment_metadata(experiment_id, {"status": "running", "started_at": "..."})

6.3 Execute script subprocess
    └─ Run: python evaluate.py [script_args]

6.4 Capture result
    ├─ ✓ Success: Update status to "completed"
    └─ ✗ Failure: Update status to "failed", save error message

┌─────────────────────────────────────────────────────────────┐
│ ERROR HANDLING & ROLLBACK                                  │
└─────────────────────────────────────────────────────────────┘

Failure in Phase 1-2 (Validation):
    └─ No cleanup needed, experiment not created yet

Failure in Phase 3 (Git):
    └─ No cleanup needed, experiment not created yet

Failure in Phase 4 (Creation):
    ├─ Delete experiment directory (exp_dir)
    └─ Raise error to user

Failure in Phase 5 (Dependency storage):
    ├─ Delete experiment directory (exp_dir)
    ├─ Rollback reverse index updates (best effort):
    │   └─ For each updated dependency, remove this experiment from depended_by
    └─ Raise error to user

Failure in Phase 6 (Script execution):
    ├─ DO NOT delete experiment
    ├─ Mark status as "failed"
    ├─ Save error message to metadata
    └─ Return error to user
```

### Atomicity Guarantees

**What IS atomic:**
- Single file writes (`metadata.json`, `config.yaml`, `dependencies.json`)
- Experiment directory creation

**What is NOT atomic (without file locking):**
- Reverse index updates across multiple experiments
- Multiple file writes in Phase 5

**Consequences of non-atomicity:**
- If process crashes during Phase 5, some reverse indexes might be incomplete
- Experiment will exist but some dependency backlinks might be missing
- Recovery: Future `yanex validate` command can rebuild reverse indexes (Phase 2)

---

## 3. Enhanced Argument Parsing

**INSERT/REPLACE AT:** Line 1078 (parse_dependency_args function)

### Complete Implementation with Validation

```python
def parse_dependency_args(depends_on: tuple[str]) -> dict[str, list[str]]:
    """
    Parse --depends-on arguments with comprehensive validation.

    Args:
        depends_on: Tuple of "slot=experiment_id" strings from Click

    Returns:
        Dictionary mapping slot names to lists of experiment IDs
        Example: {"dataprep": ["dp1"], "training": ["tr1", "tr2"]}

    Raises:
        click.UsageError: If format is invalid or IDs are malformed

    Examples:
        >>> parse_dependency_args(("dataprep=abc12345",))
        {"dataprep": ["abc12345"]}

        >>> parse_dependency_args(("training=tr1,tr2,tr3",))
        {"training": ["tr1", "tr2", "tr3"]}
    """
    import re

    result = {}

    for dep_arg in depends_on:
        # Validate format: slot=value
        if "=" not in dep_arg:
            raise click.UsageError(
                f"Invalid dependency format: '{dep_arg}'\n"
                f"Expected format: slot=experiment_id\n"
                f"Example: --depends-on dataprep=abc12345"
            )

        # Split on first '=' only (allows '=' in values, though unlikely)
        parts = dep_arg.split("=", 1)
        if len(parts) != 2:
            raise click.UsageError(
                f"Invalid dependency format: '{dep_arg}'"
            )

        slot, ids_str = parts

        # Validate slot name
        slot = slot.strip()
        if not slot:
            raise click.UsageError(
                "Slot name cannot be empty\n"
                "Example: --depends-on dataprep=abc12345"
            )

        # Validate slot name characters (alphanumeric + underscore)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', slot):
            raise click.UsageError(
                f"Invalid slot name '{slot}'\n"
                f"Slot names must start with a letter or underscore, "
                f"and contain only letters, numbers, and underscores.\n"
                f"Valid examples: dataprep, data_v2, baseline_model"
            )

        # Validate experiment IDs
        ids_str = ids_str.strip()
        if not ids_str:
            raise click.UsageError(
                f"Experiment ID cannot be empty for slot '{slot}'\n"
                f"Example: --depends-on {slot}=abc12345"
            )

        # Parse comma-separated IDs
        ids = [id_val.strip() for id_val in ids_str.split(",")]

        # Validate each ID format
        for exp_id in ids:
            if not exp_id:
                raise click.UsageError(
                    f"Empty experiment ID in slot '{slot}'\n"
                    f"Check for extra commas in: {ids_str}"
                )

            # Experiment IDs are 8-character hexadecimal strings
            if len(exp_id) != 8:
                raise click.UsageError(
                    f"Invalid experiment ID '{exp_id}' for slot '{slot}'\n"
                    f"Experiment IDs must be exactly 8 characters.\n"
                    f"Use 'yanex list' to find valid experiment IDs."
                )

            # Validate hexadecimal format
            try:
                int(exp_id, 16)
            except ValueError:
                raise click.UsageError(
                    f"Invalid experiment ID '{exp_id}' for slot '{slot}'\n"
                    f"Experiment IDs must be hexadecimal (0-9, a-f).\n"
                    f"Did you mean to use the full ID from 'yanex list'?"
                )

        # Merge with existing entries (support multiple --depends-on for same slot)
        if slot in result:
            result[slot].extend(ids)
        else:
            result[slot] = ids

    return result


def validate_dependency_slots(
    declared_slots: dict[str, dict],
    provided_slots: dict[str, list[str]]
) -> None:
    """
    Validate that provided slots match declared slots (STRICT MODE).

    Args:
        declared_slots: Slots declared in config file
        provided_slots: Slots provided via --depends-on flags

    Raises:
        click.UsageError: If validation fails
    """
    from difflib import get_close_matches

    # Check for missing required slots
    missing_slots = []
    for slot, config in declared_slots.items():
        if config.get("required", True) and slot not in provided_slots:
            missing_slots.append(slot)

    if missing_slots:
        msg = "Missing required dependencies:\n"
        for slot in missing_slots:
            script = declared_slots[slot]["script"]
            msg += f"  ✗ {slot} (requires script: {script})\n"

        msg += f"\nTo fix, add:\n"
        for slot in missing_slots:
            msg += f"  --depends-on {slot}=<experiment_id>\n"

        msg += f"\nFind experiments: yanex id --script <script_name> --status completed"

        raise click.UsageError(msg)

    # Check for extra slots (STRICT MODE - reject extras)
    extra_slots = set(provided_slots.keys()) - set(declared_slots.keys())

    if extra_slots:
        msg = f"Unknown dependency slot(s): {', '.join(sorted(extra_slots))}\n\n"
        msg += f"Expected slots (from config):\n"
        for slot, config in declared_slots.items():
            script = config["script"]
            msg += f"  ✓ {slot} (script: {script})\n"

        # Provide suggestions for typos
        for extra in extra_slots:
            suggestions = get_close_matches(extra, declared_slots.keys(), n=3, cutoff=0.6)
            if suggestions:
                msg += f"\nDid you mean '{suggestions[0]}' instead of '{extra}'?"

        raise click.UsageError(msg)
```

### Usage in `yanex run` Command

```python
# In yanex/cli/commands/run.py

# Parse dependency arguments (validates format)
parsed_deps = parse_dependency_args(depends_on)

# Validate slots match declaration (STRICT mode)
if declared_slots:
    validate_dependency_slots(declared_slots, parsed_deps)
elif parsed_deps:
    # No declaration but dependencies provided
    raise click.UsageError(
        "Cannot provide --depends-on without declaring dependencies.\n"
        "Either add 'dependencies' to your script entry in the config file,\n"
        "or remove the --depends-on flags."
    )
```

---

## 4. Config Lookup Implementation

**INSERT AT:** Line 1002 (in CLI Changes section)

### Detailed Configuration Resolution

```python
def resolve_dependency_declaration(
    script_path: Path,
    config_path: Path | None
) -> tuple[dict[str, dict], dict[str, Any]]:
    """
    Resolve dependency declaration from config file.

    Args:
        script_path: Script being executed
        config_path: Optional config file path

    Returns:
        Tuple of (declared_slots, config_data)
        - declared_slots: Normalized dependency declaration
        - config_data: Config parameters (without 'yanex' section)

    Example:
        >>> declared, config = resolve_dependency_declaration(
        ...     Path("train.py"),
        ...     Path("shared.yaml")
        ... )
        >>> declared
        {"dataprep": {"script": "dataprep.py", "required": True}}
        >>> config
        {"learning_rate": 0.01, "batch_size": 32}
    """
    declared_slots = {}
    config_data = {}

    if not config_path:
        return declared_slots, config_data

    # Load full config
    full_config = load_yaml_config(config_path)

    # Extract yanex section
    yanex_section = full_config.get("yanex", {})

    # Look up this script in scripts array
    scripts = yanex_section.get("scripts", [])
    script_name = script_path.name

    for script_entry in scripts:
        entry_name = script_entry.get("name")

        if entry_name == script_name:
            # Found matching script entry
            raw_deps = script_entry.get("dependencies", {})

            # Normalize dependencies to full schema
            for slot, dep_spec in raw_deps.items():
                if isinstance(dep_spec, str):
                    # Shorthand: "dataprep: dataprep.py"
                    declared_slots[slot] = {
                        "script": dep_spec,
                        "required": True
                    }
                elif isinstance(dep_spec, dict):
                    # Full syntax: "dataprep: {script: dataprep.py, required: true}"
                    declared_slots[slot] = {
                        "script": dep_spec.get("script"),
                        "required": dep_spec.get("required", True)
                    }
                else:
                    raise ConfigError(
                        f"Invalid dependency specification for slot '{slot}'\n"
                        f"Expected string or object, got {type(dep_spec)}"
                    )

            # Found match, stop searching
            break

    # Extract regular parameters (everything except 'yanex' section)
    config_data = {
        key: value
        for key, value in full_config.items()
        if key != "yanex"
    }

    return declared_slots, config_data


# Usage in yanex run command:

def run(..., script, config, depends_on, ...):
    """Run command implementation."""

    # Resolve dependency declaration from config
    declared_slots, config_data = resolve_dependency_declaration(
        script_path=script,
        config_path=config
    )

    # Parse and validate CLI dependencies
    if depends_on:
        parsed_deps = parse_dependency_args(depends_on)

        # STRICT validation
        if declared_slots:
            validate_dependency_slots(declared_slots, parsed_deps)
        else:
            # No declaration in config
            raise click.UsageError(
                f"Cannot provide --depends-on without declaring dependencies.\n\n"
                f"Script '{script.name}' has no dependency declaration in config.\n"
                f"Add to config file:\n\n"
                f"yanex:\n"
                f"  scripts:\n"
                f"    - name: \"{script.name}\"\n"
                f"      dependencies:\n"
                f"        slot_name: script.py"
            )
    else:
        parsed_deps = {}

    # Continue with rest of run command...
```

---

## 5. Circular Dependency Detection

**INSERT AT:** Line 891 (After DependencyValidator class)

### Implementation with DFS

```python
class CircularDependencyDetector:
    """Detect circular dependencies using depth-first search."""

    def __init__(self, storage):
        self.storage = storage

    def check_for_cycles(
        self,
        new_experiment_id: str,
        resolved_deps: dict[str, str]
    ) -> None:
        """
        Check if adding dependencies would create a cycle.

        Args:
            new_experiment_id: ID of experiment being created
            resolved_deps: Dependencies to add (slot -> experiment_id)

        Raises:
            CircularDependencyError: If a cycle is detected

        Algorithm:
            For each dependency D in resolved_deps:
                Run DFS from D
                If we reach new_experiment_id, cycle detected

        Example cycle:
            Creating exp3 with dependency on exp1
            But exp1 depends on exp2
            And exp2 depends on exp3 (not created yet)

            DFS from exp1:
                exp1 -> exp2 -> (would depend on exp3) -> exp1
                Cycle detected!
        """
        for slot_name, dep_id in resolved_deps.items():
            # Start DFS from each dependency
            visited = set()
            path = []

            if self._has_path_to(dep_id, new_experiment_id, visited, path):
                # Cycle detected: dep_id has a path back to new_experiment_id
                cycle_path = " → ".join(path + [new_experiment_id])

                raise CircularDependencyError(
                    f"Circular dependency detected:\n"
                    f"  {cycle_path}\n\n"
                    f"Adding dependency '{slot_name}' = {dep_id} would create a cycle.\n"
                    f"Experiment {dep_id} already depends on {new_experiment_id} "
                    f"(directly or indirectly).\n\n"
                    f"To fix:\n"
                    f"  1. Review your dependency chain: yanex show {dep_id}\n"
                    f"  2. Remove the circular dependency\n"
                    f"  3. Ensure your workflow is a DAG (directed acyclic graph)"
                )

    def _has_path_to(
        self,
        current_id: str,
        target_id: str,
        visited: set[str],
        path: list[str]
    ) -> bool:
        """
        DFS to check if there's a path from current_id to target_id.

        Args:
            current_id: Current node in DFS
            target_id: Target we're searching for
            visited: Set of visited nodes (prevents infinite loops)
            path: Current path (for error messages)

        Returns:
            True if path exists, False otherwise
        """
        # Base case: reached target
        if current_id == target_id:
            return True

        # Already visited this node
        if current_id in visited:
            return False

        visited.add(current_id)
        path.append(current_id)

        # Load dependencies of current node
        try:
            deps_data = self.storage.load_dependencies(
                current_id,
                include_archived=True
            )
        except Exception:
            # Experiment doesn't exist or has no dependencies
            path.pop()
            return False

        if not deps_data or not deps_data.get("resolved_dependencies"):
            # No dependencies, no path to target
            path.pop()
            return False

        # Recursively check each dependency
        for dep_id in deps_data["resolved_dependencies"].values():
            if self._has_path_to(dep_id, target_id, visited, path):
                return True  # Found path

        # No path found from this branch
        path.pop()
        return False


# Add to DependencyValidator class:

def validate_dependencies(
    self,
    experiment_id: str,
    declared_slots: dict[str, dict[str, Any]],
    resolved_deps: dict[str, str]
) -> dict[str, Any]:
    """
    Validate all dependencies for an experiment.

    UPDATED: Now includes circular dependency detection.
    """
    from datetime import datetime

    checks = []
    all_valid = True

    # Existing validation (experiment exists, script match, status)
    for slot_name, slot_config in declared_slots.items():
        # ... existing validation code ...
        pass

    # NEW: Check for circular dependencies
    if resolved_deps:
        detector = CircularDependencyDetector(self.storage)

        # This raises CircularDependencyError if cycle detected
        detector.check_for_cycles(experiment_id, resolved_deps)

    return {
        "validated_at": datetime.utcnow().isoformat(),
        "status": "valid" if all_valid else "invalid",
        "checks": checks
    }
```

### Exception Class

```python
# Add to yanex/utils/exceptions.py

class CircularDependencyError(DependencyError):
    """Raised when a circular dependency is detected."""

    pass  # Error message provided in raise statement
```

### Testing Strategy

```python
# tests/core/test_circular_dependencies.py

def test_direct_circular_dependency():
    """Test detection of A → B → A cycle."""
    # Create exp1
    exp1_id = create_experiment("prep.py")

    # Create exp2 depending on exp1
    exp2_id = create_experiment_with_deps("train.py", {"prep": exp1_id})

    # Try to create exp1b (replacing exp1) depending on exp2
    # This would create: exp1b → exp2 → exp1 (orphaned)
    # Actually, this is not a cycle since exp1 and exp1b are different

    # Better test: Try to add exp1 as dependency of itself
    with pytest.raises(CircularDependencyError, match="Circular dependency"):
        validator = DependencyValidator(storage, manager)
        validator.validate_dependencies(
            experiment_id=exp1_id,
            declared_slots={"self": {"script": "prep.py", "required": True}},
            resolved_deps={"self": exp1_id}
        )


def test_indirect_circular_dependency():
    """Test detection of A → B → C → A cycle."""
    # Create chain: exp1 → exp2 → exp3
    exp1_id = create_experiment("step1.py")
    exp2_id = create_experiment_with_deps("step2.py", {"step1": exp1_id})
    exp3_id = create_experiment_with_deps("step3.py", {"step2": exp2_id})

    # Try to create exp4 that depends on exp3
    # Then try to make exp1 depend on exp4
    # This would close the loop: exp4 → exp3 → exp2 → exp1 → exp4

    exp4_id = create_experiment_with_deps("step4.py", {"step3": exp3_id})

    # Now try to add exp4 as dependency of exp1 (not possible in practice
    # since exp1 already exists, but simulate the validation)

    with pytest.raises(CircularDependencyError):
        detector = CircularDependencyDetector(storage)
        detector.check_for_cycles(exp1_id, {"step4": exp4_id})
```

---

## 6. Dependency Sweep Implementation

**INSERT AT:** Line 1016 (in CLI Changes section)

### Complete Cartesian Product Logic

```python
def create_dependency_sweep(
    script_path: Path,
    parsed_deps: dict[str, list[str]],
    declared_slots: dict[str, dict],
    config_data: dict[str, Any],
    script_args: list[str],
    name: str | None,
    tags: list[str],
    description: str | None
) -> list[ExperimentSpec]:
    """
    Create multiple experiments for dependency sweep (cartesian product).

    Args:
        script_path: Script to execute
        parsed_deps: Dependencies with multiple values per slot
                    Example: {"dataprep": ["dp1"], "training": ["tr1", "tr2", "tr3"]}
        declared_slots: Dependency declarations from config
        config_data: Configuration parameters
        script_args: Script-specific arguments
        name: Base name for experiments
        tags: Tags to apply
        description: Description template

    Returns:
        List of ExperimentSpec objects (one per combination)

    Example:
        Input: {"dataprep": ["dp1"], "training": ["tr1", "tr2"]}
        Output: [
            ExperimentSpec(depends_on={"dataprep": "dp1", "training": "tr1"}),
            ExperimentSpec(depends_on={"dataprep": "dp1", "training": "tr2"})
        ]
    """
    from itertools import product

    # Extract slots and their value lists
    slots = sorted(parsed_deps.keys())  # Sort for consistent ordering
    value_lists = [parsed_deps[slot] for slot in slots]

    # Generate all combinations (cartesian product)
    combinations = list(product(*value_lists))

    console = Console()
    console.print(
        f"Creating dependency sweep: {len(combinations)} experiments "
        f"({' × '.join(f'{len(v)} {s}' for s, v in zip(slots, value_lists))})"
    )

    experiments = []

    for idx, combo in enumerate(combinations, start=1):
        # Build resolved dependencies for this combination
        resolved_deps = dict(zip(slots, combo))

        # Generate descriptive name
        if name:
            # User provided base name
            dep_suffix = "_".join(f"{slot}={exp_id[:4]}" for slot, exp_id in resolved_deps.items())
            exp_name = f"{name}_{dep_suffix}"
        else:
            # Auto-generate name from script and dependencies
            dep_desc = "_".join(f"{exp_id[:4]}" for exp_id in combo)
            exp_name = f"{script_path.stem}_{dep_desc}"

        # Create experiment spec
        spec = ExperimentSpec(
            script_path=script_path,
            script_args=script_args,
            config=config_data.copy(),  # Each gets own copy
            name=exp_name,
            tags=list(tags) + ["dependency-sweep"],  # Add sweep tag
            description=description,
            dependencies=declared_slots,  # Declaration
            depends_on=resolved_deps      # Fulfillment for this combo
        )

        experiments.append(spec)

        # Log details
        dep_str = ", ".join(f"{s}={id}" for s, id in resolved_deps.items())
        console.print(f"  [{idx}/{len(combinations)}] {exp_name}: {dep_str}")

    return experiments


def should_create_sweep(parsed_deps: dict[str, list[str]]) -> bool:
    """
    Check if dependencies create a sweep (multiple values for any slot).

    Args:
        parsed_deps: Parsed dependency arguments

    Returns:
        True if any slot has multiple values

    Examples:
        >>> should_create_sweep({"dataprep": ["dp1"], "training": ["tr1"]})
        False

        >>> should_create_sweep({"training": ["tr1", "tr2", "tr3"]})
        True

        >>> should_create_sweep({"a": ["x", "y"], "b": ["z"]})
        True
    """
    return any(len(ids) > 1 for ids in parsed_deps.values())


# Integration into yanex run command:

def run(..., depends_on, parallel, ...):
    """Run command with dependency sweep support."""

    # ... config loading and validation ...

    parsed_deps = parse_dependency_args(depends_on)

    # Check if sweep needed
    if should_create_sweep(parsed_deps):
        # Multiple experiments needed
        console.print("[bold]Dependency sweep detected[/bold]")

        experiments = create_dependency_sweep(
            script_path=script,
            parsed_deps=parsed_deps,
            declared_slots=declared_slots,
            config_data=config_data,
            script_args=script_args,
            name=name,
            tags=tag,
            description=description
        )

        # Execute using batch executor
        from ...executor import run_multiple

        results = run_multiple(
            experiments=experiments,
            parallel=parallel,
            verbose=verbose
        )

        # Display results summary
        display_sweep_results(results)
        return

    # Single experiment - continue with normal flow
    resolved_deps = {slot: ids[0] for slot, ids in parsed_deps.items()}
    # ... rest of single experiment logic ...


def display_sweep_results(results: list[ExperimentResult]) -> None:
    """Display summary of dependency sweep results."""
    from rich.table import Table

    console = Console()

    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    console.print(f"\n[bold]Dependency Sweep Results[/bold]")
    console.print(f"Total: {len(results)} experiments")
    console.print(f"✓ Completed: {len(completed)}")
    console.print(f"✗ Failed: {len(failed)}")

    if failed:
        console.print("\n[bold red]Failed Experiments:[/bold red]")
        for result in failed:
            console.print(f"  {result.experiment_id}: {result.error_message}")

    # Create table of results
    table = Table(title="Experiment Results")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="green")
    table.add_column("Duration", justify="right")

    for result in results:
        status_style = "green" if result.status == "completed" else "red"
        duration_str = f"{result.duration:.1f}s" if result.duration else "-"

        table.add_row(
            result.experiment_id,
            result.name or "-",
            f"[{status_style}]{result.status}[/{status_style}]",
            duration_str
        )

    console.print(table)
```

---

## 7. Results API Convenience Properties

**INSERT AFTER:** Line 1714 (in Results API section)

### Add to Experiment Class

```python
# yanex/results/experiment.py

@property
def has_dependencies(self) -> bool:
    """
    Check if experiment has any dependencies.

    Returns:
        True if experiment depends on other experiments

    Example:
        >>> exp = Experiment("eval1")
        >>> if exp.has_dependencies:
        ...     print("This experiment has dependencies")
    """
    metadata = self._load_metadata()
    summary = metadata.get("dependencies_summary", {})
    return summary.get("has_dependencies", False)


@property
def dependency_count(self) -> int:
    """
    Get number of direct dependencies.

    Returns:
        Count of experiments this one depends on

    Example:
        >>> exp = Experiment("eval1")
        >>> print(f"Depends on {exp.dependency_count} experiments")
    """
    metadata = self._load_metadata()
    summary = metadata.get("dependencies_summary", {})
    return summary.get("dependency_count", 0)


@property
def dependency_slots(self) -> list[str]:
    """
    Get list of dependency slot names.

    Returns:
        List of slot names (e.g., ["dataprep", "training"])

    Example:
        >>> exp = Experiment("eval1")
        >>> print(f"Dependencies: {', '.join(exp.dependency_slots)}")
        Dependencies: dataprep, training
    """
    metadata = self._load_metadata()
    summary = metadata.get("dependencies_summary", {})
    return summary.get("dependency_slots", [])


@property
def is_depended_by(self) -> bool:
    """
    Check if other experiments depend on this one.

    Returns:
        True if this experiment is a dependency for others

    Example:
        >>> exp = Experiment("dp1")
        >>> if exp.is_depended_by:
        ...     print("WARNING: Other experiments depend on this!")
    """
    metadata = self._load_metadata()
    summary = metadata.get("dependencies_summary", {})
    return summary.get("depended_by_count", 0) > 0


@property
def depended_by_count(self) -> int:
    """
    Get number of experiments that depend on this one.

    Returns:
        Count of dependent experiments

    Example:
        >>> exp = Experiment("dp1")
        >>> print(f"{exp.depended_by_count} experiments depend on this")
    """
    metadata = self._load_metadata()
    summary = metadata.get("dependencies_summary", {})
    return summary.get("depended_by_count", 0)


@property
def is_root(self) -> bool:
    """
    Check if this is a root node (no dependencies).

    Returns:
        True if experiment has no dependencies

    Example:
        >>> exp = Experiment("dp1")
        >>> if exp.is_root:
        ...     print("This is a pipeline entry point")
    """
    return not self.has_dependencies


@property
def is_leaf(self) -> bool:
    """
    Check if this is a leaf node (nothing depends on it).

    Returns:
        True if no experiments depend on this one

    Example:
        >>> exp = Experiment("eval1")
        >>> if exp.is_leaf:
        ...     print("This is a pipeline exit point")
    """
    return not self.is_depended_by
```

---

## 8. Enhanced Error Messages

**INSERT/EXPAND AT:** Line 3488 (Error Message Catalog)

### Complete Error Templates

```python
# yanex/core/dependency_validator.py

class DependencyErrorMessages:
    """Centralized error message templates for dependency validation."""

    @staticmethod
    def missing_dependency(slot_name: str, declared_slots: dict) -> str:
        """Error when required dependency not provided."""
        msg = f"Missing required dependency '{slot_name}'\n\n"

        msg += "Expected dependencies:\n"
        for slot, config in declared_slots.items():
            script = config["script"]
            if slot == slot_name:
                msg += f"  ✗ {slot} ({script}) - MISSING\n"
            else:
                msg += f"  ✓ {slot} ({script})\n"

        msg += f"\nTo fix:\n"
        msg += f"  yanex run <script> --depends-on {slot_name}=<experiment_id>\n\n"
        msg += f"Find experiments:\n"
        script = declared_slots[slot_name]["script"]
        msg += f"  yanex id --script {script} --status completed\n"

        return msg

    @staticmethod
    def experiment_not_found(
        slot_name: str,
        experiment_id: str,
        expected_script: str
    ) -> str:
        """Error when dependency experiment doesn't exist."""
        msg = f"Dependency experiment '{experiment_id}' not found\n\n"
        msg += f"Slot: {slot_name}\n"
        msg += f"Required script: {expected_script}\n\n"

        # Try to find recent experiments with that script
        try:
            from ...results.manager import ResultsManager
            rm = ResultsManager()
            recent = rm.filter(script_pattern=expected_script, limit=5)

            if recent:
                msg += f"Recent {expected_script} experiments:\n"
                for exp in recent:
                    status_icon = "✓" if exp.status == "completed" else "✗"
                    time_ago = _format_time_ago(exp.created_at)
                    msg += f"  {status_icon} {exp.id} ({exp.status}, {time_ago})\n"
            else:
                msg += f"No {expected_script} experiments found.\n"
                msg += f"Run: yanex run {expected_script}\n"
        except Exception:
            pass

        msg += f"\nList all: yanex list --script {expected_script}\n"
        return msg

    @staticmethod
    def script_mismatch(
        slot_name: str,
        expected_script: str,
        actual_script: str,
        experiment_id: str,
        all_declared_slots: dict
    ) -> str:
        """Error when dependency has wrong script."""
        msg = f"Dependency script mismatch for slot '{slot_name}'\n\n"
        msg += f"Expected: {expected_script}\n"
        msg += f"Actual: {actual_script} (experiment {experiment_id})\n\n"

        # Check if user swapped dependencies
        swapped_suggestions = []
        for other_slot, config in all_declared_slots.items():
            if config["script"] == actual_script and other_slot != slot_name:
                swapped_suggestions.append(other_slot)

        if swapped_suggestions:
            msg += "It looks like you may have swapped dependencies.\n"
            msg += "Did you mean:\n"
            msg += "  --depends-on "

            # Build corrected command
            parts = []
            for slot, config in all_declared_slots.items():
                if slot == slot_name:
                    # Use the swapped suggestion
                    parts.append(f"{slot}=<{expected_script}_id>")
                elif slot in swapped_suggestions:
                    parts.append(f"{slot}={experiment_id}")
                else:
                    parts.append(f"{slot}=<id>")

            msg += " \\\n  --depends-on ".join(parts)

        return msg

    @staticmethod
    def not_completed(
        slot_name: str,
        experiment_id: str,
        current_status: str
    ) -> str:
        """Error when dependency is not completed."""
        msg = f"Dependency '{slot_name}' ({experiment_id}) is not completed\n\n"
        msg += f"Current status: {current_status}\n"
        msg += f"Required status: completed\n\n"
        msg += "Dependencies must be completed experiments to ensure reproducibility.\n\n"

        if current_status == "running":
            msg += "Wait for completion:\n"
            msg += f"  yanex show {experiment_id}\n"
        elif current_status == "failed":
            msg += "Fix the failed experiment:\n"
            msg += f"  yanex show {experiment_id}\n"
            msg += "Then re-run it or use a different experiment.\n"
        elif current_status == "created":
            msg += "The experiment was created but never started.\n"
            msg += f"Delete it: yanex delete {experiment_id}\n"

        return msg

    @staticmethod
    def unknown_slot(
        unknown_slots: set[str],
        declared_slots: dict
    ) -> str:
        """Error when provided slot not in declaration (STRICT mode)."""
        from difflib import get_close_matches

        msg = f"Unknown dependency slot(s): {', '.join(sorted(unknown_slots))}\n\n"
        msg += "Expected slots (from config):\n"
        for slot, config in declared_slots.items():
            script = config["script"]
            msg += f"  ✓ {slot} (script: {script})\n"

        # Provide typo suggestions
        for unknown in unknown_slots:
            suggestions = get_close_matches(
                unknown,
                declared_slots.keys(),
                n=3,
                cutoff=0.6
            )
            if suggestions:
                msg += f"\nDid you mean '{suggestions[0]}' instead of '{unknown}'?\n"

        msg += "\nMake sure slot names match exactly (case-sensitive).\n"

        return msg

    @staticmethod
    def circular_dependency(path: list[str]) -> str:
        """Error when circular dependency detected."""
        cycle_path = " → ".join(path)

        msg = "Circular dependency detected:\n"
        msg += f"  {cycle_path}\n\n"
        msg += "This would create an infinite loop in the dependency graph.\n"
        msg += "Experiment workflows must be directed acyclic graphs (DAGs).\n\n"
        msg += "To fix:\n"
        msg += f"  1. Review dependency chain: yanex show {path[0]}\n"
        msg += "  2. Remove the circular dependency\n"
        msg += "  3. Restructure workflow as a DAG\n"

        return msg

    @staticmethod
    def archived_dependency_warning(experiment_id: str, slot_name: str) -> str:
        """Warning when using archived experiment as dependency."""
        msg = f"⚠️  Dependency '{slot_name}' is an archived experiment: {experiment_id}\n"
        msg += "   This is allowed but the experiment data is in the archive.\n"

        return msg


def _format_time_ago(timestamp: datetime) -> str:
    """Format timestamp as relative time (e.g., '2 hours ago')."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    delta = now - timestamp

    seconds = delta.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} min ago" if minutes == 1 else f"{minutes} mins ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour ago" if hours == 1 else f"{hours} hours ago"
    else:
        days = int(seconds / 86400)
        return f"{days} day ago" if days == 1 else f"{days} days ago"
```

---

## 9. Updated Phase 1 Testing Checklist

**REPLACE:** Lines 3163-3189 in main document

### Comprehensive Testing Checklist

```markdown
**Phase 1 Testing Checklist:**

CLI Parsing & Validation:
- [ ] Parse single dependency: `-d dataprep=dp1`
- [ ] Parse multiple dependencies: `-d dataprep=dp1 -d training=tr1`
- [ ] Parse sweep dependencies: `-d training=tr1,tr2,tr3`
- [ ] Parse multiple --depends-on for same slot
- [ ] Validate experiment ID format (8-char hex)
- [ ] Reject malformed IDs (too short, non-hex, empty)
- [ ] Reject empty slot names
- [ ] Reject invalid slot name characters
- [ ] Validate slot names (alphanumeric + underscore)
- [ ] Helpful error for `slot=` (empty value)
- [ ] Shell command substitution works: `$(yanex id ...)`

Config File Parsing:
- [ ] Load dependencies from `yanex.scripts[]` array
- [ ] Find correct script entry by name
- [ ] Parse shorthand syntax: `dataprep: dataprep.py`
- [ ] Parse full syntax: `dataprep: {script: dataprep.py, required: true}`
- [ ] Remove `yanex` section from parameters
- [ ] Handle script not in config (no declaration)
- [ ] Handle empty scripts array
- [ ] Handle missing yanex section

Slot Validation (STRICT mode):
- [ ] Required slots must be provided
- [ ] Extra slots rejected with error
- [ ] Typo suggestions for unknown slots (Levenshtein distance)
- [ ] Error if depends_on without declaration
- [ ] Error lists all missing slots
- [ ] Error suggests yanex id command

Dependency Validation:
- [ ] Experiment exists (regular location)
- [ ] Experiment exists (archived location - include_archived=True)
- [ ] Script matches declared slot
- [ ] Status is "completed"
- [ ] Reject running experiments
- [ ] Reject failed experiments
- [ ] Reject created (never started) experiments
- [ ] Warning for archived dependencies
- [ ] Helpful error when experiment not found
- [ ] Helpful error when script mismatch
- [ ] Suggest swapped dependencies when scripts mismatch

Circular Dependency Detection (NEW - Phase 1):
- [ ] Detect direct cycle: A → A
- [ ] Detect 2-node cycle: A → B → A
- [ ] Detect 3-node cycle: A → B → C → A
- [ ] Detect indirect cycle through multiple paths
- [ ] DFS correctly handles visited nodes
- [ ] Error message shows full cycle path
- [ ] Performance acceptable for deep graphs (>10 levels)

Storage:
- [ ] Save dependencies.json with correct schema
- [ ] Load dependencies.json
- [ ] Create empty structure for provider experiments
- [ ] Add dependent to reverse index
- [ ] Remove dependent from reverse index
- [ ] Handle missing dependencies.json (returns None)
- [ ] Update metadata.json with dependencies_summary
- [ ] dependencies_summary fields correct

Reverse Index Updates (no file locking in Phase 1):
- [ ] Update single dependency's depended_by
- [ ] Update multiple dependencies' depended_by
- [ ] Sequential updates work correctly
- [ ] Concurrent updates (best effort, potential race)
- [ ] Create dependencies.json if doesn't exist for provider

Python API - create_experiment:
- [ ] Inline dependencies declaration works
- [ ] Config file dependencies declaration works
- [ ] Inline overrides config declaration
- [ ] Error if depends_on without declaration
- [ ] Validation runs before creation
- [ ] dependencies.json saved correctly
- [ ] Reverse indexes updated

Python API - ExperimentSpec:
- [ ] ExperimentSpec with dependencies field
- [ ] run_multiple() with dependencies
- [ ] Validation runs for each experiment
- [ ] Failed validation doesn't stop batch

Python API - get_dependencies:
- [ ] Returns empty dict outside experiment context
- [ ] Returns correct DependencyReference objects
- [ ] DependencyReference.artifacts_dir correct
- [ ] DependencyReference.artifacts lists files
- [ ] DependencyReference.artifact_path works
- [ ] DependencyReference.load_artifact works
- [ ] load_artifact raises FileNotFoundError for missing

Results API - Experiment properties:
- [ ] has_dependencies property correct
- [ ] dependency_count property correct
- [ ] dependency_slots property correct
- [ ] is_depended_by property correct
- [ ] depended_by_count property correct
- [ ] is_root property correct
- [ ] is_leaf property correct

Results API - Query methods:
- [ ] get_dependencies() returns direct dependencies
- [ ] get_dependencies(recursive=True) returns all ancestors
- [ ] get_dependencies(max_depth=2) respects limit
- [ ] get_dependents() returns direct dependents
- [ ] get_dependents(recursive=True) returns all descendants
- [ ] get_dependency_info() returns slot metadata
- [ ] get_dependency_info() includes validation status
- [ ] get_pipeline() returns full connected subgraph
- [ ] get_pipeline() includes all nodes and edges
- [ ] get_pipeline() identifies root and leaf nodes
- [ ] get_pipeline() handles single-node graph
- [ ] yr.get_pipeline(exp_id) module function works

Dependency Sweeps:
- [ ] should_create_sweep() detects multiple values
- [ ] Cartesian product generated correctly
- [ ] Auto-generated names for sweep experiments
- [ ] User-provided name used as base
- [ ] "dependency-sweep" tag added automatically
- [ ] run_multiple() called with correct specs
- [ ] Each combination validated separately
- [ ] Failed combinations don't stop sweep
- [ ] Results summary displayed

CLI Commands:
- [ ] yanex run -d single dependency
- [ ] yanex run -d multiple dependencies
- [ ] yanex run -d sweep dependencies
- [ ] yanex show displays dependencies section
- [ ] yanex show displays depended_by section
- [ ] yanex list shows dependency indicator
- [ ] yanex delete prevents deletion if depended on
- [ ] yanex delete --force allows deletion with dependents
- [ ] yanex id command works
- [ ] yanex id --format csv outputs comma-separated
- [ ] yanex id --format json outputs JSON array
- [ ] yanex id --limit N limits results

Error Messages:
- [ ] Missing dependency error is helpful
- [ ] Experiment not found error suggests alternatives
- [ ] Script mismatch error suggests swapped deps
- [ ] Not completed error has status-specific guidance
- [ ] Unknown slot error has typo suggestions
- [ ] Circular dependency error shows full path
- [ ] All errors reference yanex commands for help

Edge Cases:
- [ ] Empty depends_on (no dependencies)
- [ ] Dependency on self rejected (circular)
- [ ] Archived experiment as dependency works
- [ ] Multiple experiments with same dependencies
- [ ] Creating dependency while parent is running (status check fails)
- [ ] Malformed dependencies.json (load returns None or errors gracefully)
- [ ] Very deep dependency chains (>20 levels)
- [ ] Wide fan-out (>50 dependents on one experiment)

Integration Tests:
- [ ] Full workflow: dataprep → train → evaluate
- [ ] Parameter sweep + dependency sweep combination
- [ ] Parallel execution with dependencies
- [ ] Archive experiment with dependents
- [ ] Delete experiment updates reverse indexes
- [ ] Run multiple experiments depending on same parent
```

---

## 10. Phase 1 Detailed Task Breakdown

**INSERT AT END** of main document (before "References" section)

### Phase 1 Implementation Plan (40-50 hours)

Tasks are ordered by dependency. Each task includes acceptance criteria.

#### Week 1: Core Infrastructure (16-20 hours)

**Task 1.1: Exception Classes (30 mins)**
- File: `yanex/utils/exceptions.py`
- Add: `DependencyError`, `CircularDependencyError`, `MissingDependencyError`, `InvalidDependencyError`
- Acceptance: All exception classes defined with docstrings

**Task 1.2: Storage Module (3-4 hours)**
- File: `yanex/core/storage_dependencies.py`
- Implement: `FileSystemDependencyStorage` class
- Methods: `save_dependencies`, `load_dependencies`, `add_dependent`, `remove_dependent`
- Acceptance: All methods work, unit tests pass

**Task 1.3: Integrate Storage into Composition (1 hour)**
- File: `yanex/core/storage_composition.py`
- Add dependency storage to composite
- Acceptance: `storage.save_dependencies()` works from manager

**Task 1.4: Metadata Schema Enhancement (2 hours)**
- File: `yanex/core/storage_metadata.py`
- Add `dependencies_summary` field to metadata.json
- Update when dependencies are saved
- Acceptance: Summary fields populated correctly

**Task 1.5: Circular Dependency Detector (3-4 hours)**
- File: `yanex/core/circular_dependency_detector.py`
- Implement: `CircularDependencyDetector` class with DFS
- Method: `check_for_cycles(experiment_id, resolved_deps)`
- Acceptance: Detects 2-node and 3-node cycles, unit tests pass

**Task 1.6: Dependency Validator (4-5 hours)**
- File: `yanex/core/dependency_validator.py`
- Implement: `DependencyValidator` class
- Validate: existence, script match, status, circular dependencies
- Use: `DependencyErrorMessages` for helpful errors
- Acceptance: All validation levels work, error messages helpful

**Task 1.7: Error Message Templates (2 hours)**
- File: `yanex/core/dependency_validator.py` (add `DependencyErrorMessages`)
- Implement: All error message templates from section 8
- Acceptance: All templates return helpful messages

#### Week 2: CLI Integration (12-15 hours)

**Task 2.1: Argument Parsing (2-3 hours)**
- File: `yanex/cli/commands/run.py`
- Implement: `parse_dependency_args()` with full validation
- Implement: `validate_dependency_slots()` (STRICT mode)
- Acceptance: Parses single/multiple/sweep, validates hex IDs

**Task 2.2: Config Resolution (2-3 hours)**
- File: `yanex/cli/commands/run.py`
- Implement: `resolve_dependency_declaration()`
- Parse `yanex.scripts[]` array, normalize syntax
- Acceptance: Finds script in config, normalizes shorthand/full syntax

**Task 2.3: Dependency Sweep Logic (3-4 hours)**
- File: `yanex/cli/commands/run.py`
- Implement: `create_dependency_sweep()`, `should_create_sweep()`
- Generate cartesian product, auto-name experiments
- Acceptance: Creates correct number of specs, names descriptive

**Task 2.4: Integrate into yanex run (3-4 hours)**
- File: `yanex/cli/commands/run.py`
- Add `-d/--depends-on` option
- Call validators before creation
- Update reverse indexes after creation
- Handle sweeps vs single experiments
- Acceptance: Full workflow works, dependencies saved

**Task 2.5: Update yanex show (1 hour)**
- File: `yanex/cli/commands/show.py`
- Add dependencies section to output
- Show depended_by section
- Acceptance: Displays dependencies and dependents correctly

**Task 2.6: Update yanex list (1 hour)**
- File: `yanex/cli/commands/list.py`
- Add dependency indicator column (optional)
- Acceptance: List shows which experiments have dependencies

#### Week 3: Python API & Results API (12-15 hours)

**Task 3.1: Enhance ExperimentSpec (1 hour)**
- File: `yanex/executor.py`
- Add `dependencies` and `depends_on` fields
- Update validation
- Acceptance: ExperimentSpec supports dependencies

**Task 3.2: Update create_experiment API (3-4 hours)**
- File: `yanex/api.py`
- Add `dependencies` and `depends_on` parameters
- Implement resolution hierarchy (inline → config → none)
- Call validator, save dependencies, update reverse indexes
- Acceptance: All three declaration modes work

**Task 3.3: Implement get_dependencies API (2-3 hours)**
- File: `yanex/api.py`
- Implement: `get_dependencies()`, `DependencyReference` class
- Properties: `artifacts_dir`, `artifacts`, `artifact_path`, `load_artifact`
- Acceptance: Scripts can access dependency artifacts

**Task 3.4: Results API - Convenience Properties (1 hour)**
- File: `yanex/results/experiment.py`
- Add properties: `has_dependencies`, `dependency_count`, `dependency_slots`, `is_depended_by`, `depended_by_count`, `is_root`, `is_leaf`
- Acceptance: All properties return correct values

**Task 3.5: Results API - Query Methods (4-5 hours)**
- File: `yanex/results/experiment.py`
- Implement: `get_dependencies(recursive, max_depth)`
- Implement: `get_dependents(recursive, max_depth)`
- Implement: `get_dependency_info()`
- Implement: `get_pipeline()` / `get_dependency_graph()`
- Acceptance: All methods work, handle edge cases

**Task 3.6: Results API - Module Function (30 mins)**
- File: `yanex/results/__init__.py`
- Implement: `get_pipeline(experiment_id)`
- Acceptance: Can call `yr.get_pipeline(exp_id)`

#### Week 4: Testing & Documentation (10-12 hours)

**Task 4.1: Unit Tests - Storage (2 hours)**
- File: `tests/core/test_dependency_storage.py`
- Test: save, load, add_dependent, remove_dependent
- Acceptance: 100% coverage of storage module

**Task 4.2: Unit Tests - Validation (3 hours)**
- File: `tests/core/test_dependency_validation.py`
- Test: all validation levels, error messages
- File: `tests/core/test_circular_dependencies.py`
- Test: cycle detection (direct, indirect, deep)
- Acceptance: All validation scenarios covered

**Task 4.3: Unit Tests - CLI (2 hours)**
- File: `tests/cli/test_dependency_parsing.py`
- Test: argument parsing, slot validation
- Acceptance: All parsing edge cases covered

**Task 4.4: Integration Tests (2 hours)**
- File: `tests/integration/test_dependency_workflow.py`
- Test: full workflows (dataprep → train → evaluate)
- Test: dependency sweeps
- Acceptance: End-to-end workflows work

**Task 4.5: API Tests (2 hours)**
- File: `tests/api/test_dependency_api.py`
- Test: create_experiment, get_dependencies, ExperimentSpec
- File: `tests/results/test_dependency_results_api.py`
- Test: all Results API methods
- Acceptance: 90%+ coverage maintained

**Task 4.6: Documentation (3 hours)**
- Update: README.md with dependency example
- Create: docs/dependencies.md (complete guide)
- Update: docs/configuration.md (yanex.scripts[] array)
- Update: docs/cli-reference.md (--depends-on flag)
- Create: examples/dependencies/ (dataprep, train, evaluate, config)
- Acceptance: All docs clear and accurate

---

## Summary of Additions

This addendum provides:

1. ✅ **Complete config file schema** with shorthand and full syntax
2. ✅ **Detailed validation sequence** with rollback behavior
3. ✅ **Enhanced argument parsing** with ID validation and typo detection
4. ✅ **Config lookup implementation** with normalization
5. ✅ **Circular dependency detection** with DFS algorithm (Phase 1)
6. ✅ **Dependency sweep logic** with cartesian product
7. ✅ **Results API properties** for quick access
8. ✅ **Comprehensive error messages** with templates
9. ✅ **Updated testing checklist** with 120+ test cases
10. ✅ **Detailed task breakdown** with 26 tasks over 4 weeks

**Integration Instructions:**

1. Insert sections 1-8 at indicated line numbers in main document
2. Replace Phase 1 testing checklist (section 9)
3. Append Phase 1 task breakdown (section 10) before References

**Total Estimated Effort:** 40-50 hours for Phase 1 MVP

All design decisions are finalized and ready for implementation!
