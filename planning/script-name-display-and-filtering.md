# Implementation Plan: Script Name Display and Filtering

**Date**: 2025-11-06
**Status**: Planning
**Feature**: Add script name display and filtering to yanex commands

## Overview

Add script name (filename with extension, e.g., "train.py") to `yanex list`, `yanex show`, and `yanex compare` commands, plus filtering via `--script/-c` flag across all filtering commands.

## Motivation

Users often run multiple different yanex scripts (e.g., `dataprep.py`, `train.py`, `predict.py`) and need to:
- Quickly identify which script was used for an experiment
- Filter experiments by script name
- Group/compare experiments from the same script

Currently, the full script path is stored in metadata but not prominently displayed or filterable.

**Design Decision**: Display full filename with extension (e.g., "train.py" not "train") for clarity and familiarity. Users can filter with or without extension for flexibility.

## Impact: Commands Affected by `--script` Filter

The `@experiment_filter_options()` decorator will add `--script/-c` to **6 commands**:

1. **`yanex list`** - Display script column + filtering ✓
2. **`yanex compare`** - Display script column + filtering ✓
3. **`yanex archive`** - Filtering only (useful!)
4. **`yanex unarchive`** - Filtering only (useful!)
5. **`yanex delete`** - Filtering only (useful!)
6. **`yanex update`** - Filtering only (useful!)

**Decision**: This is beneficial - users can archive/delete/update experiments by script type.

### Shortcut Selection

**Chosen**: `-c` for `--script`

**Rationale**:
- `-c` is available (not used by any current filter)
- Mnemonic: "-c" for "code" or "command"
- Existing shortcuts: `-t` (tag), `-n` (name), `-s` (status), `-a` (archived), `-l` (limit)

## Changes Required (11 files)

### Phase 1: Core Filtering Infrastructure

#### 1. CLI Filter Arguments - `yanex/cli/filters/arguments.py`

**Location**: After line 78 (in core filters section)

**Add**:
```python
func = click.option(
    "--script",
    "-c",
    "script_pattern",
    help="Filter by script name using glob patterns (e.g., 'train.py', '*prep*'). Extensions are optional.",
)(func)
```

**Impact**: All 6 commands using `@experiment_filter_options()` automatically get this option.

---

#### 2. Core Filtering Logic - `yanex/core/filtering.py`

**Changes**:

1. **Add parameter to `filter_experiments()` (line 38)**:
   ```python
   def filter_experiments(
       self,
       # ... existing parameters ...
       script_pattern: str | None = None,
   ) -> list[dict[str, Any]]:
   ```

2. **No normalization needed in `_normalize_filter_inputs()`**:
   - Filter will match both with and without extension
   - No need to strip `.py` extension

3. **Add filtering in `_apply_all_filters()` (after line 218)**:
   ```python
   if "script_pattern" in filters:
       filtered = [
           exp
           for exp in filtered
           if self._matches_script_pattern(exp, filters["script_pattern"])
       ]
   ```

4. **Create helper method `_matches_script_pattern()` (after line 327)**:
   ```python
   def _matches_script_pattern(
       self, experiment: dict[str, Any], pattern: str
   ) -> bool:
       """Check if script name matches glob pattern.

       Matches against both the full filename (e.g., 'train.py') and stem (e.g., 'train').
       This allows users to filter with or without the .py extension.
       Returns False if no script_path in experiment metadata.
       """
       from pathlib import Path
       import fnmatch

       script_path = experiment.get("script_path", "")
       if not script_path:
           return False

       script_name = Path(script_path).name  # Full filename: "train.py"
       script_stem = Path(script_path).stem  # Stem only: "train"

       # Match against both full name and stem for flexibility
       return (
           fnmatch.fnmatch(script_name.lower(), pattern.lower()) or
           fnmatch.fnmatch(script_stem.lower(), pattern.lower())
       )
   ```

---

#### 3. Filter Validation - `yanex/cli/filters/validation.py`

**Review**: Check if any validation logic needs updating for script patterns.

**Likely**: No changes needed (glob patterns handled by fnmatch, same as name filtering).

---

### Phase 2: Update All 6 Command Signatures

All commands using `@experiment_filter_options()` need to accept the new `script` parameter and pass it to filtering logic.

#### 4. List Command - `yanex/cli/commands/list.py`

**Line 20**: Add parameter to function signature:
```python
def list_experiments(
    # ... existing parameters ...
    script: str | None = None,
) -> None:
```

**Line 119**: Pass to `filter_experiments()`:
```python
experiments = experiment_filter.filter_experiments(
    # ... existing parameters ...
    script_pattern=script,
)
```

---

#### 5. Compare Command - `yanex/cli/commands/compare.py`

**Add parameter** and **pass to filtering call**.

Pattern same as list command.

---

#### 6. Archive Command - `yanex/cli/commands/archive.py`

**Add parameter** and **pass to filtering call**.

Pattern same as list command.

---

#### 7. Unarchive Command - `yanex/cli/commands/unarchive.py`

**Add parameter** and **pass to filtering call**.

Pattern same as list command.

---

#### 8. Delete Command - `yanex/cli/commands/delete.py`

**Add parameter** and **pass to filtering call**.

Pattern same as list command.

---

#### 9. Update Command - `yanex/cli/commands/update.py`

**Add parameter** and **pass to filtering call**.

Pattern same as list command.

---

### Phase 3: Display Formatting

#### 10. Table Formatter - `yanex/cli/formatters/console.py`

**Method**: `format_experiments_table()` (line 55-87)

**Changes**:

1. **Add "Script" column (line 69, as 2nd column after ID)**:
   ```python
   table.add_column("ID", style="dim", width=8)
   table.add_column("Script", style="cyan", width=18)  # NEW - width 18 for "filename.py"
   table.add_column("Name", min_width=12, max_width=25)
   # ... rest of columns
   ```

2. **Add script to row building (line 85)**:
   ```python
   for exp in experiments:
       table.add_row(
           self._format_id(exp.get("id", "")),
           self._format_script(exp.get("script_path")),  # NEW
           self._format_name(exp.get("name")),
           # ... rest of row values
       )
   ```

3. **Create helper method `_format_script()` (after line 285)**:
   ```python
   def _format_script(self, script_path: str | None) -> Text:
       """Format script name from full path.

       Extracts filename with extension and truncates if needed.
       """
       from pathlib import Path

       if not script_path:
           return Text("-", style="dim")

       script_name = Path(script_path).name  # Full filename: "train.py"

       # Truncate if too long (keep extension visible)
       if len(script_name) > 18:
           # Keep first chars and extension: "very_long_na....py"
           script_name = script_name[:14] + "..." + script_name[-3:]

       return Text(script_name, style="cyan")
   ```

---

#### 11. Comparison Formatter - `yanex/core/comparison.py`

**Method**: `_build_experiment_row()` (line 216-260)

**Change**: Add script field to fixed columns (after line 238):

```python
row = {
    "id": exp_data["id"],
    "script": Path(exp_data.get("script_path", "")).name or "-",  # NEW (2nd field) - full filename with extension
    "name": exp_data.get("name") or "[unnamed]",
    "started": self._format_datetime(exp_data.get("started_at")),
    "duration": self._calculate_duration(...),
    "status": exp_data["status"],
    "tags": self._format_tags(exp_data.get("tags", [])),
}
```

**Note**: The comparison module already extracts `script_path` from metadata (line 101), so no additional data extraction needed.

---

#### 12. Show Command - `yanex/cli/commands/show.py`

**Status**: Already displays script_path in Environment section (line 492-494)

**No changes needed**: Full path is appropriate for detail view.

---

## Implementation Details

### Script Name Extraction Pattern

```python
from pathlib import Path

# Get filename WITH extension for display
script_name = Path(script_path).name  # "train.py"
```

### Filter Normalization

**No normalization needed** - filter matches both with and without extension:
- User types `--script train` → matches "train.py"
- User types `--script train.py` → matches "train.py"
- Both work due to dual matching in `_matches_script_pattern()`

### Pattern Matching

Match against BOTH full filename and stem for flexibility:

```python
import fnmatch
from pathlib import Path

def _matches_script_pattern(self, experiment: dict, pattern: str) -> bool:
    script_path = experiment.get("script_path", "")
    if not script_path:
        return False

    script_name = Path(script_path).name  # "train.py"
    script_stem = Path(script_path).stem  # "train"

    # Match against both for user flexibility
    return (
        fnmatch.fnmatch(script_name.lower(), pattern.lower()) or
        fnmatch.fnmatch(script_stem.lower(), pattern.lower())
    )
```

### Column Layout

**yanex list** table columns:
- ID (8 chars, dim)
- **Script (18 chars, cyan)** ← NEW, 2nd column (fits "filename.py" + margin)
- Name (12-25 chars)
- Status (12 chars)
- Duration (10 chars, right-aligned)
- Tags (8-20 chars)
- Started (15 chars, right-aligned)

**yanex compare** fixed columns:
- id
- **script** ← NEW, 2nd field
- name
- started
- duration
- status
- tags

---

## Testing Strategy

### Unit Tests

1. **Filtering Logic Tests** (`tests/core/test_filtering.py`):
   - Test `_matches_script_pattern()` with various patterns
   - Test exact matches with extension: `"train.py"` matches `train.py`
   - Test exact matches without extension: `"train"` matches `train.py`
   - Test glob patterns: `"*prep*"` matches `dataprep.py`, `data_prep.py`
   - Test case insensitivity: `"Train"` and `"TRAIN.PY"` both match `train.py`
   - Test missing script_path: returns `False`
   - Test dual matching: both `"train"` and `"train.py"` match the same experiment

2. **Command Tests**:
   - Test all 6 commands accept `--script` and `-c` flags
   - Test filtering works end-to-end
   - Test invalid patterns (should not error, just return no results)

3. **Formatter Tests** (`tests/cli/formatters/test_console.py`):
   - Test script column appears in correct position (2nd)
   - Test `_format_script()` handles missing paths
   - Test truncation for long script names (preserves extension)
   - Test script name extraction (with extension: "train.py")

4. **Integration Tests**:
   - Create experiments with different scripts
   - Test `yanex list --script train` filters correctly
   - Test `yanex compare --script "*prep*"` glob matching
   - Test column ordering and formatting

### Edge Cases

- Missing `script_path` in metadata (old experiments) → display "-"
- Very long script names (> 18 chars) → truncate but preserve extension
- Scripts with special characters in filename
- Scripts in subdirectories (should show filename only, not full path)
- Empty script_pattern (should show all experiments)
- Filter with/without extension: both `"train"` and `"train.py"` should match

---

## User Experience Examples

### Filtering Examples

```bash
# List all training experiments (both forms work identically)
yanex list --script train        # matches "train.py"
yanex list --script train.py     # matches "train.py"

# Short form
yanex list -c train.py

# Glob patterns
yanex list --script "*prep*"     # matches dataprep.py, data_prep.py, etc.
yanex list --script "data*"      # matches dataprep.py, data_analysis.py, etc.
yanex list --script "*.py"       # matches all .py scripts

# Archive all prediction scripts
yanex archive --script predict.py

# Delete old data preparation runs
yanex delete --script dataprep --ended-before "1 week ago"

# Compare training experiments
yanex compare --script train

# Update all training experiments with a tag
yanex update --script train.py --add-tag "training-v2"
```

### Display Examples

**yanex list** output:
```
ID       Script           Name              Status     Duration  Tags        Started
abc123   train.py         baseline-run      completed  15m 23s   ml,v1       2 hours ago
def456   dataprep.py      clean-dataset     completed  3m 12s    data        5 hours ago
ghi789   predict.py       inference-test    running    5m 02s    inference   10 mins ago
jkl012   data_analysis.py feature-eng      completed  8m 45s    data        1 day ago
```

**yanex compare** output (interactive table):
- Script column appears between ID and Name
- Sortable by script name
- Filterable in interactive mode

**yanex show** output:
```
Environment
  Script:     /Users/thomas/code/project/train.py
  Working Dir: /Users/thomas/code/project
  ...
```

---

## Implementation Order

1. **Core filtering** (phase 1): Add filter infrastructure
2. **Command signatures** (phase 2): Update all 6 commands
3. **Display formatting** (phase 3): Add columns to list/compare
4. **Testing**: Add comprehensive test coverage
5. **Documentation**: Update CLI help text (automatic via decorator)

## Backward Compatibility

- ✅ No breaking changes
- ✅ Existing experiments without script_path display "-"
- ✅ All existing commands continue to work without `--script` flag
- ✅ Metadata structure unchanged (script_path already stored)

## Success Criteria

- [ ] All 6 commands accept `--script/-c` filter option
- [ ] `yanex list` shows Script column (2nd position) with full filename including `.py` extension
- [ ] `yanex compare` shows Script column (2nd position) with full filename including `.py` extension
- [ ] `yanex show` continues to show full script path (unchanged)
- [ ] Glob pattern matching works (exact, wildcards, case-insensitive)
- [ ] Dual matching works: both `--script train` and `--script train.py` match the same experiments
- [ ] Missing script_path handled gracefully (shows "-")
- [ ] Long script names truncated but preserve extension visibility
- [ ] All tests pass with 90%+ coverage
- [ ] `uv run ruff check` and `uv run ruff format` pass
