# CLI Implementation Plan

## Overview

The yanex CLI will provide a seamless way to run experiment scripts with automatic tracking, while ensuring scripts work unchanged in both standalone and CLI modes.

## Core Design Principles

1. **Zero script modifications**: Scripts should work identically with `python script.py` and `yanex run script.py`
2. **Graceful degradation**: All yanex functions work safely in standalone mode
3. **Clean API**: No try/except blocks required in user scripts
4. **Flexible configuration**: Multiple ways to specify parameters with clear precedence

## CLI Library Choice

**Selected: Click**
- Modern, well-maintained, widely adopted
- Excellent subcommand support
- Clean decorator-based syntax
- Built-in configuration file support
- Great error handling and help generation

## Improved Experiment API for CLI Compatibility

### Current Issues
- `experiment.get_param()` raises exceptions in standalone mode
- Logging functions throw errors when no experiment context exists
- Requires try/except blocks in user scripts

### Proposed Improvements

1. **Safe parameter access:**
   ```python
   # Works in both modes - returns default if no context
   learning_rate = experiment.get_param("learning_rate", 0.01)
   ```

2. **Mode detection utilities:**
   ```python
   if experiment.is_standalone():
       print("Running without yanex")
   else:
       print(f"Experiment: {experiment.get_experiment_id()}")
   ```

3. **Safe logging (no-op in standalone):**
   ```python
   # Always safe to call - does nothing in standalone mode
   experiment.log_results({"accuracy": 0.95})
   experiment.log_artifact("model.pkl", path)
   ```

### Implementation Details

- Modify `_get_current_experiment_id()` to return None instead of raising exception
- Update `get_param()` to handle None experiment ID gracefully
- Make all logging functions check for experiment context before operating
- Add `is_standalone()` and `has_context()` utility functions

## Run Command Design

### Command Structure
```bash
yanex run <script> [options]
```

### Options
```bash
--config FILE         # Configuration file (YAML/JSON)
--param KEY=VALUE     # Parameter override (repeatable)
--name NAME           # Experiment name
--tag TAG             # Experiment tag (repeatable)  
--description TEXT    # Experiment description
--dry-run            # Validate without running
--help               # Show help
```

### Configuration Precedence (highest to lowest)
1. CLI parameter overrides (`--param key=value`)
2. CLI config file (`--config file.yaml`)
3. Default config file (`yanex.yaml` in current directory)
4. Environment variables (`YANEX_PARAM_*`)
5. Script defaults (from `get_param()` calls)

### Execution Workflow

1. **Parse CLI arguments**
   - Extract script path, config files, parameter overrides
   - Validate all inputs before execution

2. **Merge configuration**
   - Load and merge configs according to precedence
   - Validate parameter types and constraints

3. **Create experiment**
   - Generate experiment ID
   - Create experiment with merged configuration
   - Set up experiment directory structure

4. **Prepare execution environment**
   - Patch yanex.experiment module to auto-set experiment context
   - Set environment variables if needed for subprocess isolation

5. **Execute script**
   - Run script in same process with patched experiment context
   - Handle script exceptions and map to experiment status
   - Ensure experiment lifecycle is properly managed

6. **Handle completion**
   - Mark experiment as completed/failed based on script exit
   - Clean up temporary resources
   - Report experiment ID and location

### Example Usage

**Script (works unchanged):**
```python
import yanex.experiment as experiment

def main():
    # Works in both standalone and CLI modes
    lr = experiment.get_param("learning_rate", 0.01)
    epochs = experiment.get_param("epochs", 10)
    
    if experiment.is_standalone():
        print("Running standalone")
    else:
        print(f"Experiment: {experiment.get_experiment_id()}")
    
    for epoch in range(epochs):
        accuracy = train_epoch()
        experiment.log_results({"accuracy": accuracy}, step=epoch)

if __name__ == "__main__":
    main()
```

**Usage scenarios:**
```bash
# Standalone (no changes to script)
python script.py

# Basic yanex run
yanex run script.py

# With configuration
yanex run script.py --config config.yaml

# With overrides
yanex run script.py --param learning_rate=0.05 --param epochs=20

# Full experiment setup
yanex run script.py \
  --config config.yaml \
  --param learning_rate=0.05 \
  --name "lr-tuning" \
  --tag "hyperopt" \
  --description "Learning rate optimization experiment"
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Set up Click CLI structure with basic `run` command
2. Implement configuration loading and merging logic
3. Add parameter parsing and validation

### Phase 2: Experiment API Improvements  
1. Modify experiment module for standalone compatibility
2. Add `is_standalone()` and safe parameter access
3. Make all logging functions no-op in standalone mode
4. Update existing tests to verify both modes

### Phase 3: Run Command Implementation
1. Implement experiment creation from CLI config
2. Add experiment context patching for script execution
3. Handle script lifecycle and error propagation
4. Add comprehensive error handling and user feedback

### Phase 4: Additional CLI Commands
1. Implement `yanex list` command with comprehensive filtering
2. Implement `yanex show <id>` command for detailed experiment view
3. Implement `yanex status` command for system status
4. Design reusable filtering and formatting components

### Phase 5: Testing and Polish
1. Write comprehensive CLI tests for all commands
2. Test with various script patterns and edge cases
3. Add progress indicators and rich console output
4. Write documentation and examples

## Additional CLI Commands Design

### List Command (`yanex list`)

**Purpose**: Display experiments with powerful filtering and formatting options

**Basic Usage:**
```bash
yanex list                    # Last 10 experiments
yanex list --all              # All experiments
yanex list -n 5               # Last 5 experiments
```

**Filtering Options:**
```bash
--status STATUS               # Filter by status (created/running/completed/failed/cancelled)
--name PATTERN               # Filter by name with glob patterns
--tag TAG                    # Filter by tag (repeatable, AND logic)
--started TIMESPEC           # Filter by start time (human-readable)
--ended TIMESPEC            # Filter by end time (human-readable)
```

**Time Specification Examples:**
- "today", "yesterday", "last week", "2 hours ago"
- ISO dates: "2023-01-01", "2023-12-31T15:30:00"
- Relative: "1 week ago", "2 days ago"

**Output Format:**
- Rich table with colors for different statuses
- Columns: ID, Name, Status, Duration, Tags, Started
- Status colors: 🟢 completed, 🔴 failed, 🟡 running, ⚪ created, 🟠 cancelled

**Architecture Components:**

1. **ExperimentFilter Class** (reusable for other commands):
   ```python
   class ExperimentFilter:
       def filter_experiments(
           self,
           status: Optional[str] = None,
           name_pattern: Optional[str] = None,
           tags: List[str] = None,
           started_after: Optional[datetime] = None,
           started_before: Optional[datetime] = None,
           ended_after: Optional[datetime] = None,
           ended_before: Optional[datetime] = None,
           limit: Optional[int] = None
       ) -> List[Dict[str, Any]]
   ```

2. **Rich Console Formatter**:
   - Status-based colors
   - Duration calculation and formatting
   - Table layout with proper column alignment

3. **Time Parsing Utilities**:
   - Library: `dateparser` for human-readable time parsing
   - Support for natural language and ISO formats
   - Timezone handling and validation

**File Structure:**
```
yanex/cli/
├── commands/
│   ├── run.py           # ✅ Existing
│   ├── list.py          # 🆕 List command
│   ├── show.py          # 🆕 Show command (future)
│   └── status.py        # 🆕 Status command (future)
├── filters/
│   ├── __init__.py      # 🆕 Filter exports
│   ├── base.py          # 🆕 ExperimentFilter class
│   └── time_utils.py    # 🆕 Date parsing utilities
├── formatters/
│   ├── __init__.py      # 🆕 Formatter exports
│   └── console.py       # 🆕 Rich console formatting
└── main.py              # Update to include new commands
```

**Dependencies:**
- `dateparser`: Human-readable date parsing
- `rich`: Enhanced console output (already included)

### Show Command (`yanex show <id>`) - Future

**Purpose**: Display detailed information about a specific experiment
- Complete metadata, configuration, results
- Artifact list with file sizes
- Execution timeline and logs

### Status Command (`yanex status`) - Future

**Purpose**: Show system status and health information
- Repository status, storage usage
- Running experiments
- Recent activity summary

## Technical Considerations

- **Script isolation**: Execute in same process with context patching vs subprocess
- **Signal handling**: Properly handle Ctrl+C and other interrupts
- **Error propagation**: Map script exceptions to experiment status
- **Module patching**: Thread-safe experiment context injection
- **Configuration validation**: Early validation before script execution
- **Reusable components**: Filter and formatting logic shared across commands
- **Performance**: Efficient experiment loading and filtering for large datasets

## Future Extensions

- Support for parameter sweeps (`--sweep param1=1,2,3`)
- Integration with external config systems
- Advanced logging and progress tracking
- Parallel execution support
- Experiment comparison and diff tools
- Export capabilities (CSV, JSON)
- Experiment deletion and archiving