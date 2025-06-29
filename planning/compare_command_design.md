# yanex compare Command Design

## Overview

The `yanex compare` command provides an interactive terminal-based table for comparing experiments, their parameters, and metrics. Similar to `guild compare`, it allows users to sort by any column and navigate through experiment data efficiently.

## Technology Choice

**Python Textual with DataTable widget**
- Built-in interactive sorting and navigation
- Rich text rendering for beautiful terminal UI
- Keyboard shortcuts similar to Guild Compare
- Efficient handling of large datasets
- Mature library with comprehensive documentation

## Data Structure

### Experiment Data Sources
1. **Metadata** (`metadata.json`): id, name, description, tags, status, start/end times
2. **Configuration** (`config.yaml`): parameters like learning_rate, epochs, model_type, dataset
3. **Results** (`results.json`): logged metrics like accuracy, loss, final_accuracy, final_loss

### Column Structure

**Fixed Columns** (always present):
- `run` - Experiment ID (8-char hex)
- `operation` - Script name/path
- `started` - Start timestamp (formatted)
- `time` - Duration (formatted as HH:MM:SS or "-" if still running)
- `status` - created/running/completed/failed/cancelled
- `label` - Experiment name (or "[unnamed]" if not set)

**Dynamic Columns** (auto-discovered or specified):
- **Parameters**: All unique config keys across selected experiments
- **Metrics**: All unique result keys across selected experiments

### Missing Value Handling
- Use `"-"` symbol for experiments that don't have a specific parameter or metric
- Ensure consistent column ordering across all experiments
- Sort missing values consistently (treat "-" as empty/null for sorting purposes)

## Command Interface

```bash
yanex compare [EXPERIMENT_IDENTIFIERS] [FILTER_OPTIONS] [COLUMN_OPTIONS] [TABLE_OPTIONS]

# Filter Options (reuse existing yanex filter system):
--status STATUS               # Filter by experiment status
--name PATTERN               # Filter by name pattern (glob)
--tag TAG                    # Filter by tags (multiple allowed)
--started-after TIMESPEC     # Filter by start time
--started-before TIMESPEC    # Filter by start time
--ended-after TIMESPEC       # Filter by end time
--ended-before TIMESPEC      # Filter by end time
--archived                   # Include archived experiments

# Column Selection Options (new):
--params PARAM1,PARAM2,...   # Show only specified parameters
--metrics METRIC1,METRIC2,... # Show only specified metrics
--all-columns               # Show all available columns (default: auto-discover)
--only-different           # Show only columns where values differ between experiments

# Table Options (new):
--export PATH               # Export table data to CSV file
--no-interactive           # Print static table instead of interactive view
--max-rows N               # Limit number of rows displayed
```

### Examples

```bash
# Interactive comparison of all experiments
yanex compare

# Compare specific experiments
yanex compare exp1 exp2 exp3

# Compare completed training experiments with specific metrics
yanex compare --status completed --tag training --metrics accuracy,loss,final_accuracy

# Compare experiments with specific parameters and export to CSV
yanex compare --params learning_rate,epochs,batch_size --export training_comparison.csv

# Compare recent experiments with all data
yanex compare --started-after "1 week ago" --all-columns

# Show only parameters/metrics that differ between experiments
yanex compare --only-different

# Focus on differences in completed training runs
yanex compare --status completed --tag training --only-different

# Combine with specific column selection (show only different values among specified columns)
yanex compare --params learning_rate,epochs,batch_size --only-different

# Static table output for scripting
yanex compare --status completed --no-interactive --export results.csv
```

## Interactive Features

### Keyboard Navigation (Guild Compare compatible)
- `↑/↓` or `j/k` - Navigate rows
- `←/→` or `h/l` - Navigate columns
- `Home/End` - Jump to first/last row
- `PageUp/PageDown` - Navigate by page

### Sorting
- `s` - Sort ascending by current column
- `S` - Sort descending by current column  
- `1` - Numerical sort ascending
- `2` - Numerical sort descending
- `r` - Reset to original order
- `R` - Reverse current sort order

### Other Controls
- `?` or `F1` - Show help
- `q` or `Ctrl+C` - Quit
- `e` - Export current view to CSV
- `/` - Search/filter within current view (future enhancement)

## Data Extraction Strategy

### 1. Experiment Discovery
- Use existing `ExperimentFilter` to find matching experiments
- Support both identifier-based and filter-based selection
- Include archived experiments if `--archived` flag is used

### 2. Data Collection
For each experiment:
1. Load `metadata.json` for basic experiment info
2. Load `config.yaml` for parameters (handle missing files gracefully)
3. Load `results.json` for metrics (handle missing files gracefully)
4. Extract script path from metadata or config

### 3. Column Schema Discovery
- **Auto-discovery mode** (default): Collect all unique parameter and metric keys across all experiments
- **Selective mode**: Use only keys specified in `--params` and `--metrics`
- **Difference filtering**: When `--only-different` is used, filter out columns where all values are identical
- Create unified column schema with consistent ordering

### 4. Data Matrix Construction
- Build table with experiments as rows
- Fill in values for each experiment, using `"-"` for missing values
- Apply proper data types for sorting (numeric vs string)

### 5. Column Filtering Logic (--only-different)
- After building the complete data matrix, analyze each parameter/metric column
- Identify columns where all non-missing values are identical
- Handle edge cases:
  - Columns with all missing values (`"-"`) - keep or exclude based on policy
  - Columns with only one non-missing value - consider as "not different"
  - Mixed missing/non-missing with same non-missing value - consider as "not different"
- Remove invariant columns from the final display
- Combine with `--params`/`--metrics` selection (apply difference filtering to selected columns)

## Implementation Plan

### Phase 1: Data Layer (`yanex/core/comparison.py`)
1. Create `ExperimentComparisonData` class
2. Implement experiment data collection methods
3. Add column schema discovery (auto and selective)
4. Build data matrix with missing value handling
5. Add data type inference for proper sorting
6. Implement column difference filtering logic (`--only-different`)

### Phase 2: UI Layer (`yanex/ui/compare_table.py`)
1. Create Textual DataTable application
2. Implement keyboard navigation handlers
3. Add sorting functionality with proper type handling
4. Create help dialog/overlay
5. Add export functionality from UI

### Phase 3: CLI Layer (`yanex/cli/commands/compare.py`)
1. Create compare command with all CLI options
2. Integrate with existing filter system
3. Add parameter/metric selection logic
4. Implement static table output mode
5. Add CSV export functionality

### Phase 4: Integration & Testing
1. Register command in main CLI
2. Create comprehensive test suite
3. Add integration tests with sample data
4. Test with various experiment configurations
5. Performance testing with large datasets

## File Structure

```
yanex/
├── core/
│   └── comparison.py          # Data extraction and processing
├── ui/
│   └── compare_table.py       # Interactive Textual DataTable app
├── cli/
│   └── commands/
│       └── compare.py         # CLI command implementation
└── tests/
    ├── core/
    │   └── test_comparison.py
    ├── ui/
    │   └── test_compare_table.py
    └── cli/
        └── test_compare_command.py
```

## Dependencies

### New Dependencies (to be added to requirements)
- `textual` - Terminal UI framework
- `textual[dev]` - Development tools (optional)

### Existing Dependencies (already available)
- `rich` - Used by Textual for rendering
- `click` - CLI framework
- `pyyaml` - Config file parsing
- Standard library: `json`, `pathlib`, `datetime`, etc.

## Data Types and Sorting

### Column Type Inference
- **Numeric**: Detect integers and floats for proper numerical sorting
- **Datetime**: Detect ISO datetime strings for chronological sorting  
- **String**: Default fallback for text data
- **Missing**: Handle `"-"` values consistently in sorting

### Sort Order Priority
1. Non-missing values first (sorted by type)
2. Missing values (`"-"`) last
3. Maintain stable sort for equal values

## Error Handling

### Graceful Degradation
- Continue processing if individual experiment files are corrupted
- Show warning messages for failed experiment loads
- Provide partial results rather than complete failure

### User Feedback
- Progress indication for large experiment sets
- Clear error messages for invalid parameters/metrics
- Helpful suggestions for typos in column names

## Future Enhancements

### Phase 2 Features (post-MVP)
- Search/filter within the table view
- Column width auto-adjustment
- Save/load comparison configurations
- Integration with plotting libraries for quick visualizations
- Diff mode to highlight differences between experiments
- Grouping by common parameter values

### Performance Optimizations
- Lazy loading for very large experiment sets
- Streaming data processing
- Caching of processed comparison data
- Background data loading with progress indicators

## Testing Strategy

### Unit Tests
- Data extraction accuracy
- Column schema discovery
- Missing value handling
- Type inference and sorting

### Integration Tests  
- Full command execution with sample experiments
- CSV export functionality
- Filter integration
- Interactive UI navigation (automated where possible)

### Performance Tests
- Large experiment set handling
- Memory usage profiling
- UI responsiveness with large tables

This design provides a comprehensive, user-friendly experiment comparison tool that integrates seamlessly with yanex's existing architecture while providing Guild Compare-like functionality.