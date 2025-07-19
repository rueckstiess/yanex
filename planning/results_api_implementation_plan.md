# Results API Implementation Plan

## Overview

This document outlines the comprehensive implementation plan for extending Yanex with a unified Results API that provides programmatic access to experiment data. The plan includes both new Python API development and significant CLI refactoring to use a shared filtering system.

## Current State Analysis

### Existing CLI Filtering System

The current CLI has filtering capabilities in `yanex/cli/filters/base.py` with the following characteristics:

- **ExperimentFilter class** handles filtering by status, name patterns, tags, and time ranges
- **Mutual exclusion** between experiment identifiers and filter arguments
- **AND logic** for multiple filter criteria
- **Tag filtering** requires ALL specified tags (AND logic)
- **Default limits** (10 experiments) unless explicitly overridden

### Current CLI Commands with Filtering

1. **`yanex list`** - Full filtering support
2. **`yanex archive`** - Full filtering support  
3. **`yanex compare`** - Full filtering support
4. **`yanex delete`** - Identifier-based and filter-based
5. **`yanex show`** - Single experiment by ID/name
6. **`yanex unarchive`** - Identifier-based only
7. **`yanex update`** - Identifier-based only

### Inconsistencies to Address

1. **Mixed argument handling**: Some commands support both identifiers and filters (with mutual exclusion), others only support one approach
2. **Inconsistent CLI interfaces**: Different commands have different argument patterns
3. **No ID+filter combination**: Current system treats IDs and filters as mutually exclusive
4. **No shared filtering logic**: Each command implements its own validation

## Goals

### Primary Goals

1. **Unified Filtering System**: Single filtering component used by both CLI and Python API
2. **Enhanced CLI Consistency**: All multi-experiment commands use same argument structure
3. **New Python API**: Intuitive results access with pandas DataFrame integration
4. **Flexible Filtering**: Support both IDs and filter criteria with AND/OR logic
5. **Zero Breaking Changes**: Since tool isn't in production use yet

### Secondary Goals

1. **Comprehensive Testing**: All new functionality thoroughly tested
2. **Performance Optimization**: Efficient filtering for large experiment sets
3. **Type Safety**: Full type hints throughout
4. **Documentation**: Complete API reference and examples

## New Filtering Logic Design

### Unified Filter Arguments

All multi-experiment operations will support these arguments:

```python
# Core filtering arguments
ids: list[str] = None                    # OR logic: match any of these IDs
status: str | list[str] = None           # OR logic: match any of these statuses  
name_pattern: str = None                 # Glob pattern matching
tags: list[str] = None                   # AND logic: must have ALL these tags

# Time filtering arguments
started_after: str | datetime = None     # Started >= this time
started_before: str | datetime = None    # Started <= this time
ended_after: str | datetime = None       # Ended >= this time  
ended_before: str | datetime = None      # Ended <= this time

# Boolean filtering arguments
archived: bool = None                    # True/False/None (both)

# Result control arguments  
limit: int = None                        # Limit number of results
```

### Filter Combination Logic

#### Between Filter Types: AND Logic
All specified filter types must be satisfied:
```python
# This finds experiments that match ALL criteria:
# - ID is abc123 OR def456 OR ghi789 AND
# - Status is completed OR failed AND  
# - Has both "training" AND "cnn" tags AND
# - Started after 2024-01-01 AND
# - Not archived
filter_experiments(
    ids=["abc123", "def456", "ghi789"],
    status=["completed", "failed"], 
    tags=["training", "cnn"],
    started_after="2024-01-01",
    archived=False
)
```

#### Within List-Based Filters: OR Logic
```python
# IDs: OR logic
ids=["abc123", "def456"] → id == "abc123" OR id == "def456"

# Status: OR logic  
status=["completed", "failed"] → status == "completed" OR status == "failed"
```

#### Tags: AND Logic (Special Case)
```python
# Tags: AND logic (experiments must have ALL specified tags)
tags=["training", "cnn"] → "training" IN tags AND "cnn" IN tags
```

## Implementation Plan

### Phase 1: Core Infrastructure Refactoring

#### 1.1 Enhanced Universal Filter Component

**File**: `yanex/core/filtering.py` (new)

- **UnifiedExperimentFilter class**: Enhanced version of current ExperimentFilter
- **Support for new filter logic**: AND/OR combinations as specified
- **ID filtering integration**: IDs treated as another filter type, not mutually exclusive
- **Performance optimizations**: Efficient filtering for large datasets
- **Type safety**: Full type hints and validation

```python
class UnifiedExperimentFilter:
    def filter_experiments(
        self,
        ids: list[str] = None,
        status: str | list[str] = None,
        name_pattern: str = None,
        tags: list[str] = None,
        started_after: str | datetime = None,
        started_before: str | datetime = None,
        ended_after: str | datetime = None,
        ended_before: str | datetime = None,
        archived: bool = None,
        limit: int = None,
        sort_by: str = "created_at",
        sort_desc: bool = True
    ) -> list[dict[str, Any]]:
        """Unified filtering with AND/OR logic as specified."""
```

#### 1.2 Filter Logic Implementation

**File**: `yanex/core/filtering.py`

- **Validate and normalize inputs**: Handle string vs list inputs for status
- **ID filtering**: OR logic within ID list  
- **Status filtering**: OR logic within status list
- **Tag filtering**: AND logic (must have ALL tags)
- **Time filtering**: Range-based filtering with proper datetime parsing
- **Combination logic**: AND between different filter types

#### 1.3 CLI Argument Standardization

**File**: `yanex/cli/filters/arguments.py` (new)

- **Common argument decorators**: Shared Click options for all commands
- **Consistent naming**: Standardize argument names across commands
- **Validation helpers**: Shared validation logic
- **Error messages**: Consistent error reporting

```python
def experiment_filter_options(include_ids=True, include_archived=True):
    """Decorator factory for adding standard filter options to CLI commands."""
    
def validate_filter_arguments(**kwargs):
    """Validate and normalize filter arguments."""
```

### Phase 2: CLI Commands Refactoring

#### 2.1 Update All Multi-Experiment Commands

**Commands to update**:
- `yanex list` ✓ (minimal changes - already well-structured)
- `yanex archive` ✓ (add ID support) 
- `yanex unarchive` (add filter support)
- `yanex delete` (enhance with new logic)
- `yanex compare` ✓ (minimal changes)
- `yanex update` (add filter support)

#### 2.2 New CLI Argument Structure

**Before** (current mutually exclusive approach):
```bash
# Either identifiers OR filters
yanex archive exp1 exp2 exp3
yanex archive --status failed --tag old
```

**After** (unified approach):
```bash  
# IDs only
yanex archive --ids exp1 exp2 exp3

# Filters only  
yanex archive --status failed --tag old

# Combined (ID AND filter criteria)
yanex archive --ids exp1 exp2 exp3 --status completed

# Multiple statuses  
yanex archive --status completed failed --tag old
```

#### 2.3 Backwards Compatibility Strategy

Since the tool isn't in production use:
- **No backwards compatibility required**
- **Clean slate approach**: Implement the best possible interface
- **Consistent behavior**: All commands work the same way

### Phase 3: Python Results API Implementation

#### 3.1 Core API Structure

**File**: `yanex/results/__init__.py`

Module-level convenience functions that delegate to underlying classes:

```python
# Individual experiment access
def get_experiment(experiment_id: str) -> Experiment
def get_latest(**filters) -> Experiment | None  
def get_best(metric: str, maximize: bool = True, **filters) -> Experiment | None

# Multiple experiment access (unified filter arguments)
def find(**filters) -> list[dict]
def get_experiments(**filters) -> list[Experiment]
def list_experiments(limit: int = 10, **filters) -> list[dict]

# Comparison and DataFrames
def compare(params=None, metrics=None, only_different=False, **filters) -> pd.DataFrame

# Bulk operations
def archive_experiments(**filters) -> None
def export_experiments(path: str, **filters) -> None
```

#### 3.2 Experiment Class

**File**: `yanex/results/experiment.py`

Individual experiment access and manipulation:

```python
class Experiment:
    def __init__(self, experiment_id: str, manager: ExperimentManager = None)
    
    # Properties (read-only)
    @property
    def id(self) -> str
    @property  
    def name(self) -> str | None
    @property
    def status(self) -> str
    @property
    def tags(self) -> list[str]
    @property
    def started_at(self) -> datetime | None
    @property
    def completed_at(self) -> datetime | None
    @property 
    def duration(self) -> timedelta | None
    
    # Data access methods
    def get_params(self) -> dict[str, Any]
    def get_param(self, key: str, default=None) -> Any
    def get_metrics(self) -> dict[str, Any]  
    def get_metric(self, key: str, default=None) -> Any
    def get_artifacts(self) -> list[Path]
    def get_executions(self) -> list[dict]
    
    # Metadata update methods
    def set_name(self, name: str) -> None
    def set_description(self, description: str) -> None
    def add_tags(self, tags: list[str]) -> None
    def remove_tags(self, tags: list[str]) -> None
    def set_status(self, status: str) -> None
    
    # Utility methods
    def to_dict(self) -> dict[str, Any]
    def refresh(self) -> None
```

#### 3.3 Results Manager

**File**: `yanex/results/manager.py`

Backend class that handles the heavy lifting:

```python
class ResultsManager:
    def __init__(self, storage_path: Path = None)
    
    def find(self, **filters) -> list[dict[str, Any]]
    def get_experiment(self, experiment_id: str) -> Experiment
    def get_experiments(self, **filters) -> list[Experiment]
    def compare_experiments(self, params=None, metrics=None, **filters) -> pd.DataFrame
    def archive_experiments(self, **filters) -> int  # Returns count
    def export_experiments(self, path: str, **filters) -> None
```

#### 3.4 DataFrame Integration

**File**: `yanex/results/dataframe.py`

pandas DataFrame conversion and utilities:

```python
def experiments_to_dataframe(
    experiments_data: list[dict],
    params: list[str] = None,
    metrics: list[str] = None,
    only_different: bool = False
) -> pd.DataFrame:
    """Convert experiment data to pandas DataFrame with hierarchical columns."""

def format_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame for analysis (proper dtypes, index, etc.)."""
```

**DataFrame Structure**:
```python
# Hierarchical columns: (category, name)
df.columns = [
    ('meta', 'id'),
    ('meta', 'name'), 
    ('meta', 'status'),
    ('meta', 'started_at'),
    ('param', 'learning_rate'),
    ('param', 'epochs'),
    ('metric', 'accuracy'),
    ('metric', 'loss')
]

# Easy access patterns:
df[('param', 'learning_rate')]     # Parameter column
df[('metric', 'accuracy')]         # Metric column  
df.xs('param', axis=1, level=0)    # All parameters
df.xs('metric', axis=1, level=0)   # All metrics
```

### Phase 4: Testing Strategy

#### 4.1 Unit Tests

**New test files**:
- `tests/core/test_unified_filtering.py`
- `tests/results/test_experiment.py`
- `tests/results/test_manager.py`
- `tests/results/test_dataframe.py`

**Updated test files**:
- `tests/cli/commands/test_*.py` (all multi-experiment commands)

#### 4.2 Integration Tests

**File**: `tests/integration/test_results_api.py`

- **End-to-end workflows**: CLI → Python API consistency
- **Cross-system tests**: Ensure CLI and Python API produce same results
- **Large dataset tests**: Performance with many experiments
- **Error handling**: Comprehensive error condition testing

#### 4.3 CLI Behavior Testing

**File**: `tests/cli/test_unified_filtering.py`

- **Argument parsing**: All filter combinations work correctly
- **Filter logic**: AND/OR behavior as specified
- **Error messages**: Helpful validation errors
- **Backwards compatibility**: Ensure no regressions where applicable

#### 4.4 Test Data Strategy

**Enhanced test fixtures**:
- **Large experiment datasets**: For performance testing
- **Diverse experiment data**: Various statuses, tags, date ranges
- **Edge cases**: Empty tags, missing metadata, etc.

### Phase 5: Documentation and Examples

#### 5.1 API Documentation

**File**: `docs/results_api.md` (new)

- **Complete API reference**: All methods with examples
- **Usage patterns**: Common workflows and best practices
- **DataFrame guide**: Working with comparison data
- **Performance tips**: Optimizing for large datasets

#### 5.2 CLI Documentation Updates

**Files**: Update existing CLI documentation

- **New argument structure**: Document unified filtering approach  
- **Migration guide**: Changes from previous behavior
- **Examples**: Comprehensive usage examples

#### 5.3 Jupyter Notebook Examples

**File**: `examples/results_analysis.ipynb` (new)

- **Basic usage**: Getting started with the Results API
- **Data analysis workflows**: Comparing experiments, visualization
- **Advanced filtering**: Complex query examples
- **Integration examples**: Using with matplotlib, seaborn, plotly

## Implementation Timeline

### Week 1: Core Infrastructure
- ✅ Analyze current filtering system
- ✅ Design new unified filtering logic
- ✅ Implement `yanex/core/filtering.py`
- ✅ Create CLI argument standardization
- ✅ Write comprehensive tests for new filtering

### Week 2: CLI Refactoring  
- 🔲 Update all multi-experiment CLI commands
- 🔲 Implement new argument parsing
- 🔲 Update CLI error handling and validation
- 🔲 Test all CLI commands with new filtering

### Week 3: Python Results API
- ✅ Implement Experiment class
- ✅ Implement ResultsManager class  
- ✅ Create module-level convenience functions
- ✅ Add DataFrame integration
- ✅ Write comprehensive API tests

### Week 4: Integration and Polish
- 🔲 Integration testing (CLI ↔ Python API)
- 🔲 Performance optimization
- 🔲 Documentation and examples
- 🔲 Final testing and validation

## Detailed File Changes

### New Files

```
yanex/
  core/
    filtering.py                 # Unified filtering system
  results/
    __init__.py                 # Module-level convenience functions
    experiment.py               # Experiment class
    manager.py                  # ResultsManager class
    dataframe.py               # DataFrame integration
    utils.py                   # Helper functions
  cli/
    filters/
      arguments.py             # Standard CLI argument decorators
      
tests/
  core/
    test_unified_filtering.py  # Filtering system tests
  results/
    test_experiment.py         # Experiment class tests
    test_manager.py           # ResultsManager tests  
    test_dataframe.py         # DataFrame tests
  integration/
    test_results_api.py       # End-to-end tests
  cli/
    test_unified_filtering.py # CLI filtering tests
    
planning/
  results_api_implementation_plan.md  # This document
  
examples/
  results_analysis.ipynb      # Jupyter examples
  
docs/
  results_api.md             # API documentation
```

### Modified Files

```
yanex/cli/commands/
  list.py                    # Minimal changes
  archive.py                 # Add ID filtering support
  unarchive.py              # Add filter support  
  delete.py                 # Enhance filtering
  compare.py                # Minimal changes
  update.py                 # Add filter support
  
yanex/cli/filters/
  base.py                   # Deprecated in favor of core/filtering.py
  
yanex/__init__.py           # Add results module to exports
  
tests/cli/commands/
  test_*.py                 # Update all CLI command tests
```

## Risk Assessment

### Technical Risks

1. **Performance Impact**: Large experiment datasets may slow filtering
   - **Mitigation**: Implement efficient indexing and caching
   - **Testing**: Performance benchmarks with large datasets

2. **pandas Dependency**: Adding pandas as dependency increases package size
   - **Mitigation**: Make pandas optional with graceful degradation
   - **Testing**: Ensure API works without pandas installed

3. **CLI Breaking Changes**: New argument structure changes user experience
   - **Mitigation**: Clear migration documentation and error messages
   - **Testing**: Comprehensive CLI testing

### Implementation Risks

1. **Complexity**: Unified filtering system is complex to implement correctly
   - **Mitigation**: Thorough unit testing and incremental implementation
   - **Testing**: Edge case testing and integration tests

2. **Backwards Compatibility**: Ensuring no regressions in existing functionality
   - **Mitigation**: Comprehensive test coverage of existing behavior
   - **Testing**: Before/after behavior comparison tests

## Success Criteria

### Functional Success Criteria

1. **✅ CLI Consistency**: All multi-experiment commands use same argument structure
2. **✅ Filter Logic**: AND/OR combinations work as specified in all contexts
3. **✅ Python API**: Complete results access API with DataFrame integration
4. **✅ Performance**: No significant performance degradation for existing use cases
5. **✅ Test Coverage**: >90% test coverage for all new code

### User Experience Success Criteria

1. **✅ Intuitive Interface**: New users can quickly learn and use the API
2. **✅ Flexible Filtering**: Complex queries can be expressed naturally  
3. **✅ Error Messages**: Clear, helpful error messages for invalid usage
4. **✅ Documentation**: Complete documentation with practical examples

## Future Enhancements

### Phase 2 Enhancements (Post-Implementation)

1. **Advanced Filtering**: 
   - Regex support for name patterns
   - Numeric range filtering for metrics
   - Custom filter functions

2. **Performance Optimizations**:
   - Experiment metadata indexing
   - Lazy loading for large datasets
   - Caching frequently accessed data

3. **Additional Export Formats**:
   - Excel export with multiple sheets
   - JSON export with rich metadata
   - Database export support

4. **Visualization Integration**:
   - Built-in plotting functions
   - Dashboard generation
   - Interactive experiment browsers

## Conclusion

This implementation plan provides a comprehensive roadmap for creating a unified, powerful, and user-friendly Results API for Yanex. The plan addresses both immediate needs (programmatic access to experiment data) and long-term goals (consistent, flexible interface across CLI and Python API).

The key innovations are:

1. **Unified Filtering System**: Single source of truth for experiment filtering logic
2. **Flexible Filter Combinations**: Natural AND/OR logic that matches user expectations  
3. **Consistent Interface**: Same filtering capabilities across CLI and Python API
4. **Progressive Complexity**: Simple cases are simple, complex cases are possible

By following this plan, we'll create a robust foundation for experiment analysis and comparison that will scale with user needs and maintain consistency across all interfaces.