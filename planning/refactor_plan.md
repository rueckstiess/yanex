# Comprehensive Yanex Refactoring Plan

## Executive Summary

Based on thorough analysis of the yanex codebase, I've identified significant opportunities for improvement across code quality, maintainability, and testability. The codebase shows good architectural foundations but suffers from code duplication, complex monolithic functions, and inconsistent patterns.

## High-Priority Refactoring Targets

### 🔴 **1. Script Execution Duplication (Critical)**
**Location**: `yanex/cli/commands/run.py:194-318, 521-625`
**Impact**: 131 lines of duplicated subprocess execution logic
**Effort**: Medium (2-3 days)
**Solution**: Extract `ScriptExecutor` class to `yanex/core/script_executor.py`

**Proposed Implementation**:
```python
# New file: yanex/core/script_executor.py
class ScriptExecutor:
    def __init__(self, manager: ExperimentManager):
        self.manager = manager
    
    def execute_script(
        self, 
        experiment_id: str,
        script_path: Path, 
        config: Dict[str, Any],
        verbose: bool = False
    ) -> None:
        """Centralized script execution logic"""
        # Unified implementation of subprocess execution
        pass
```

### 🔴 **2. Configuration Parsing Complexity (Critical)**
**Location**: `yanex/core/config.py:115-304` (189-line function)
**Impact**: Single responsibility violation, difficult to test/maintain
**Effort**: High (1 week)
**Solution**: Implement strategy pattern with `ParameterParser` classes

**Proposed Implementation**:
```python
# Refactored config.py
class ParameterParser:
    def __init__(self):
        self.type_parsers = {
            'sweep': SweepParameterParser(),
            'basic': BasicParameterParser(),
            'list': ListParameterParser()
        }
    
    def parse(self, value_str: str) -> Any:
        for parser_type, parser in self.type_parsers.items():
            if parser.can_parse(value_str):
                return parser.parse(value_str)
        return value_str
```

### 🔴 **3. Storage Layer Monolith (Critical)**
**Location**: `yanex/core/storage.py` (683 lines)
**Impact**: Too many responsibilities, poor separation of concerns
**Effort**: High (1-2 weeks)
**Solution**: Extract storage interface and implementations

**Proposed Implementation**:
```python
# yanex/core/storage/base.py
class ExperimentStorageInterface:
    """Abstract storage interface for experiments"""
    
# yanex/core/storage/filesystem.py
class FileSystemStorage(ExperimentStorageInterface):
    """File system implementation"""
    
# yanex/core/storage/factory.py
class StorageFactory:
    @staticmethod
    def create_storage(storage_type: str = "filesystem") -> ExperimentStorageInterface:
        pass
```

## Medium-Priority Refactoring Targets

### 🟡 **4. DateTime Parsing Duplication**
**Locations**: 
- `yanex/cli/filters/base.py:206-217, 230-242, 254-263, 276-284`
- `yanex/cli/formatters/console.py:172-197`
- `yanex/core/comparison.py:272-275, 306-307`

**Impact**: Maintenance burden, potential inconsistencies
**Effort**: Low (1 day)
**Solution**: Create `yanex/utils/datetime_utils.py`

**Proposed Implementation**:
```python
# yanex/utils/datetime_utils.py
class DateTimeParser:
    @staticmethod
    def parse_iso_datetime(dt_str: str) -> datetime:
        """Centralized ISO datetime parsing with timezone handling"""
        if dt_str.endswith("Z"):
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        elif "+" in dt_str:
            return datetime.fromisoformat(dt_str)
        else:
            from datetime import timezone
            return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
```

### 🟡 **5. CLI Error Handling Inconsistency**
**Locations**: 
- `yanex/cli/commands/archive.py:174-177`
- `yanex/cli/commands/compare.py:225-229`
- `yanex/cli/commands/list.py:215-221`

**Impact**: Poor user experience, debugging difficulties
**Effort**: Medium (2-3 days)
**Solution**: Implement `CLIErrorHandler` class

**Proposed Implementation**:
```python
# yanex/cli/error_handling.py
class CLIErrorHandler:
    @staticmethod
    def handle_command_error(ctx: click.Context, error: Exception, verbose: bool = False):
        """Standardized error handling for CLI commands"""
        if isinstance(error, click.ClickException):
            raise
        elif isinstance(error, YanexError):
            click.echo(f"Error: {error}", err=True)
            ctx.exit(1)
        else:
            click.echo(f"Unexpected error: {error}", err=True)
            if verbose:
                import traceback
                click.echo(traceback.format_exc(), err=True)
            ctx.exit(1)
```

### 🟡 **6. Test Infrastructure Duplication**
**Locations**: 303 occurrences of temp directory setup, 15 test experiment helpers
**Impact**: Test maintenance overhead, brittle tests
**Effort**: Medium (3-4 days)
**Solution**: Shared fixtures and test helpers

**Proposed Implementation**:
```python
# tests/conftest.py additions
@pytest.fixture
def experiment_manager(temp_dir):
    """Shared ExperimentManager with temp directory."""
    
@pytest.fixture  
def sample_experiment_data():
    """Standardized test experiment data."""

# tests/helpers.py
class ExperimentTestHelper:
    """Centralized experiment creation and management for tests."""
    
    def create_test_experiment(self, exp_id, **kwargs):
        """Standardized test experiment creation."""
```

## Low-Priority Refactoring Targets

### 🟢 **7. Missing Dependency Injection**
**Impact**: Tight coupling, testing difficulties
**Effort**: High (1-2 weeks)
**Solution**: Implement DI container and interfaces

**Proposed Implementation**:
```python
# yanex/core/container.py
class DIContainer:
    def __init__(self):
        self._services = {}
    
    def register(self, interface, implementation):
        self._services[interface] = implementation
    
    def resolve(self, interface):
        return self._services.get(interface)
```

### 🟢 **8. Validation Logic Scattered**
**Locations**: `yanex/cli/_utils.py:51-97`, `yanex/utils/validation.py`, various core modules
**Impact**: Inconsistent validation patterns
**Effort**: Medium (3-4 days)
**Solution**: Centralized validation with validator pattern

### 🟢 **9. Long Parameter Lists**
**Examples**: 
- `ExperimentManager.create_experiment()` (7 parameters)
- `ExperimentFilter.filter_experiments()` (10 parameters)
- `ExperimentComparisonData.get_comparison_data()` (6 parameters)

**Impact**: API usability, maintainability
**Effort**: Medium (4-5 days)
**Solution**: Configuration objects and builder patterns

### 🟢 **10. CLI Command Patterns**
**Issue**: Repetitive filter parsing and experiment discovery across commands
**Examples**: Similar logic in `archive.py`, `compare.py`, `list.py`
**Solution**: Create base command class

**Proposed Implementation**:
```python
# yanex/cli/base_command.py
class FilterableCommand:
    """Base class for commands that filter experiments"""
    
    def parse_filters(self, **kwargs) -> ExperimentFilter:
        """Standardized filter parsing"""
        pass
    
    def find_experiments(self, identifiers, filters) -> List[Dict]:
        """Standardized experiment discovery"""
        pass
```

## Test Structure Issues

### High Duplication Areas

**Temporary Directory Setup (303 occurrences)**:
- Pattern repeated across many files
- Found in: `tests/cli/test_compare.py:22-28`, `tests/core/test_manager.py:89-93`, `tests/test_api.py:89-92`

**Test Experiment Creation (15 occurrences)**:
- Helper method duplicated across files
- Found in: `tests/cli/test_compare.py:36-55`, `tests/core/test_comparison.py:25-46`

### Missing Test Coverage Areas

- Limited end-to-end CLI workflow testing
- Missing cross-component integration scenarios
- No tests for concurrent experiment handling edge cases
- Limited testing of filesystem permission errors
- Missing tests for corrupted data recovery scenarios

### Inconsistent Testing Approaches

- Manager tests heavily mock dependencies
- Storage tests use real filesystem operations
- Mixed approaches to CLI command testing

## Recommended Implementation Timeline

### Phase 1 (Week 1-2): Quick Wins
1. Extract datetime parsing utilities
2. Standardize CLI error handling
3. Create shared test fixtures

### Phase 2 (Week 3-4): Core Refactoring
1. Extract script execution logic
2. Begin configuration parsing refactor
3. Improve test infrastructure

### Phase 3 (Week 5-8): Architectural Improvements
1. Complete configuration parsing refactor
2. Implement storage layer abstraction
3. Add dependency injection framework

### Phase 4 (Week 9-12): Long-term Improvements
1. Implement validation layer
2. Refactor parameter handling
3. Add missing integration tests

## Expected Benefits

### Immediate (Phase 1-2)
- Reduced code duplication by ~30%
- Improved error handling consistency
- Faster test execution and maintenance

### Medium-term (Phase 3-4)
- Better separation of concerns
- Improved testability and coverage
- Enhanced code maintainability

### Long-term
- Increased extensibility
- Better performance
- Reduced technical debt

## Risk Assessment

### High-Risk Changes
- Storage layer refactoring (backward compatibility)
- Configuration parsing changes (API breaking)

### Medium-Risk Changes
- Script execution extraction
- Test infrastructure changes

### Low-Risk Changes
- Utility extractions
- Error handling standardization

## Success Metrics

- **Code Duplication**: Reduce from current ~25% to <10%
- **Test Coverage**: Maintain >90% while improving test quality
- **Cyclomatic Complexity**: Reduce functions >15 complexity by 50%
- **File Size**: No files >500 lines (current: 3 files >600 lines)

## Specific Code Analysis

### Current Architecture Issues

1. **Script Execution Duplication**: Nearly identical subprocess execution logic with 131 duplicated lines including environment variable setup, subprocess execution with threading, output capture and streaming, and error handling patterns.

2. **Complex Configuration Management**: Overly complex parameter parsing with massive function having multiple responsibilities, complex nested logic for sweep parameter parsing, and mixed concerns (parsing, validation, type conversion).

3. **Storage Layer Abstraction Issues**: Direct file system operations mixed with business logic, large class (683 lines) with too many responsibilities, and difficult to test and extend.

4. **DateTime Parsing Duplication**: Repeated timezone-aware datetime parsing logic found across multiple modules with potential for inconsistent behavior.

5. **Missing Factory Patterns**: Complex object creation scattered throughout code including experiment creation logic mixed with business logic, format-specific parsers created inline, and storage implementations hard-coded.

This refactoring plan will significantly improve the codebase's maintainability, testability, and extensibility while preserving existing functionality.