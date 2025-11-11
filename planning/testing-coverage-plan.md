# Yanex Testing Coverage Improvement Plan

**Goal**: Improve test coverage from current ~76% to 90%+ by adding comprehensive tests for critical gaps and quick wins.

**Scope**: Excludes Web UI testing (requires separate E2E testing strategy)

**Timeline**: 6-9 days total across 3 phases

---

## Available Testing Infrastructure

### Fixtures (tests/conftest.py)
- `temp_dir` - Temporary directory creation
- `git_repo` - Git repository with initial commit
- `clean_git_repo` - Clean git repository (no uncommitted changes)
- `sample_config_yaml` - Sample configuration file
- `sample_experiment_script` - Basic yanex experiment script
- `isolated_experiments_dir` - Isolated experiment storage
- `isolated_storage` - ExperimentStorage instance
- `isolated_manager` - ExperimentManager instance
- `cli_runner` - Click CLI test runner

### Test Utilities (tests/test_utils.py)
- **TestDataFactory**: Create standardized test data (metadata, configs, results)
- **TestAssertions**: Domain-specific validation helpers
- **TestFileHelpers**: Script and file creation utilities
- **MockHelpers**: Git and environment mocks

### Common Testing Patterns
1. **CLI Command Tests**: Use `cli_runner.invoke(cli, [...])` with assertions on exit codes and output
2. **Integration Tests**: Create real experiments in clean git repos (like `test_update_command.py`)
3. **Unit Tests**: Mock dependencies and test specific functions
4. **Parametrized Tests**: Use `@pytest.mark.parametrize` for multiple scenarios

---

## Phase 1: Quick Wins (1-2 days, +5-7% coverage)

### 1.1 Delete Command Tests
**File**: Create `tests/cli/commands/test_delete.py`
**Coverage**: 50% → 90% (+40%)
**Impact**: +1-2% overall coverage
**Difficulty**: Easy

**Test Cases**:
```python
class TestDeleteCommand:
    """Test delete command functionality."""

    # Basic operations
    def test_delete_by_experiment_id()
    def test_delete_by_experiment_name()
    def test_delete_multiple_experiments()
    def test_delete_no_experiments_found()

    # Filtering
    def test_delete_with_status_filter()
    def test_delete_with_name_pattern()
    def test_delete_with_tags()
    def test_delete_with_time_filters()

    # Archived experiments
    def test_delete_archived_experiments()
    def test_delete_archived_flag_searches_archive()

    # Confirmation and force
    def test_delete_requires_confirmation()
    def test_delete_force_flag_skips_confirmation()
    def test_delete_additional_warning_for_bulk_operations()

    # Edge cases
    def test_delete_nonexistent_experiment()
    def test_delete_with_filters_no_matches()

class TestDeleteCommandIntegration:
    """Integration tests with real experiments."""

    def test_delete_experiment_permanently(clean_git_repo, sample_experiment_script):
        """Create experiment, delete it, verify it's gone."""

    def test_delete_archived_integration(clean_git_repo, sample_experiment_script):
        """Create, archive, then delete experiment."""
```

**Pattern**: Follow `test_update_command.py` structure with validation tests + integration tests

---

### 1.2 Confirm Module Tests
**File**: Create `tests/cli/commands/test_confirm.py`
**Coverage**: 81% → 95% (+14%)
**Impact**: +1% overall coverage
**Difficulty**: Easy

**Test Cases**:
```python
class TestConfirmExperimentOperation:
    """Test confirmation prompts."""

    def test_confirm_single_experiment()
    def test_confirm_multiple_experiments()
    def test_confirm_with_force_flag_skips_prompt()
    def test_confirm_no_experiments_returns_early()
    def test_confirm_custom_operation_verb()
    def test_confirm_default_yes_option()
    def test_confirm_user_abort_raises_click_abort()

class TestFindExperimentsByIdentifiers:
    """Test experiment resolution by ID/name."""

    def test_find_by_exact_experiment_id()
    def test_find_by_id_prefix_unique()
    def test_find_by_id_prefix_ambiguous_raises()
    def test_find_by_name_exact_match()
    def test_find_by_name_ambiguous_raises()
    def test_find_experiment_not_found_raises()
    def test_find_multiple_identifiers()
    def test_find_archived_experiments()

class TestFindExperimentsByFilters:
    """Test experiment resolution by filters."""

    def test_find_by_status_filter()
    def test_find_by_multiple_filter_criteria()
    def test_find_archived_with_flag()
    def test_find_no_matches_returns_empty()
```

**Pattern**: Unit tests with mocked `ExperimentFilter` and `Click` contexts

---

### 1.3 UI Command Tests
**File**: Create `tests/cli/commands/test_ui.py`
**Coverage**: 35% → 90% (+55%)
**Impact**: +1% overall coverage
**Difficulty**: Easy

**Test Cases**:
```python
class TestUICommand:
    """Test UI server command."""

    # Help and validation
    def test_ui_help_output()
    def test_ui_missing_build_directory_error()

    # Server options
    def test_ui_default_host_and_port()
    def test_ui_custom_host_and_port()
    def test_ui_reload_flag()
    def test_ui_no_browser_flag()
    def test_ui_verbose_output()

    # Server lifecycle
    @patch("yanex.cli.commands.ui.uvicorn.run")
    def test_ui_starts_uvicorn_server(mock_run)

    @patch("yanex.cli.commands.ui.webbrowser.open")
    def test_ui_opens_browser_by_default(mock_open)

    @patch("yanex.cli.commands.ui.uvicorn.run")
    def test_ui_keyboard_interrupt_handled(mock_run)
```

**Pattern**: Mock `uvicorn.run`, `webbrowser.open`, and `Path.exists`

---

### 1.4 Git Utils Expansion
**File**: Expand `tests/core/test_git_utils.py`
**Coverage**: 74% → 90% (+16%)
**Impact**: +1% overall coverage
**Difficulty**: Easy

**Additional Test Cases**:
```python
class TestGetGitRepo:
    """Test get_git_repo function - ADD MISSING CASES."""

    # Existing tests cover basic cases
    # Add these edge cases:
    def test_get_repo_no_git_executable()
    def test_get_repo_corrupted_git_directory()

class TestHasUncommittedChanges:
    """Test has_uncommitted_changes function - ADD ERROR CASES."""

    def test_has_uncommitted_changes_git_error()
    def test_has_uncommitted_changes_detached_head()

class TestGenerateGitPatch:
    """Test generate_git_patch function - ADD ERROR CASES."""

    def test_generate_patch_git_command_fails()
    def test_generate_patch_detached_head()
    def test_generate_patch_submodule_changes()

class TestGetCurrentCommitInfo:
    """Test get_current_commit_info - ADD EDGE CASES."""

    def test_get_commit_info_shallow_clone()
    def test_get_commit_info_new_repo_no_commits()
```

**Pattern**: Expand existing test file with edge cases and error conditions

---

### 1.5 Filter Arguments Tests
**File**: Create `tests/cli/filters/test_arguments.py`
**Coverage**: 29% → 85% (+56%)
**Impact**: +2-3% overall coverage
**Difficulty**: Easy-Medium

**Test Cases**:
```python
class TestExperimentFilterOptions:
    """Test filter options decorator."""

    def test_decorator_adds_all_filter_options()
    def test_decorator_with_include_ids_false()
    def test_decorator_with_include_archived_false()
    def test_decorator_with_include_limit_false()
    def test_decorator_with_custom_default_limit()
    def test_decorated_command_has_correct_params()

class TestValidateFilterArguments:
    """Test filter argument validation and normalization."""

    def test_validate_normalizes_experiment_ids()
    def test_validate_normalizes_status_values()
    def test_validate_normalizes_name_pattern()
    def test_validate_normalizes_tags()
    def test_validate_normalizes_time_filters()
    def test_validate_script_pattern_with_path_separators_warns()
    def test_validate_removes_none_and_empty_values()
    def test_validate_handles_all_none_input()

class TestRequireFiltersOrConfirmation:
    """Test confirmation requirement logic."""

    def test_with_meaningful_filters_no_confirmation()
    def test_without_filters_requires_confirmation()
    def test_force_flag_skips_confirmation()
    def test_limit_not_counted_as_meaningful_filter()
    def test_archived_flag_not_counted_as_filter()

class TestFormatFilterSummary:
    """Test filter summary formatting for display."""

    def test_format_empty_filters()
    def test_format_single_filter()
    def test_format_multiple_filters()
    def test_format_ids_with_truncation_over_5()
    def test_format_all_filter_types_together()
    def test_format_time_filters()

class TestParseCliTimeFilters:
    """Test CLI time filter parsing."""

    @pytest.mark.parametrize("time_string,expected", [
        ("2024-01-01", datetime(...)),
        ("yesterday", ...),
        ("1 week ago", ...),
    ])
    def test_parse_valid_time_strings(time_string, expected)

    def test_parse_none_values_returns_none()
    def test_parse_invalid_time_raises_bad_parameter()
    def test_parse_all_none_returns_empty_dict()
```

**Pattern**: Unit tests with parametrize for multiple input variations

---

## Phase 2: Medium Impact Commands (3-4 days, +10-12% coverage)

### 2.1 List Command Tests (HIGH PRIORITY)
**File**: Create `tests/cli/commands/test_list.py`
**Coverage**: 16% → 90% (+74%)
**Impact**: +3-4% overall coverage
**Difficulty**: Medium

**Test Cases**:
```python
class TestListCommand:
    """Test list command functionality."""

    # Basic listing
    def test_list_help_output()
    def test_list_default_shows_last_10_experiments()
    def test_list_all_flag_shows_all_experiments()
    def test_list_with_custom_limit()
    def test_list_no_experiments_found()
    def test_list_empty_experiments_directory()

    # Filtering
    def test_list_filter_by_status()
    def test_list_filter_by_name_pattern()
    def test_list_filter_by_single_tag()
    def test_list_filter_by_multiple_tags()
    def test_list_filter_by_script_pattern()
    def test_list_filter_by_time_ranges()
    def test_list_complex_filter_combinations()
    def test_list_filters_no_matches()

    # Archived experiments
    def test_list_archived_experiments()
    def test_list_archived_with_filters()
    def test_list_archived_applies_correct_limit()
    def test_list_archived_empty()

    # Verbose mode
    def test_list_verbose_shows_filter_summary()
    def test_list_verbose_shows_total_count()
    def test_list_verbose_with_filters_shows_matched_count()

    # Output formatting
    def test_list_output_includes_columns()
    def test_list_output_formats_duration()
    def test_list_output_handles_missing_names()

    # Edge cases
    def test_list_invalid_time_filter()
    def test_list_shows_filter_suggestions_on_no_matches()

class TestListCommandIntegration:
    """Integration tests with real experiments."""

    def test_list_multiple_experiments(clean_git_repo, sample_experiment_script):
        """Create 15 experiments, verify list shows last 10 by default."""

    def test_list_with_name_filtering(clean_git_repo, sample_experiment_script):
        """Create experiments with different names, test pattern matching."""

    def test_list_archived_integration(clean_git_repo, sample_experiment_script):
        """Create, archive experiments, verify --archived flag works."""

    def test_list_all_flag_integration(clean_git_repo, sample_experiment_script):
        """Create 15 experiments, verify --all shows all."""
```

**Pattern**: Mix of validation tests and integration tests creating real experiments

---

### 2.2 Show Command Expansion (HIGH PRIORITY)
**File**: Expand `tests/cli/test_show.py`
**Coverage**: 50% → 90% (+40%)
**Impact**: +4-5% overall coverage
**Difficulty**: Medium

**Additional Test Cases**:
```python
class TestDisplayExperimentDetails:
    """Test detailed experiment display function."""

    # Basic info display
    def test_display_basic_experiment_info()
    def test_display_with_tags()
    def test_display_with_description()
    def test_display_without_name_shows_unnamed()

    # Configuration display
    def test_display_empty_configuration()
    def test_display_nested_configuration()
    def test_display_large_configuration()

    # Results/Metrics display
    def test_display_results_with_few_metrics()
    def test_display_results_with_many_metrics_truncates()
    def test_display_results_with_requested_metrics_filter()
    def test_display_results_missing_requested_metrics()
    def test_display_no_results()

    # Artifacts display
    def test_display_artifacts_list()
    def test_display_no_artifacts()
    def test_display_git_patch_artifact()

    # Environment and git display
    def test_display_environment_variables()
    def test_display_git_info_clean_state()
    def test_display_git_info_with_uncommitted_changes()
    def test_display_git_info_with_patch_file()

    # Status-specific display
    def test_display_completed_experiment_shows_duration()
    def test_display_running_experiment_shows_elapsed_time()
    def test_display_failed_experiment_shows_error()
    def test_display_cancelled_experiment_shows_reason()

    # Archived experiments
    def test_display_archived_experiment_label()

class TestShowCommandIntegration:
    """Integration tests for show command."""

    def test_show_by_full_id(clean_git_repo, sample_experiment_script):
        """Create experiment, show by full ID."""

    def test_show_by_id_prefix(clean_git_repo, sample_experiment_script):
        """Create experiment, show by ID prefix."""

    def test_show_by_name(clean_git_repo, sample_experiment_script):
        """Create named experiment, show by name."""

    def test_show_with_metrics_filter(clean_git_repo, sample_experiment_script):
        """Create experiment with metrics, show specific metrics."""

    def test_show_archived_experiment(clean_git_repo, sample_experiment_script):
        """Create, archive, then show experiment."""

    def test_show_experiment_with_artifacts(clean_git_repo, sample_experiment_script):
        """Create experiment with artifacts, verify display."""
```

**Pattern**: Expand existing file with display function tests and more integration tests

---

### 2.3 Results/DataFrame API (HIGHEST IMPACT)
**File**: Create `tests/results/test_dataframe.py`
**Coverage**: 0% → 85% (+85%)
**Impact**: +8-10% overall coverage
**Difficulty**: Medium

**Test Cases**:
```python
class TestExperimentsToDataFrame:
    """Test conversion of comparison data to DataFrame."""

    def test_empty_comparison_data_returns_empty_dataframe()
    def test_single_experiment_conversion()
    def test_multiple_experiments_conversion()
    def test_hierarchical_columns_structure()
    def test_experiment_id_as_index()
    def test_parameter_columns_extracted()
    def test_metric_columns_extracted()
    def test_metadata_columns_extracted()
    def test_nested_parameters_flattened()
    def test_pandas_not_installed_raises_import_error()

class TestFormatDataFrameForAnalysis:
    """Test DataFrame optimization for analysis."""

    def test_numeric_conversion_for_parameters()
    def test_numeric_conversion_for_metrics()
    def test_datetime_conversion_for_timestamps()
    def test_duration_to_timedelta_conversion()
    def test_categorical_conversion_for_low_cardinality()
    def test_high_cardinality_remains_object()
    def test_no_pandas_returns_dataframe_unchanged()
    def test_memory_optimization()

class TestFlattenDataFrameColumns:
    """Test column flattening."""

    def test_flatten_two_level_hierarchical_columns()
    def test_meta_columns_without_prefix()
    def test_param_columns_with_prefix()
    def test_metric_columns_with_prefix()
    def test_already_flat_columns_unchanged()
    def test_empty_dataframe_returns_unchanged()

class TestGetParameterSummary:
    """Test parameter summary statistics."""

    def test_numeric_parameter_summary()
    def test_categorical_parameter_summary()
    def test_mixed_parameters_summary()
    def test_empty_parameters_returns_empty()
    def test_non_hierarchical_columns_returns_empty()

class TestGetMetricSummary:
    """Test metric summary statistics."""

    def test_metric_summary_with_multiple_metrics()
    def test_metric_summary_statistics()
    def test_empty_metrics_returns_empty()

class TestCorrelationAnalysis:
    """Test correlation analysis."""

    def test_correlation_matrix_calculation()
    def test_correlation_only_numeric_columns()
    def test_correlation_empty_returns_empty()
    def test_correlation_single_column()
    def test_correlation_with_nan_values()

class TestFindBestExperiments:
    """Test finding best experiments."""

    def test_find_best_maximize_metric()
    def test_find_best_minimize_metric()
    def test_find_best_top_n_selection()
    def test_find_best_missing_metric_raises_value_error()
    def test_find_best_handles_missing_values()
    def test_find_best_all_nan_values()

class TestExportComparisonSummary:
    """Test Excel export."""

    def test_export_creates_excel_file()
    def test_export_multiple_sheets()
    def test_export_overview_sheet()
    def test_export_parameters_sheet()
    def test_export_metrics_sheet()
    def test_export_correlations_sheet()
    def test_openpyxl_not_installed_raises_import_error()
```

**Pattern**:
- Mock pandas/openpyxl imports for error testing
- Use TestDataFactory to create comparison data
- Test with real pandas operations when available
- Parametrize for different data types

---

### 2.4 Storage Archive Tests
**File**: Create `tests/core/test_storage_archive.py`
**Coverage**: 51% → 85% (+34%)
**Impact**: +1-2% overall coverage
**Difficulty**: Medium

**Test Cases**:
```python
class TestFileSystemArchiveStorage:
    """Test archive storage operations."""

    # Archive operations
    def test_archive_experiment_moves_directory()
    def test_archive_updates_metadata()
    def test_archive_already_archived_raises()
    def test_archive_with_custom_archive_directory()
    def test_archive_nonexistent_experiment_raises()

    # Unarchive operations
    def test_unarchive_experiment_moves_back()
    def test_unarchive_restores_to_original_location()
    def test_unarchive_not_found_raises()
    def test_unarchive_already_exists_raises()
    def test_unarchive_with_custom_directory()

    # Delete operations
    def test_delete_regular_experiment()
    def test_delete_archived_experiment()
    def test_delete_removes_directory_completely()
    def test_delete_not_found_raises()

    # List and check operations
    def test_list_archived_experiments()
    def test_list_empty_archive_returns_empty()
    def test_archived_experiment_exists()
    def test_get_archived_experiment_directory()

    # Edge cases
    def test_archive_preserves_all_files()
    def test_concurrent_archive_operations()
```

**Pattern**: Use `isolated_experiments_dir` fixture, create real experiments to archive

---

## Phase 3: Complex Features (2-3 days, +3-5% coverage)

### 3.1 Parallel Executor Tests
**File**: Create `tests/test_executor.py`
**Coverage**: 75% → 95% (+20%)
**Impact**: +2-3% overall coverage
**Difficulty**: Medium-Hard

**Test Cases**:
```python
class TestExperimentSpec:
    """Test ExperimentSpec dataclass."""

    def test_spec_with_script_path_only()
    def test_spec_with_function_only()
    def test_spec_validate_both_script_and_function_raises()
    def test_spec_validate_neither_raises()
    def test_spec_validate_function_not_implemented_yet()
    def test_spec_with_all_optional_fields()

class TestRunMultiple:
    """Test run_multiple function."""

    # Validation
    def test_run_multiple_empty_list_raises()
    def test_run_multiple_invalid_spec_raises()
    def test_run_multiple_negative_parallel_raises()

    # Sequential execution
    def test_sequential_execution_all_success()
    def test_sequential_execution_with_single_failure()
    def test_sequential_execution_continues_after_failure()
    def test_sequential_execution_all_failures()

    # Parallel execution
    def test_parallel_execution_all_success()
    def test_parallel_execution_auto_detect_workers()
    def test_parallel_execution_with_failures()
    def test_parallel_execution_worker_crash()
    def test_parallel_execution_process_pool_error()

    # Results validation
    def test_results_include_all_experiments()
    def test_results_have_correct_status()
    def test_results_include_duration()
    def test_results_include_error_messages_for_failures()
    def test_results_preserve_experiment_order()

    # Verbose mode
    def test_verbose_shows_progress()
    def test_verbose_shows_errors()

class TestExecuteSingleExperiment:
    """Test single experiment execution."""

    def test_execute_success()
    def test_execute_script_failure()
    def test_execute_click_abort_error()
    def test_execute_experiment_creation_failure()
    def test_execute_captures_script_output()
    def test_execute_with_script_args()
```

**Pattern**:
- Use TestFileHelpers to create test scripts
- Mock ProcessPoolExecutor for parallel tests
- Test error recovery and result aggregation
- Integration tests with real script execution

---

## Coverage Target Summary

**Current**: 76%

**Expected Coverage by Phase:**
- **After Phase 1**: 76% → 83% (+7%)
- **After Phase 2**: 83% → 95% (+12%)  ← **90% TARGET ACHIEVED**
- **After Phase 3**: 95% → 98% (+3%)

**Total Expected**: 22% coverage improvement

---

## Implementation Guidelines

### Test File Structure Template

```python
"""
Tests for [module description].
"""

import pytest
from tests.test_utils import (
    TestDataFactory,
    TestFileHelpers,
    TestAssertions,
    create_cli_runner
)
from yanex.cli.main import cli

class Test[FeatureName]:
    """Test [feature] functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()

    def test_[basic_case]():
        """Test [what]."""
        # Arrange
        # Act
        # Assert

    @pytest.mark.parametrize("input,expected", [...])
    def test_[parameterized_case](input, expected):
        """Test [variations]."""
        # Test logic

class Test[FeatureName]Integration:
    """Integration tests for [feature]."""

    def test_[end_to_end](clean_git_repo, sample_experiment_script):
        """Test [complete workflow]."""
        # Integration test with real experiments
```

### Best Practices

1. **Follow Existing Patterns**
   - Study `test_update_command.py` for CLI integration test patterns
   - Use fixtures consistently
   - Parametrize for multiple scenarios

2. **Use Available Utilities**
   - `TestDataFactory` for creating test data
   - `TestFileHelpers` for scripts and configs
   - `TestAssertions` for domain validation
   - `create_cli_runner()` for CLI testing

3. **Integration Tests**
   - Use `clean_git_repo` fixture for git operations
   - Create real experiments with `sample_experiment_script`
   - Extract experiment IDs from command output
   - Verify with `show` command or direct storage access

4. **Mock Strategically**
   - Mock external dependencies (pandas, uvicorn, webbrowser)
   - Don't mock core yanex functionality in integration tests
   - Use `@patch` decorator for isolated unit tests

5. **Test Coverage**
   - Run `uv run pytest tests/ --cov=yanex --cov-report=term-missing` frequently
   - Focus on uncovered lines shown in coverage report
   - Aim for meaningful tests, not just coverage numbers

---

## Success Criteria

- [ ] All new test files follow existing patterns
- [ ] Tests use available fixtures from conftest.py
- [ ] Integration tests create real experiments in clean repos
- [ ] Parametrized tests for multiple scenarios
- [ ] Mock external dependencies appropriately
- [ ] Coverage reaches 90%+ overall
- [ ] All tests pass: `uv run pytest tests/`
- [ ] Code formatted: `uv run ruff format`
- [ ] Linting passes: `uv run ruff check`
- [ ] No regressions in existing tests

---

## Next Steps

1. **Start with Phase 1** (Quick Wins) - easiest to implement, good momentum builder
2. **Prioritize List and Show commands** in Phase 2 - high user impact
3. **DataFrame API** is highest single-file impact - tackle when ready for complex tests
4. **Run tests frequently** - `uv run pytest tests/ -v` after each test addition
5. **Check coverage** - `uv run pytest tests/ --cov=yanex --cov-report=html` to identify remaining gaps

**Recommended order**:
1. Confirm module (easiest, good warm-up)
2. Delete command (similar to update)
3. UI command (simple mocking)
4. List command (important, medium difficulty)
5. Show command expansion (build on existing)
6. DataFrame API (complex but high impact)
7. Parallel executor (most complex)
