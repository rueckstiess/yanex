# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Minimum Python version increased to 3.11** (dropped Python 3.10 support)
- Added Python 3.12 to officially supported versions
- Added README badges for PyPI version, Python versions, and license

## [0.5.0] - 2025-12-05

This beta release includes significant new features: a comprehensive dependency tracking system for multi-stage workflows, an AI-friendly `yanex get` command with lineage visualization, a completely redesigned artifact API, and parameter access tracking.

### Breaking Changes

- **New Artifact API**: Complete redesign of artifact handling with automatic format detection
  - Removed: `log_artifact()`, `log_text()`, `log_matplotlib_figure()`
  - Added: `copy_artifact()`, `save_artifact()`, `load_artifact()`, `artifact_exists()`, `list_artifacts()`
  - Automatic format detection based on file extension (.txt, .json, .pt, .png, etc.)
  - Standalone mode support: saves to `./artifacts/` when run without yanex tracking

- **Dependency API Changed**: Dependencies now use named slots instead of list-based IDs
  - Old: `get_dependencies()` returned `list[Experiment]`
  - New: `get_dependencies()` returns `dict[str, Experiment]` (slot name → experiment)
  - New: `get_dependency("slot_name")` for single slot access
  - Run `yanex migrate` to update existing experiments

- **Removed Clean Git State Enforcement**: Experiments no longer require a clean working directory
  - Removed `allow_dirty` parameter from `ExperimentManager.create_experiment()` and `yanex.create_experiment()`
  - Removed `allow_dirty` parameter from `yanex.run_multiple()` batch execution API
  - Removed `validate_clean_working_directory()` function from `yanex.core.git_utils`
  - Uncommitted changes are now captured automatically as git patches

- **Results API Change**: `exp.get_metrics()` now defaults to `as_dataframe=True` for consistency with the new metrics visualization API

- **Storage Renamed**: `config.yaml` renamed to `params.yaml` in experiment storage

### Added

- **Dependency Tracking System**: Complete infrastructure for multi-stage ML workflows
  - CLI: `-D data=abc123` syntax for declaring dependencies with named slots
  - API: `get_dependency("slot")`, `get_dependencies()`, `assert_dependency()`
  - Transitive dependency traversal with topological ordering
  - Artifact search across dependency chains with `load_artifact()`
  - Circular dependency detection
  - Dependency sweeps: `-D data=exp1,exp2 -D model=exp3` creates cross-product
  - Storage: `dependencies.json` file per experiment

- **AI-Friendly CLI (`yanex get` command)**: Extract specific field values for scripting and AI agents
  - Field extraction: `yanex get params <exp>`, `yanex get metrics <exp>`
  - Support for nested fields: `git.commit`, `environment.python_version`
  - Output formats: `--format json`, `--format csv`, `--format markdown`
  - Multi-experiment mode with filters: `-s`, `-n`, `-t`, `-l`
  - Real-time log streaming: `--follow/-f` for stdout/stderr (like `tail -f`)
  - Command reconstruction: `cli-command` and `run-command` fields

- **Experiment Lineage Visualization**: ASCII DAG visualization of experiment dependencies
  - `yanex get upstream <exp>` - Show what an experiment depends on
  - `yanex get downstream <exp>` - Show what depends on an experiment
  - `yanex get lineage <exp>` - Show both directions combined
  - Multi-experiment support: filter multiple experiments to visualize connected/disconnected graphs
  - `--depth N` option to limit traversal depth
  - `-F sweep` format for scripting with comma-separated IDs

- **Parameter Access Tracking**: Store only parameters actually used by scripts
  - `TrackedDict` wrapper monitors parameter access patterns
  - Two-phase save: full config at creation, accessed params at exit
  - `params.yaml` contains only parameters accessed via `get_param()`

- **Metrics Visualization API**: Multi-experiment time-series analysis with pandas
  - `yr.get_metrics()` returns long-format DataFrame for matplotlib groupby patterns
  - Auto-detection of varying parameters across experiments
  - Filter specific metrics and control parameter inclusion
  - Supports parameter sweeps, grid searches, and training curve comparisons

- **Multiple Config Files**: Merge multiple configuration files in `yanex run`
  - `--config base.yaml --config override.yaml` merges left-to-right
  - Later files take precedence over earlier ones
  - Enables modular configuration (data, model, infrastructure)

- **Automatic Git Patch Capture**: Captures uncommitted changes for reproducibility
  - Generates `git diff HEAD` patch when uncommitted changes detected
  - Saves patches as `artifacts/git_diff.patch`
  - New metadata fields: `has_uncommitted_changes`, `patch_file`
  - Secret scanning with detect-secrets integration
  - Patch size validation with warnings for large patches

- **`yanex open` Command**: Open experiment directory in file explorer
  - Cross-platform support (Finder, Explorer, xdg-open)
  - Works with ID, ID prefix, or experiment name
  - `--archived/-a` flag for archived experiments

- **`yanex migrate` Command**: Migration system for experiment data format upgrades
  - Migrates dependency_ids (list) to dependencies (dict with slots)
  - `--dry-run` mode to preview changes

- **Clone Experiments**: `--clone-from` argument to yanex run
  - Copy parameters from existing experiments
  - Override specific parameters while keeping others

- **Artifact API Enhancements**:
  - Custom artifact handlers via registry API
  - Format-specific options passed through to underlying libraries
  - `list_artifacts(transitive=True)` for dependency traversal
  - Security: Path traversal prevention, PyTorch safe loading, file size limits

- **Simplified Sweep Syntax**:
  - Comma-separated lists: `n_epochs=10,20,30` (preferred over `list(10,20,30)`)
  - Flexible range(): `range(stop)`, `range(start, stop)`, `range(start, stop, step)`

- **Results API Enhancements**:
  - `get_artifacts_dir()` and `artifacts_dir` property for easier artifact access
  - Flattened nested parameters with dot notation in show/compare commands
  - Dynamic script column width in `yanex list`
  - Middle truncation for long experiment names

- **Unified CLI Output Formats**: `--format/-F` option across all commands
  - Replaces individual `--json`, `--csv`, `--markdown` flags
  - Legacy flags still work but are hidden
  - Centralized theme constants for consistent styling

- **Parallel Test Execution**: pytest-xdist integration for 4x faster tests

- **Web UI**: Single-page application for browsing experiments
  - `yanex ui` launches FastAPI server with Next.js frontend
  - Browse, filter, and compare experiments in the browser
  - View experiment details, metrics, and artifacts

- **Results API**: Programmatic access to experiment data
  - `yanex.results.get_experiment(id)` - Load single experiment
  - `yanex.results.get_experiments(filters)` - Query multiple experiments
  - `yanex.results.get_best(metric, filters)` - Find best experiment
  - `yanex.results.compare(filters)` - Compare experiments as DataFrame
  - `Experiment` class with full access to metadata, metrics, artifacts

- **Programmatic Batch Execution**: `yanex.run_multiple()` API
  - `ExperimentSpec` for defining experiments programmatically
  - Parallel execution with `ProcessPoolExecutor`
  - K-fold cross-validation, grid search, ensemble training patterns
  - `yanex.get_cli_args()` for orchestrator scripts to access CLI flags

- **Parallel Experiment Execution**: Run multiple experiments simultaneously
  - `--parallel N` / `-j N` flag for managed execution
  - `--parallel 0` for auto-detection (CPU count)
  - Direct sweep execution without staging

- **ID Prefix Matching**: Use short ID prefixes instead of full 8-character IDs
  - `yanex show abc` matches `abc12345`
  - Works across all commands
  - Error on ambiguous matches

- **CLI Argument Shortcuts**: Concise filtering for common operations
  - `-s/--status`, `-n/--name`, `-t/--tag`, `-l/--limit`
  - Consistent across list, show, compare, archive, delete, update

- **Script Name Display**: Show script filename in experiment listings
  - `--script/-S` filter for script name patterns

- **Script Argument Pass-Through**: Pass arguments directly to experiment scripts
  - `yanex run train.py -- --epochs 10 --lr 0.001`

### Changed

- **Improved Developer Workflow**: Clean git state no longer enforced
  - Experiments run with uncommitted changes
  - Changes captured as patches for reproducibility
  - `--ignore-dirty` flag deprecated (no-op with warning)

- **Ongoing Duration Format**: Changed from "(ongoing)" suffix to "+ " prefix
  - More compact: "+ 5m 12s" instead of "5m 12s (ongoing)"

- **Sweep Experiment Naming**: Includes dependency IDs and parameter values
  - Format: `<base_name>-<dep_id>-<val1>-<val2>`
  - Values sanitized (lowercase, special chars replaced)

- **Documentation Overhaul**: Comprehensive restructuring
  - New: cli-commands.md, best-practices.md, commands/ui.md, commands/open.md, commands/get.md
  - Updated all examples to use new APIs
  - Removed version references from docs (docs reflect current state)

### Deprecated

- **`--ignore-dirty` flag**: No longer needed, displays deprecation warning
- **`log_artifact()`, `log_text()`, `log_matplotlib_figure()`**: Use `save_artifact()` instead
- **`log_results()`**: Use `log_metrics()` instead (deprecated in v0.4.0)

### Fixed

- **Empty Name Pattern Filtering**: `yanex <cmd> -n ""` now correctly filters unnamed experiments
- **Sweep Dependency Cross-Product**: Multiple `-D` flags now correctly create cross-products
- **Test Isolation**: Fixed global state issues with `per_test_experiments_dir` fixture
- **CLI Command Reconstruction**: Preserves sweep syntax in `cli-command` field

### Security

- **Artifact Path Traversal Prevention**: Validates filenames to prevent `../` attacks
- **PyTorch Safe Loading**: Uses `weights_only=True` by default
- **File Size Limits**: 100MB default limit on artifact loading
- **Secret Scanning**: Detects API keys and credentials in git patches

## [0.4.0] - 2025-07-18

### Added
- **Config-based CLI Defaults**: The `yanex run` command now supports setting CLI parameter defaults via the config file
  - Add a `yanex` section to your config file to set defaults for `--name`, `--tag`, `--description`, `--ignore-dirty`, `--dry-run`, and `--stage`
  - CLI arguments still override config defaults, maintaining expected precedence
  - Example: `yanex: {name: "my-experiment", tag: [dev, testing], ignore_dirty: true}`
  - Fully backwards compatible - existing configs without `yanex` section work unchanged
  - Includes comprehensive test coverage and documentation
- **New `log_metrics()` API Method**: Renamed `log_results()` to `log_metrics()` for clearer semantics
  - `yanex.log_metrics(data, step=None)` is the new preferred method for logging experiment metrics
  - **Metrics Merging**: Multiple calls to the same step now merge metrics instead of replacing them
  - Conflicting metric keys overwrite previous values while preserving other metrics
  - Original timestamp preserved with `last_updated` field tracking latest modifications
  - All internal usage, examples, and documentation updated to use the new method
  - Full backwards compatibility maintained - `log_results()` still works but shows deprecation warning
  - Future-proof: `log_results()` will be removed in a future major version

### Fixed
- **Real-time Output Display**: Fixed `yanex run` buffering issue where print statements appeared all at once at the end instead of in real-time
  - Added `-u` (unbuffered) flag to Python subprocess execution to force immediate output display
  - Long-running experiments now show progress incrementally as expected
  - Maintains full backward compatibility and artifact capture functionality
  - Enables better user experience with ability to monitor progress and abort if needed
- **Standalone Mode for execute_bash_script()**: Fixed `yanex.execute_bash_script()` to work in both experiment and standalone modes
  - Previously threw `ExperimentContextError` when called outside experiment context (e.g., in `python script.py`)
  - Now works seamlessly in both `yanex run script.py` (with experiment tracking) and `python script.py` (standalone mode)
  - In standalone mode: no metrics logging, no artifact saving, working directory defaults to current directory
  - In experiment mode: full functionality with metrics logging, artifact saving, and experiment directory
  - Maintains full backward compatibility for existing experiment-mode usage
- **Metrics Storage Separation**: Separated user metrics from system execution logs for better organization
  - Renamed `results.json` to `metrics.json` for user-logged metrics via `yanex.log_metrics()`
  - Created new `script_runs.json` file for bash script execution logs from `yanex.execute_bash_script()`
  - Automatic migration from legacy `results.json` to `metrics.json` with full backward compatibility
  - Script execution metadata (command, exit code, timing, etc.) now stored separately from user metrics
  - Cleaner separation between user-defined experiment metrics and system-generated script execution data
- **GitHub Actions CI Configuration**: Fixed invalid workflow configuration that caused CI failures on PR merges
  - Removed duplicate `push` events in workflow trigger configuration
  - Streamlined CI to use lightweight feature branch testing only
  - Eliminated spurious failure notifications after successful PR merges

### Deprecated
- **`log_results()` Method**: Use `log_metrics()` instead for logging experiment metrics
  - Shows deprecation warning when used, encouraging migration to new API
  - Functionally identical to `log_metrics()` - simply calls the new method internally
  - Will be removed in a future major version to clean up the API

## [0.3.0] - 2025-07-17

### Added
- **Configurable Artifact Prefix**: `execute_bash_script()` now accepts an optional `artifact_prefix` parameter to customize output filenames
  - Default behavior unchanged (uses "script" prefix: `script_stdout.txt`, `script_stderr.txt`)
  - Custom prefix enables better organization: `artifact_prefix="task1"` creates `task1_stdout.txt`, `task1_stderr.txt`
  - Comprehensive unit tests cover both default and custom prefix scenarios

## [0.2.0] - 2025-07-03

### Added
- **Bash Script Integration**: New API methods `yanex.get_experiment_dir()` and `yanex.execute_bash_script()` for seamless shell script integration
  - Automatic parameter passing via `YANEX_PARAM_*` environment variables
  - Experiment context via `YANEX_EXPERIMENT_ID` environment variable
  - Automatic stdout/stderr capture as artifacts
  - Timeout support and configurable error handling
  - Real-time output streaming option
- **Comprehensive Command Documentation**: Added detailed documentation for `show`, `archive`, `unarchive`, `delete`, and `update` commands
- **CLI Filtering Enhancements**: Support for filtering unnamed experiments with empty pattern `""`
- **Colored Terminal Output**: Rich console output with colored yanex messages (green for success, red for errors, dim for verbose info)
- **GitHub Actions CI/CD**: Comprehensive workflows for testing, release validation, and PyPI publishing
  - Matrix testing across Python 3.8-3.12 and Ubuntu/macOS/Windows
  - Code coverage reporting with Codecov integration
  - Automated dependency updates via Dependabot
- **GitHub Templates**: Pull request template, bug report template, and feature request template

### Changed
- **Major Refactoring**: Extensive codebase refactoring for maintainability and code quality
  - Extracted script execution duplication into `ScriptExecutor` class
  - Broke down storage layer monolith using composition pattern
  - Centralized CLI error handling and eliminated duplication
  - Added test infrastructure utilities to reduce duplication
- **Modern Type Annotations**: Updated to Python 3.10+ style type annotations (using `|` instead of `Union`)
- **Improved CLI Consistency**: Standardized time filtering across all commands with `--*-before`/`--*-after` pattern
- **Enhanced User Experience**: Always show experiment directory path after completion for easy access
- **Development Workflow**: Updated to use ruff exclusively (removed mypy from CI pipeline)

### Fixed
- **Parameter Sweep Naming**: Fixed sweep experiment naming to only include varying parameters, excluding constant parameters
  - Names changed from `foo-x_1-y_constant, foo-x_2-y_constant` to `foo-x_1, foo-x_2`
- **Experiment Name Length Limit**: Removed 100-character experiment name length limit and truncation logic
- **CLI Output Separation**: Fixed yanex messages to use stderr while preserving script stdout/stderr capture
- **Empty Name Pattern Filtering**: Enhanced name filtering with comprehensive edge case handling
- **Test Infrastructure**: Eliminated test duplication while maintaining zero regressions

### Security
- **Input Validation**: Enhanced validation throughout the system with proper error handling
- **Safe Operations**: Improved file operations and Git state verification

## [0.1.0] - 2025-07-01

### Added
- Initial release of Yanex experiment tracking system
- Core experiment lifecycle management
- File-based storage backend
- Command-line interface for experiment management
- Python API for experiment creation and tracking
- Git integration for code state tracking
- Configuration management with parameter overrides
- Experiment comparison and visualization tools
- Comprehensive test suite with >90% coverage
- Documentation and examples

### Security
- Input validation for all user-provided data
- Safe file operations with proper error handling
- Git state verification for experiment reproducibility