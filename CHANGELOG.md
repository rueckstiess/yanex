# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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