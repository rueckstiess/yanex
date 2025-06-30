# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive parameter sweep functionality with range(), linspace(), logspace(), and list() syntax
- Parameter-aware experiment naming for sweep experiments
- Cross-product parameter expansion for multiple sweep parameters
- Staging mechanism for deferred experiment execution
- Interactive experiment comparison with rich console output
- Git integration for reproducible experiment tracking
- Comprehensive CLI with run, list, show, compare, archive commands
- Python API for programmatic experiment management
- Configuration file support (YAML/JSON) with parameter overrides
- Artifact and result logging capabilities
- Time-based and tag-based experiment filtering
- Thread-local experiment state management

### Changed
- Modernized package configuration with pyproject.toml
- Enhanced README with parameter sweep examples
- Improved error handling and validation throughout

### Fixed
- Parameter parsing precedence for numeric values
- Experiment name validation for generated sweep names
- Scientific notation formatting in experiment names
- Boolean comparison patterns in codebase

## [0.1.0] - 2024-XX-XX

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