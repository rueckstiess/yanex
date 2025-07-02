# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Yanex** (Yet Another Experiment Tracker) is a lightweight, Git-aware experiment tracking system for Python designed for machine learning and research reproducibility. It's currently in alpha development (v0.1.0) and provides both CLI and Python API interfaces.

## Essential Commands

### Development Setup
```bash
make install          # Install in development mode with dev dependencies
```

### Testing
```bash
make test            # Run all tests
make test-cov        # Run tests with coverage report (90%+ required)
pytest path/to/test  # Run specific test file
```

### Code Quality
```bash
make lint           # Run ruff linting and mypy type checking
make format         # Format code with ruff
make format-check   # Check formatting without changes
make check          # Run all quality checks (format-check + lint + test)

# Direct ruff commands (use these during development)
python -m ruff check              # Check for linting errors
python -m ruff check --fix        # Auto-fix linting errors where possible
python -m ruff format             # Format all code with ruff
python -m ruff format --check     # Check if formatting is needed
```

### Build and Distribution
```bash
make build          # Build distribution packages
make clean          # Clean build artifacts and cache files
```

## Architecture Overview

### Core Components

**ExperimentManager** (`yanex/core/manager.py`)
- Central orchestration component handling experiment lifecycle
- Generates unique 8-character hex experiment IDs
- Coordinates between git, config, storage, and environment components

**Modular Storage Layer** (`yanex/core/storage_*.py`)
- Uses composition pattern for modularity instead of monolithic design
- Main interface: `storage.py` (backwards-compatible wrapper)
- Specialized components: `storage_metadata.py`, `storage_results.py`, `storage_artifacts.py`, etc.
- Archive handling: `storage_archive.py`

**Configuration System**
- Strategy pattern implementation for complex parameter handling
- Config parsing: `config.py`
- Parameter parsers: `parameter_parsers.py`, `parameter_parser_factory.py`

**CLI Architecture** (`yanex/cli/`)
- Click-based command system with centralized error handling
- Individual commands in `commands/` directory
- Shared utilities: `error_handling.py`, `filters/`, `formatters/`

### Entry Points

**CLI Entry Point**: `yanex` command → `yanex/cli/main.py`
**Python API**: `yanex.get_params()`, `yanex.log_results()` → `yanex/api.py`

### Key Patterns

- **Composition Pattern**: Storage layer modularity
- **Strategy Pattern**: Configuration parsing
- **Context Manager Pattern**: Experiment lifecycle (`yanex.create_experiment()`)
- **Factory Pattern**: Test data generation and parameter parsing

## Test Infrastructure

### Test Organization
- Comprehensive test coverage (455+ tests, 90%+ coverage required)
- Structure mirrors source code: `tests/cli/`, `tests/core/`, `tests/utils/`
- Shared fixtures in `conftest.py`
- Test utilities in `test_utils.py` with factory patterns

### Running Tests
- Each test uses isolated temporary directories and git repositories
- Use `TestDataFactory` for consistent test data generation
- Use `TestAssertions` for domain-specific validation
- Recent refactoring focused on eliminating test duplication while maintaining zero regressions

## Configuration and Dependencies

### Key Files
- `pyproject.toml`: Modern Python packaging with build system and tool configurations
- `Makefile`: Development workflow automation
- `requirements.txt`: Core runtime dependencies
- `requirements-dev.txt`: Development dependencies

### Key Dependencies
- **Click**: CLI framework
- **PyYAML**: Configuration file parsing  
- **Rich**: Terminal formatting and output
- **GitPython**: Git integration
- **textual**: Interactive terminal interfaces
- **pytest**: Testing framework
- **ruff**: Code linting and formatting
- **mypy**: Type checking

## Development Guidelines

### Code Quality Standards
- Maintain 90%+ test coverage
- All code must pass `make check` (format-check + lint + test)
- Use type hints (mypy configuration enforces strict typing)
- Follow existing patterns and conventions

### CRITICAL Development Workflow
**ALWAYS run these commands after implementing new code:**
```bash
python -m ruff check --fix    # Auto-fix linting issues
python -m ruff format         # Apply consistent formatting
python -m ruff check          # Verify no remaining lint errors
```

**Why this is essential:**
- GitHub Actions CI will fail if ruff check or format issues exist
- Local ruff behavior may differ from CI environment
- Prevents CI failures and maintains code quality standards
- Modern Python 3.10+ type annotations are required (use `X | None` not `Optional[X]`)

### Recent Architecture Changes
The codebase has undergone significant refactoring to:
- Centralize CLI error handling and eliminate duplication
- Break down storage layer monolith using composition pattern
- Extract configuration parsing complexity into strategy pattern
- Add test infrastructure utilities to reduce duplication
- Eliminate date/time parsing duplication using centralized utilities
- Modernize type annotations for Python 3.10+ (completed Dec 2024)

### Working with Experiments
- Experiments have unique 8-character hex IDs
- Git state is automatically tracked for reproducibility
- Two usage patterns: CLI-first (recommended) and explicit experiment creation
- Configuration supports YAML files with CLI parameter overrides