# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Yanex** (Yet Another Experiment Tracker) is a lightweight, Git-aware experiment tracking system for Python designed for machine learning and research reproducibility. It's currently in alpha development (v0.4.0) and provides CLI, Python API, and web UI interfaces.

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
make lint           # Run ruff linting (mypy removed for alpha development)
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
make build          # Build distribution packages (includes web UI build)
make build-web      # Build Next.js web UI only (outputs to yanex/web/out/)
make clean          # Clean build artifacts and cache files
make clean-web      # Clean web UI build artifacts
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

**Web UI** (`yanex/web/`)
- Next.js 14 static site with TypeScript and Tailwind CSS
- FastAPI backend: `app.py` (main), `api.py` (endpoints)
- Build script: `build_web_ui.sh` → static export to `out/`
- Development server: `yanex/web/dev.py`

### Entry Points

**CLI Entry Point**: `yanex` command → `yanex/cli/main.py`
**Python API**: `yanex.get_params()`, `yanex.log_metrics()` → `yanex/api.py`
**Web UI**: `yanex ui` → launches FastAPI server with Next.js static frontend

### Key Patterns

- **Composition Pattern**: Storage layer modularity
- **Strategy Pattern**: Configuration parsing
- **Context Manager Pattern**: Experiment lifecycle (`yanex.create_experiment()`)
- **Factory Pattern**: Test data generation and parameter parsing

## Test Infrastructure

### Test Organization
- Comprehensive test coverage (12,000+ lines, 90%+ coverage required)
- Structure mirrors source code: `tests/cli/`, `tests/core/`, `tests/utils/`, `tests/web/`
- Shared fixtures in `conftest.py`: `temp_dir`, `git_repo`, `clean_git_repo`, `sample_config_yaml`, `sample_experiment_script`
- Test utilities in `test_utils.py` with factory patterns

### Running Tests
- Each test uses isolated temporary directories and git repositories
- Use `TestDataFactory` for consistent test data generation
- Use `TestAssertions` for domain-specific validation
- Tests run with pytest-cov for coverage reporting

## Configuration and Dependencies

### Key Files
- `pyproject.toml`: Modern Python packaging with build system and tool configurations
- `Makefile`: Development workflow automation
- `requirements.txt`: Core runtime dependencies
- `requirements-dev.txt`: Development dependencies

### Key Dependencies
**Python Core:**
- **Click**: CLI framework
- **PyYAML**: Configuration file parsing
- **Rich**: Terminal formatting and output
- **GitPython**: Git integration
- **textual**: Interactive terminal interfaces
- **FastAPI + uvicorn**: Web UI backend server
- **httpx**: HTTP client for API communication
- **dateparser**: Natural language date parsing

**Development:**
- **pytest + pytest-cov**: Testing framework and coverage
- **ruff**: Code linting and formatting (includes basic type checking)
- **mypy**: Type checking (disabled for alpha development)

**Web UI (Node.js):**
- **Next.js 14**: React framework with static export
- **TypeScript**: Type safety for web code
- **Tailwind CSS**: Styling framework
- **Recharts**: Data visualization

## Development Guidelines

### Code Quality Standards
- Maintain 90%+ test coverage
- All code must pass `make check` (format-check + lint + test)
- Use type hints (ruff provides basic type checking, mypy disabled for alpha development)
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
- Add web UI with FastAPI backend and Next.js frontend (v0.4.0)

### Working with Experiments
- Experiments have unique 8-character hex IDs generated via `secrets.token_hex(4)`
- Git state is automatically tracked for reproducibility (commit hash, branch, dirty status)
- Configuration supports YAML files with CLI parameter overrides
- Experiment lifecycle: `created` → `running` → `completed`/`failed`/`cancelled`
- Default storage location: `~/.yanex/experiments/` (configurable via `YANEX_EXPERIMENTS_DIR`)

### Two Execution Patterns
**1. CLI-First (Recommended):**
- Run via `yanex run script.py --param key=value`
- Script uses `yanex.get_params()` and `yanex.log_metrics()` which work standalone or with tracking
- Experiment context set via environment variables (`YANEX_EXPERIMENT_ID`, `YANEX_CLI_ACTIVE`)

**2. Explicit Context (Advanced):**
- Use `yanex.create_experiment()` context manager in code
- Intended for Jupyter notebooks or programmatic control
- Cannot mix with CLI-first pattern (raises `ExperimentContextError`)

### Web UI Development
- Frontend: Next.js in `yanex/web/` with static export
- Backend: FastAPI serves static files + REST API
- Build workflow: `npm run build` → static export to `out/` → served by FastAPI
- Development: `cd yanex/web && npm run dev` for Next.js hot reload
- Production: `yanex ui` launches integrated server on port 8000

## Ruff Linting Memories

- Don't add whitespace to empty lines to pass ruff's rule W293. 

## Documentation Guidelines

- After implementing user-facing changes, check if the documentation needs updating and make the necessary changes.