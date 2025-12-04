# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Yanex** (Yet Another Experiment Tracker) is a lightweight, Git-aware experiment tracking system for Python designed for machine learning and research reproducibility. It's currently in beta (v0.5.0) and provides CLI, Python API, and web UI interfaces.

## Essential Commands

### Development Setup
```bash
make install          # Install in development mode with dev dependencies
```

### Testing
```bash
make test            # Run all tests (parallel by default, 4x faster)
make test-cov        # Run tests with coverage report (90%+ required)
pytest path/to/test  # Run specific test file

# Tests run in parallel by default using pytest-xdist (16 workers on multi-core systems)
# This provides 4x speedup: ~17s vs ~69s for full suite
# To run sequentially (e.g., for debugging):
pytest -n 0          # Disables parallel execution
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
- Multiple config file support: `--config` can be repeated, files merged in order (left-to-right, later takes precedence)

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

### Test Isolation and Global State

**Problem**: Some modules cache global state (like `yanex.results._default_manager`) that persists across tests, causing test isolation failures when tests run in parallel or sequentially.

**Symptoms**: Tests pass individually but fail when run together, often with errors about experiments not being found in unexpected directories.

**Available Fixtures for Isolation**:
- `per_test_experiments_dir` - Creates isolated experiments directory AND resets `yanex.results._default_manager`. Use this when tests need complete isolation from other tests.
- `clean_git_repo` - Ensures git repo is in clean state (depends on `git_repo` and `temp_dir`)
- `isolated_experiments_dir` - Creates experiments dir inside `temp_dir`
- `clean_api_state` - Cleans up `yanex.api` global state (for tests using `yanex.get_params()`)

**Best Practices**:
1. Use `per_test_experiments_dir` for tests that create experiments and invoke CLI commands
2. Use `clean_git_repo` for tests that need a clean git repository for experiment scripts
3. Don't manually set `YANEX_EXPERIMENTS_DIR` in test env overrides - use fixtures instead
4. If a test fails with "experiment not found" in the wrong directory, check for cached global state

**Example Pattern** (for CLI integration tests):
```python
def test_something(self, per_test_experiments_dir, git_repo):
    from pathlib import Path
    from yanex.core.manager import ExperimentManager

    # per_test_experiments_dir handles env and _default_manager reset
    manager = ExperimentManager(per_test_experiments_dir)
    script_path = Path(git_repo.working_dir) / "test.py"
    script_path.write_text("print('test')")

    # Create experiments...
    exp_id = manager.create_experiment(script_path, config={})

    # CLI will use the correct directory automatically
    result = self.runner.invoke(cli, ["get", "status", exp_id])
    assert result.exit_code == 0
```

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

**First-time setup or after pulling changes:**
```bash
uv sync --all-extras --dev    # Ensure correct tool versions from uv.lock
```

**ALWAYS run these commands after implementing new code:**
```bash
uv run ruff check --fix    # Auto-fix linting issues
uv run ruff format         # Apply consistent formatting (REQUIRED before push)
uv run ruff check          # Verify no remaining lint errors
```

**Why this is essential:**
- GitHub Actions CI will fail if ruff check or format issues exist
- **uv.lock pins ruff version** - local and CI must use the same version
- Run `uv sync --all-extras --dev` if you get different formatting results than CI
- Prevents CI failures and maintains code quality standards
- Modern Python 3.10+ type annotations are required (use `X | None` not `Optional[X]`)
- **ALWAYS run `uv run ruff format` before pushing to GitHub** to ensure consistent formatting across the codebase

### Recent Architecture Changes
The codebase has undergone significant refactoring to:
- Centralize CLI error handling and eliminate duplication
- Break down storage layer monolith using composition pattern
- Extract configuration parsing complexity into strategy pattern
- Add test infrastructure utilities to reduce duplication
- Eliminate date/time parsing duplication using centralized utilities
- Modernize type annotations for Python 3.10+ (completed Dec 2024)
- Add web UI with FastAPI backend and Next.js frontend

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

### Experiment Lineage Visualization
The `yanex get` command supports lineage visualization for experiment dependencies:
- `yanex get upstream <exp>` - Show what an experiment depends on
- `yanex get downstream <exp>` - Show what depends on an experiment
- `yanex get lineage <exp>` - Show both upstream and downstream

**Multi-experiment lineage** supports filtering to visualize multiple experiments:
```bash
yanex get lineage -n "train-*"       # All experiments matching pattern
yanex get upstream -s completed      # Dependencies of all completed experiments
yanex get lineage --ids a1,b2,c3     # Specific experiments by ID
```

Output behavior:
- Connected targets render as a single DAG with all targets highlighted in yellow
- Disconnected targets render as separate DAGs with blank line separation
- Single experiment is a special case (list with one element)

### Web UI Development
- Frontend: Next.js in `yanex/web/` with static export
- Backend: FastAPI serves static files + REST API
- Build workflow: `npm run build` → static export to `out/` → served by FastAPI
- Development: `cd yanex/web && npm run dev` for Next.js hot reload
- Production: `yanex ui` launches integrated server on port 8000

### Parallel Experiment Execution
- **New in v0.5.0**: Run multiple experiments simultaneously on multi-core systems
- **Enhanced in v0.6.0**: Direct parameter sweep execution without staging
- **Unrestricted concurrent execution**: Independent `yanex run` commands from different shells can run simultaneously (no restrictions)
- Use `--parallel N` flag to throttle managed execution (sweeps and staged experiments)
- `--parallel 0` uses auto-detection (number of CPU cores)
- Short flag: `-j N` (similar to `make -j`)
- Each experiment runs in isolated process with separate storage
- Useful for parameter sweeps and batch processing

#### Direct Sweep Execution (v0.6.0+)
Parameter sweeps can now execute immediately without staging:
  ```bash
  # Run sweep sequentially, immediately (NEW)
  yanex run train.py --param "lr=range(0.01, 0.1, 0.01)"

  # Run sweep in parallel, immediately (NEW)
  yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --parallel 4

  # Stage for later (existing workflow still supported)
  yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --stage
  yanex run --staged --parallel 4
  ```

- **Implementation details**:
  - Uses `ProcessPoolExecutor` for true parallelism (bypasses GIL)
  - Each experiment has unique process ID tracked in metadata
  - Sequential execution remains default for backward compatibility
  - Direct sweep execution avoids "staged" status to prevent interference
  - PID tracking added for debugging and process monitoring

### CLI Argument Access in Scripts

**New API**: `yanex.get_cli_args()` provides scripts access to CLI arguments used to invoke them.

**Use Case**: Orchestrator scripts that spawn child experiments can access CLI flags (like `--parallel`) without special-casing each flag.

**Example**:
```python
import yanex

# Get CLI args used to run this script
cli_args = yanex.get_cli_args()
# Returns: ['run', 'orchestrator.py', '--parallel', '3']

# Extract --parallel value
parallel_workers = None
if '--parallel' in cli_args:
    idx = cli_args.index('--parallel')
    parallel_workers = int(cli_args[idx + 1])

# Use when spawning children
results = yanex.run_multiple(experiments, parallel=parallel_workers)
```

**Usage**:
```bash
# Run orchestrator with parallelism
yanex run orchestrator.py --parallel 3

# Orchestrator can access --parallel via get_cli_args()
# No need to special-case each CLI flag!
```

**Implementation**: CLI args stored in experiment metadata and available via `YANEX_CLI_ARGS` environment variable.

### Programmatic Batch Execution API
- **New in v0.5.0**: Python API for running multiple experiments programmatically
- **Core module**: `yanex/executor.py` provides batch execution infrastructure
- **Shared architecture**: CLI internally uses the same executor for sweeps
- **Use cases**: K-fold cross-validation, grid search, ensemble training, batch processing

#### Key Components

**ExperimentSpec** - Specification for a single experiment:
```python
from pathlib import Path
import yanex

spec = yanex.ExperimentSpec(
    script_path=Path("train.py"),           # Script to execute
    config={"learning_rate": 0.01},         # Parameters
    script_args=["--data-exp", "abc123"],   # Script-specific arguments
    name="experiment-1",                     # Optional name
    tags=["ml", "training"],                 # Optional tags
    description="Training run 1"             # Optional description
)
```

**yanex.run_multiple()** - Execute multiple experiments:
```python
results = yanex.run_multiple(
    experiments=[spec1, spec2, spec3],  # List of ExperimentSpec objects
    parallel=4,                          # Workers (None=sequential, 0=auto)
    allow_dirty=False,                   # Allow uncommitted changes
    verbose=False                        # Show detailed output
)
```

**ExperimentResult** - Result of experiment execution:
```python
result = results[0]
# Properties: experiment_id, status, error_message, duration, name
if result.status == "completed":
    print(f"Success: {result.experiment_id}")
else:
    print(f"Failed: {result.error_message}")
```

#### Orchestration/Execution Pattern

For k-fold cross-validation and similar workflows, use a sentinel parameter to detect mode:

```python
# train.py - Acts as both orchestrator and executor
import yanex
from pathlib import Path

# Detect mode using sentinel parameter
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # ORCHESTRATION MODE: Spawn experiments for each fold
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={'_fold_idx': i, 'learning_rate': 0.01},
            name=f'fold-{i}'
        )
        for i in range(5)
    ]
    results = yanex.run_multiple(experiments, parallel=5, allow_dirty=True)
else:
    # EXECUTION MODE: Train single fold
    print(f"Training fold {fold_idx}")
    # ... training code ...
    yanex.log_metrics({'fold': fold_idx, 'accuracy': 0.95})
```

**Why this works:**
- Script runs without yanex tracking when called directly: `python train.py`
- Orchestrator creates experiments with `_fold_idx` set
- Each spawned experiment runs with yanex tracking enabled
- Experiment context prevents nested `run_multiple()` calls (raises `ExperimentContextError`)

#### Examples

**K-fold cross-validation:**
```bash
# Run the orchestration script
python examples/api/kfold_training.py --folds 5 --parallel 5
```

**Hyperparameter grid search:**
```python
experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": lr, "batch_size": bs},
        name=f"grid-lr{lr}-bs{bs}"
    )
    for lr in [0.001, 0.01, 0.1]
    for bs in [16, 32, 64]
]
results = yanex.run_multiple(experiments, parallel=4, allow_dirty=True)
```

**Ensemble training with different seeds:**
```python
experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"random_seed": seed, "learning_rate": 0.001},
        name=f"ensemble-{seed}",
        tags=["ensemble"]
    )
    for seed in range(10)
]
results = yanex.run_multiple(experiments, parallel=10, allow_dirty=True)
```

#### Error Handling

- Individual experiment failures don't abort the batch
- Each experiment marked as `failed` with error message
- Batch continues to completion regardless of failures
- Results include both successful and failed experiments

```python
results = yanex.run_multiple(experiments, parallel=4, allow_dirty=True)

completed = [r for r in results if r.status == "completed"]
failed = [r for r in results if r.status == "failed"]

print(f"Completed: {len(completed)}/{len(experiments)}")
for fail in failed:
    print(f"Failed {fail.name}: {fail.error_message}")
```

#### Code Elimination

This feature eliminated ~222 lines of duplicate code from the CLI by refactoring sweep execution to use the shared `run_multiple()` function. The CLI now builds `ExperimentSpec` objects and delegates to the executor.

## Ruff Linting Memories

- Don't add whitespace to empty lines to pass ruff's rule W293. 

## Documentation Guidelines

- After implementing user-facing changes, check if the documentation needs updating and make the necessary changes.
- At the end of a larger code change, always run all unit tests to confirm we didn't break anything else.