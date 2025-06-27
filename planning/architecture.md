# Yanex Architecture Plan

## Package Structure
```
yanex/
├── __init__.py              # Main package exports
├── experiment.py            # Public API (context manager, logging)
├── core/
│   ├── __init__.py
│   ├── manager.py           # Experiment management, ID generation
│   ├── storage.py           # File I/O, directory management
│   ├── config.py            # YAML config handling, parameter merging
│   ├── git_utils.py         # Git integration, clean state checking
│   └── environment.py       # Environment capture (Python, system info)
├── cli/
│   ├── __init__.py
│   ├── main.py              # CLI entry point, argument parsing
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── run.py           # yanex run command
│   │   ├── list.py          # yanex list command
│   │   ├── rerun.py         # yanex rerun command
│   │   ├── archive.py       # yanex archive command
│   │   └── compare.py       # yanex compare command
│   └── utils.py             # CLI utilities, formatting
└── utils/
    ├── __init__.py
    ├── exceptions.py        # Custom exceptions
    ├── validation.py        # Input validation
    └── logging.py           # Internal logging setup
```

## Component Responsibilities

### Public API Layer

#### yanex/experiment.py
- **Context Manager**: Main `run()` context manager implementation
- **Parameter Access**: `get_params()`, `get_param()` functions
- **Logging Interface**: `log_results()`, `log_artifact()`, `log_matplotlib_figure()`, `log_text()`
- **Status Control**: `get_status()`, `completed()`, `fail()`, `cancel()`
- **Thread Safety**: Thread-local experiment state management

### Core Business Logic

#### yanex/core/manager.py
- **Experiment Lifecycle**: Create, start, finish experiments
- **ID Generation**: Random 8-char hex IDs with collision detection
- **Status Management**: Track experiment states (running, completed, failed, cancelled)
- **Concurrency Control**: Prevent parallel experiment execution
- **Experiment Discovery**: Find experiments by ID or name
- **Directory Management**: Create and organize experiment folders

#### yanex/core/storage.py
- **File Operations**: Read/write metadata.json, results.json, config.yaml
- **Results Management**: Step-based result logging with replacement logic
- **Artifact Handling**: Copy files to artifacts/ directory
- **Matplotlib Integration**: Save figures with proper cleanup
- **Atomic Operations**: Ensure data consistency during writes

#### yanex/core/config.py
- **YAML Processing**: Load default and custom config files
- **Parameter Merging**: Combine YAML configs with CLI overrides
- **Type Conversion**: Handle parameter type validation and conversion
- **Default Values**: Manage default parameter resolution

#### yanex/core/git_utils.py
- **Git State Validation**: Check for clean working directory
- **Commit Info**: Extract current commit hash and branch
- **Repository Detection**: Verify git repository presence
- **Diff Analysis**: Report uncommitted changes when validation fails

#### yanex/core/environment.py
- **Python Environment**: Capture Python version, executable path
- **System Information**: OS, architecture, hostname
- **Dependencies**: Extract requirements.txt or environment.yml
- **Git Context**: Repository URL, branch, commit details

### CLI Interface Layer

#### yanex/cli/main.py
- **Argument Parsing**: Main CLI entry point with subcommands
- **Global Options**: Handle common flags and configuration
- **Error Handling**: User-friendly error messages and exit codes
- **Help System**: Generate comprehensive help documentation

#### yanex/cli/commands/run.py
- **Script Execution**: Launch Python scripts with experiment context
- **Parameter Override**: Process --param key=value arguments
- **Output Redirection**: Capture stdout/stderr to log files
- **Process Management**: Handle script interruption and cleanup

#### yanex/cli/commands/list.py
- **Experiment Discovery**: Find experiments matching filter criteria
- **Filtering Logic**: Support multiple filter types (status, date, tags, etc.)
- **Output Formatting**: Tabular display with sorting options
- **Date Parsing**: Human-readable date specifications

#### yanex/cli/commands/rerun.py
- **Experiment Lookup**: Find source experiment by ID or name
- **Parameter Inheritance**: Copy original parameters with overrides
- **Validation**: Ensure source experiment exists and is accessible

#### yanex/cli/commands/archive.py
- **Archive Operations**: Move experiments to archive directory
- **Metadata Update**: Mark experiments as archived
- **Cleanup**: Handle artifact and log file relocation

#### yanex/cli/commands/compare.py
- **Multi-Experiment Analysis**: Load data from multiple experiments
- **Interactive Table**: Rich terminal interface with sorting
- **Result Alignment**: Handle experiments with different result schemas
- **Export Options**: Save comparison results to files

#### yanex/cli/utils.py
- **Table Formatting**: Reusable table display functions
- **Date Utilities**: Parse and format timestamps
- **ID Resolution**: Convert names to IDs and vice versa
- **Progress Indicators**: Long-running operation feedback

### Utility Layer

#### yanex/utils/exceptions.py
- **Custom Exceptions**: Domain-specific error types
- **Error Hierarchy**: Organized exception inheritance
- **Context Information**: Rich error messages with debugging info

#### yanex/utils/validation.py
- **Input Validation**: Parameter type checking and constraints
- **Schema Validation**: Experiment metadata validation
- **Path Validation**: File and directory path checking

#### yanex/utils/logging.py
- **Internal Logging**: Yanex system logging configuration
- **Debug Support**: Detailed tracing for development
- **Log Rotation**: Manage yanex internal log files

## Data Flow Architecture

### Experiment Creation Flow
1. **CLI Entry**: `yanex run script.py --param key=value`
2. **Git Validation**: Check clean working directory
3. **Parameter Resolution**: Merge config YAML + CLI overrides
4. **Environment Capture**: Record system and Python environment
5. **Experiment Creation**: Generate ID, create directory structure
6. **Script Execution**: Launch Python script with experiment context

### Experiment Execution Flow
1. **Context Entry**: `with experiment.run():`
2. **Status Update**: Set status to 'running', create metadata
3. **Result Logging**: Append to results.json with step management
4. **Artifact Storage**: Copy files to artifacts/ directory
5. **Context Exit**: Update status (completed/failed), finalize logs

### CLI Query Flow
1. **Command Parsing**: Parse filter criteria and options
2. **Experiment Discovery**: Scan experiments directory
3. **Metadata Loading**: Read experiment metadata files
4. **Filtering**: Apply user-specified filters
5. **Output Formatting**: Generate tabular or interactive display

## Dependencies

### Core Dependencies
- **click**: CLI framework for command parsing and help
- **pyyaml**: YAML configuration file processing
- **rich**: Terminal formatting and interactive tables
- **gitpython**: Git repository interaction

### Optional Dependencies
- **matplotlib**: Figure saving support (optional import)
- **pandas**: Enhanced comparison features (future)

## Thread Safety & Concurrency

- Thread-local storage for experiment context
- File locking for experiment directory access
- Atomic file operations for metadata updates
- Single experiment execution enforcement

## Error Handling Strategy

- **Graceful Degradation**: Continue operation when non-critical features fail
- **Rich Error Messages**: Provide actionable error information
- **Cleanup on Failure**: Ensure partial experiments are properly marked
- **User-Friendly CLI**: Convert technical errors to user-understandable messages

## Extension Points

- **Custom Artifact Types**: Plugin system for specialized artifact handling
- **Result Processors**: Custom result formatting and validation
- **CLI Commands**: Easy addition of new commands
- **Storage Backends**: Alternative storage implementations (future)