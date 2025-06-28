# Experiment Manager Implementation Plan

## Overview

The experiment manager is the central component that orchestrates experiment lifecycle management. It provides the context manager interface, handles experiment state, and coordinates between all core components (git, config, storage, environment).

## Core Components

### 1. Experiment Manager Class (`yanex/core/manager.py`)

**Responsibilities:**
- Generate unique experiment IDs
- Manage experiment lifecycle (create, start, complete, fail, cancel)
- Coordinate with storage, git, config, and environment components
- Prevent concurrent experiment execution
- Handle experiment state transitions

**Key Methods:**
```python
class ExperimentManager:
    def __init__(self, experiments_dir: Optional[Path] = None)
    def generate_experiment_id(self) -> str
    def create_experiment(self, script_path: Path, name: Optional[str], config: Dict[str, Any], tags: List[str], description: Optional[str]) -> str
    def start_experiment(self, experiment_id: str) -> None
    def complete_experiment(self, experiment_id: str) -> None
    def fail_experiment(self, experiment_id: str, error_message: str) -> None
    def cancel_experiment(self, experiment_id: str, reason: str) -> None
    def find_experiment_by_name(self, name: str) -> Optional[str]
    def get_running_experiment(self) -> Optional[str]
    def archive_experiment(self, experiment_id: str) -> Path
```

### 2. Experiment Context Manager (`yanex/experiment.py`)

**Responsibilities:**
- Provide the public API for the `with experiment.run():` pattern
- Manage thread-local experiment state
- Handle parameter access and result logging
- Coordinate matplotlib figure saving
- Handle context entry/exit and error scenarios

**Key Functions:**
```python
def get_params() -> Dict[str, Any]
def get_param(key: str, default: Any = None) -> Any
def run() -> ExperimentContext  # Returns context manager
def get_status() -> str
def completed() -> None
def fail(message: str) -> None
def cancel(message: str) -> None
def log_results(data: Dict[str, Any], step: Optional[int] = None) -> None
def log_artifact(name: str, file_path: Path) -> None
def log_matplotlib_figure(fig, filename: str, **kwargs) -> None
def log_text(content: str, filename: str) -> None
```

## Implementation Steps

### Step 1: ID Generation and Concurrency Control

**File:** `yanex/core/manager.py` (partial)

**Features:**
- Random 8-character hex ID generation with collision detection
- Running experiment tracking (prevent parallel execution)
- Basic experiment manager structure

**Implementation Details:**
```python
def generate_experiment_id(self) -> str:
    """Generate unique 8-character hex experiment ID."""
    # Use secrets.token_hex(4) for cryptographically secure random
    # Check for collisions with existing experiments
    # Retry up to 10 times before failing

def get_running_experiment(self) -> Optional[str]:
    """Check if there's currently a running experiment."""
    # Scan experiments directory for experiments with status='running'
    # Return experiment ID if found, None otherwise

def prevent_concurrent_execution(self) -> None:
    """Ensure no other experiment is currently running."""
    # Raise ExperimentAlreadyRunningError if found
```

**Dependencies:** Storage, utils.exceptions
**Tests:** Generate IDs, collision detection, concurrency prevention

### Step 2: Experiment Creation and Metadata Management

**File:** `yanex/core/manager.py` (continuation)

**Features:**
- Experiment creation with git validation
- Metadata capture and storage
- Environment capture integration

**Implementation Details:**
```python
def create_experiment(
    self, 
    script_path: Path, 
    name: Optional[str] = None,
    config: Dict[str, Any] = None,
    tags: List[str] = None,
    description: Optional[str] = None
) -> str:
    """Create new experiment with metadata."""
    # 1. Validate git working directory is clean
    # 2. Generate unique experiment ID
    # 3. Validate inputs (name uniqueness, tags, etc.)
    # 4. Create experiment directory structure
    # 5. Capture and save environment info
    # 6. Save initial metadata with status='created'
    # 7. Save resolved configuration
    # Return experiment ID

def build_metadata(self, experiment_id: str, script_path: Path, name: Optional[str], tags: List[str], description: Optional[str]) -> Dict[str, Any]:
    """Build complete experiment metadata."""
    # Include: ID, name, script_path, git info, timestamps, status
    # Include: tags, description, environment capture
```

**Dependencies:** Git utils, environment, validation
**Tests:** Creation flow, git validation, metadata structure

### Step 3: Experiment Lifecycle Management

**File:** `yanex/core/manager.py` (continuation)

**Features:**
- Status transitions (created → running → completed/failed/cancelled)
- Metadata updates
- Experiment lookup by ID or name

**Implementation Details:**
```python
def start_experiment(self, experiment_id: str) -> None:
    """Transition experiment to running state."""
    # 1. Verify experiment exists and is in 'created' state
    # 2. Update status to 'running'
    # 3. Set start timestamp
    # 4. Save updated metadata

def complete_experiment(self, experiment_id: str) -> None:
    """Mark experiment as completed."""
    # Update status, end timestamp, duration
    
def fail_experiment(self, experiment_id: str, error_message: str) -> None:
    """Mark experiment as failed with error details."""
    # Update status, error info, end timestamp

def find_experiment_by_name(self, name: str) -> Optional[str]:
    """Find experiment ID by name."""
    # Search through metadata files for matching name
```

**Dependencies:** Storage
**Tests:** Status transitions, timestamp handling, lookup functions

### Step 4: Thread-Local State Management

**File:** `yanex/experiment.py` (partial)

**Features:**
- Thread-local current experiment tracking
- Parameter access functions
- Status query functions

**Implementation Details:**
```python
import threading

_local = threading.local()

def _get_current_experiment_id() -> str:
    """Get current experiment ID from thread-local storage."""
    if not hasattr(_local, 'experiment_id'):
        raise ExperimentContextError("No active experiment context")
    return _local.experiment_id

def _set_current_experiment_id(experiment_id: str) -> None:
    """Set current experiment ID in thread-local storage."""
    _local.experiment_id = experiment_id

def get_params() -> Dict[str, Any]:
    """Get experiment parameters."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    return manager.storage.load_config(experiment_id)
```

**Dependencies:** Core manager, threading
**Tests:** Thread isolation, context errors

### Step 5: Context Manager Implementation

**File:** `yanex/experiment.py` (continuation)

**Features:**
- ExperimentContext class with __enter__ and __exit__
- Error handling and cleanup
- Integration with experiment manager

**Implementation Details:**
```python
class ExperimentContext:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.manager = ExperimentManager()
    
    def __enter__(self):
        # 1. Set thread-local experiment ID
        # 2. Start experiment (update status to 'running')
        # 3. Return self for potential context variable access
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Clear thread-local experiment ID
        # 2. Update experiment status based on exit conditions:
        #    - Normal exit: mark as completed
        #    - Exception: mark as failed with error details
        #    - Explicit exit via fail()/cancel(): already handled
        # 3. Return False to propagate exceptions

def run() -> ExperimentContext:
    """Create experiment context manager."""
    # This will be called by CLI, needs experiment setup
    # For API usage, create experiment first, then return context
```

**Dependencies:** Core manager, threading
**Tests:** Context entry/exit, error handling, status updates

### Step 6: Result and Artifact Logging

**File:** `yanex/experiment.py` (continuation)

**Features:**
- Result logging with step management
- Artifact file handling
- Matplotlib figure integration

**Implementation Details:**
```python
def log_results(data: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log experiment results for current step."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    
    # Warn if replacing existing step
    if step is not None:
        existing_results = manager.storage.load_results(experiment_id)
        if any(r.get('step') == step for r in existing_results):
            print(f"Warning: Replacing existing results for step {step}")
    
    manager.storage.add_result_step(experiment_id, data, step)

def log_artifact(name: str, file_path: Path) -> None:
    """Log file artifact."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    manager.storage.save_artifact(experiment_id, name, file_path)

def log_matplotlib_figure(fig, filename: str, **kwargs) -> None:
    """Save matplotlib figure as artifact."""
    # 1. Import matplotlib safely (optional dependency)
    # 2. Save figure to temporary file with specified format/options
    # 3. Use log_artifact to copy to experiment artifacts
    # 4. Clean up temporary file
```

**Dependencies:** Core manager, matplotlib (optional)
**Tests:** Result logging, step replacement warnings, artifact handling

### Step 7: Advanced Features and Integration

**File:** `yanex/experiment.py` (continuation)

**Features:**
- Manual experiment control (completed, fail, cancel)
- Status queries
- Enhanced error handling

**Implementation Details:**
```python
def get_status() -> str:
    """Get current experiment status."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    metadata = manager.storage.load_metadata(experiment_id)
    return metadata['status']

def completed() -> None:
    """Manually mark experiment as completed and exit context."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    manager.complete_experiment(experiment_id)
    # Raise special exception to exit context cleanly

def fail(message: str) -> None:
    """Mark experiment as failed with message and exit context."""
    experiment_id = _get_current_experiment_id()
    manager = ExperimentManager()
    manager.fail_experiment(experiment_id, message)
    # Raise special exception to exit context
```

**Dependencies:** Core manager
**Tests:** Manual control, status queries, context exit handling

## Implementation Order and Dependencies

### Phase 1: Core Infrastructure
1. **Step 1**: ID Generation and Concurrency Control
2. **Step 4**: Thread-Local State Management
3. **Step 2**: Experiment Creation and Metadata Management

### Phase 2: Experiment Lifecycle
4. **Step 3**: Experiment Lifecycle Management
5. **Step 5**: Context Manager Implementation

### Phase 3: Feature Complete
6. **Step 6**: Result and Artifact Logging
7. **Step 7**: Advanced Features and Integration

## Testing Strategy

### Unit Tests for Each Step
- **Step 1**: ID uniqueness, collision detection, concurrency prevention
- **Step 2**: Git validation, metadata structure, environment capture
- **Step 3**: Status transitions, error handling, lookup functions
- **Step 4**: Thread isolation, context error handling
- **Step 5**: Context manager behavior, exception propagation
- **Step 6**: Result logging, artifact handling, matplotlib integration
- **Step 7**: Manual controls, status queries

### Integration Tests
- Full experiment lifecycle end-to-end
- Multiple concurrent experiment attempts
- Context manager error scenarios
- CLI integration (when implemented)

## Error Handling Strategy

### Expected Exceptions
- `ExperimentContextError`: Used outside context manager
- `ExperimentAlreadyRunningError`: Concurrent execution attempt
- `DirtyWorkingDirectoryError`: Git state validation failure
- `ExperimentNotFoundError`: Invalid experiment ID/name
- `StorageError`: File system issues
- `ValidationError`: Invalid input parameters

### Context Manager Error Handling
- **Normal exit**: Mark experiment as completed
- **Unhandled exception**: Mark as failed, capture traceback
- **Explicit fail/cancel**: Already handled, exit cleanly
- **KeyboardInterrupt**: Mark as cancelled, re-raise

## Integration Points

### CLI Integration
- `yanex run` command will use experiment manager to create and run experiments
- `yanex rerun` will use experiment lookup and context manager
- Other commands will use manager for experiment discovery and manipulation

### API Integration
- Public `experiment` module will provide clean interface
- All core functionality accessible through simple function calls
- Thread-safe for potential future multi-threading support

## Future Extension Points
- **Nested experiments**: Add parent/child experiment relationships
- **Experiment templates**: Pre-configured experiment types
- **Hooks system**: Custom callbacks for experiment lifecycle events
- **Remote storage**: Alternative storage backends
- **Experiment queuing**: Support for delayed/scheduled execution