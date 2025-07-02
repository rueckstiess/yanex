"""
Public API for yanex experiment tracking.

This module provides the main interface for experiment tracking using context managers
and thread-local storage for safe concurrent usage.
"""

import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.manager import ExperimentManager
from .utils.exceptions import ExperimentContextError, ExperimentNotFoundError

# Thread-local storage for current experiment context
_local = threading.local()


def _get_current_experiment_id() -> str | None:
    """Get current experiment ID from thread-local storage or environment.

    Returns:
        Current experiment ID, or None if no active experiment context
    """
    # First check thread-local storage (for direct API usage)
    if hasattr(_local, "experiment_id"):
        return _local.experiment_id

    # Then check environment variables (for CLI subprocess execution)
    return os.environ.get("YANEX_EXPERIMENT_ID")


def _set_current_experiment_id(experiment_id: str) -> None:
    """Set current experiment ID in thread-local storage.

    Args:
        experiment_id: Experiment ID to set as current
    """
    _local.experiment_id = experiment_id


def _clear_current_experiment_id() -> None:
    """Clear current experiment ID from thread-local storage."""
    if hasattr(_local, "experiment_id"):
        delattr(_local, "experiment_id")


def _get_experiment_manager() -> ExperimentManager:
    """Get experiment manager instance.

    Returns:
        ExperimentManager instance
    """
    # Use default experiments directory unless overridden
    return ExperimentManager()


def is_standalone() -> bool:
    """Check if running in standalone mode (no experiment context).

    Returns:
        True if no active experiment context exists
    """
    return _get_current_experiment_id() is None


def has_context() -> bool:
    """Check if there is an active experiment context.

    Returns:
        True if there is an active experiment context
    """
    return _get_current_experiment_id() is not None


def get_params() -> dict[str, Any]:
    """Get experiment parameters.

    Returns:
        Dictionary of experiment parameters (empty dict in standalone mode)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return {}

    # If experiment ID comes from environment (CLI mode), read params from environment
    if hasattr(_local, "experiment_id"):
        # Direct API usage - read from storage
        manager = _get_experiment_manager()
        return manager.storage.load_config(experiment_id)
    else:
        # CLI subprocess mode - read from environment variables
        params = {}
        for key, value in os.environ.items():
            if key.startswith("YANEX_PARAM_"):
                param_key = key[12:]  # Remove "YANEX_PARAM_" prefix
                # Try to parse as JSON for complex types, fallback to string
                try:
                    import json

                    params[param_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    params[param_key] = value
        return params


def get_param(key: str, default: Any = None) -> Any:
    """Get a specific experiment parameter with support for dot notation.

    Args:
        key: Parameter key to retrieve. Supports dot notation (e.g., "model.learning_rate")
        default: Default value if key not found

    Returns:
        Parameter value or default (default is returned in standalone mode)
    """
    params = get_params()

    # Handle dot notation for nested parameters
    if "." in key:
        keys = key.split(".")
        current = params

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                print(
                    f"Warning: Parameter '{key}' not found in config. Using default value: {default}"
                )
                return default

        return current
    else:
        # Simple key access
        if key not in params:
            print(
                f"Warning: Parameter '{key}' not found in config. Using default value: {default}"
            )
        return params.get(key, default)


def get_status() -> str | None:
    """Get current experiment status.

    Returns:
        Current experiment status, or None in standalone mode
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return None

    manager = _get_experiment_manager()
    return manager.get_experiment_status(experiment_id)


def get_experiment_id() -> str | None:
    """Get current experiment ID.

    Returns:
        Current experiment ID, or None in standalone mode
    """
    return _get_current_experiment_id()


def get_experiment_dir() -> Path | None:
    """Get absolute path to current experiment directory.

    Returns:
        Path to experiment directory, or None in standalone mode
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return None

    manager = _get_experiment_manager()
    return manager.storage.get_experiment_directory(experiment_id)


def get_metadata() -> dict[str, Any]:
    """Get complete experiment metadata.

    Returns:
        Complete experiment metadata (empty dict in standalone mode)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return {}

    manager = _get_experiment_manager()
    return manager.get_experiment_metadata(experiment_id)


def log_results(data: dict[str, Any], step: int | None = None) -> None:
    """Log experiment results for current step.

    Args:
        data: Results data to log
        step: Optional step number (auto-incremented if None)

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return  # No-op in standalone mode

    manager = _get_experiment_manager()

    # Warn if replacing existing step
    if step is not None:
        existing_results = manager.storage.load_results(experiment_id)
        if any(r.get("step") == step for r in existing_results):
            print(f"Warning: Replacing existing results for step {step}")

    manager.storage.add_result_step(experiment_id, data, step)


def log_artifact(name: str, file_path: Path) -> None:
    """Log file artifact.

    Args:
        name: Name for the artifact
        file_path: Path to source file

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return  # No-op in standalone mode

    manager = _get_experiment_manager()
    manager.storage.save_artifact(experiment_id, name, file_path)


def log_text(content: str, filename: str) -> None:
    """Save text content as an artifact.

    Args:
        content: Text content to save
        filename: Name for the artifact file

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return  # No-op in standalone mode

    manager = _get_experiment_manager()
    manager.storage.save_text_artifact(experiment_id, filename, content)


def log_matplotlib_figure(fig, filename: str, **kwargs) -> None:
    """Save matplotlib figure as artifact.

    Args:
        fig: Matplotlib figure object
        filename: Name for the artifact file
        **kwargs: Additional arguments passed to fig.savefig()

    Raises:
        ImportError: If matplotlib is not available

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return  # No-op in standalone mode

    try:
        import os
        import tempfile

        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for log_matplotlib_figure. Install it with: pip install matplotlib"
        ) from err

    manager = _get_experiment_manager()

    # Save figure to temporary file
    with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

        try:
            # Save figure with provided options
            fig.savefig(temp_path, **kwargs)

            # Copy to experiment artifacts
            manager.storage.save_artifact(experiment_id, filename, temp_path)
        finally:
            # Clean up temporary file
            if temp_path.exists():
                os.unlink(temp_path)


def execute_bash_script(
    command: str,
    timeout: float | None = None,
    raise_on_error: bool = False,
    stream_output: bool = True,
    working_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute bash script/command within experiment context.

    Args:
        command: Shell command to execute
        timeout: Optional timeout in seconds
        raise_on_error: Raise exception on non-zero exit code
        stream_output: Print output in real-time
        working_dir: Working directory (defaults to experiment directory)

    Returns:
        Execution result dictionary with exit_code, stdout, stderr, etc.

    Raises:
        ExperimentContextError: If no active experiment context
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If raise_on_error=True and command fails

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot execute bash script in standalone mode."
        )

    manager = _get_experiment_manager()

    # Set working directory to experiment directory if not specified
    if working_dir is None:
        working_dir = manager.storage.get_experiment_directory(experiment_id)

    # Prepare environment with experiment context
    env = os.environ.copy()
    env["YANEX_EXPERIMENT_ID"] = experiment_id

    # Add experiment parameters as environment variables
    try:
        params = get_params()
        for key, value in params.items():
            # Convert nested parameters to JSON for complex types
            if isinstance(value, dict | list):
                import json

                env[f"YANEX_PARAM_{key}"] = json.dumps(value)
            else:
                env[f"YANEX_PARAM_{key}"] = str(value)
    except Exception:
        # If we can't get params, continue without them
        pass

    # Record start time
    start_time = time.time()
    start_timestamp = datetime.utcnow().isoformat()

    stdout_lines = []
    stderr_lines = []

    try:
        # Execute command
        if stream_output:
            # Stream output in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=working_dir,
                env=env,
                bufsize=1,
                universal_newlines=True,
            )

            # Read output line by line and print
            while True:
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()

                if stdout_line:
                    print(stdout_line.rstrip())
                    stdout_lines.append(stdout_line.rstrip())

                if stderr_line:
                    print(stderr_line.rstrip(), file=__import__("sys").stderr)
                    stderr_lines.append(stderr_line.rstrip())

                if not stdout_line and not stderr_line and process.poll() is not None:
                    break

            # Wait for process to complete
            exit_code = process.wait(timeout=timeout)
        else:
            # Capture all output at once
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                env=env,
                timeout=timeout,
            )
            exit_code = result.returncode
            stdout_lines = result.stdout.splitlines() if result.stdout else []
            stderr_lines = result.stderr.splitlines() if result.stderr else []

        # Calculate execution time
        execution_time = time.time() - start_time

        # Prepare result data
        result_data = {
            "command": command,
            "exit_code": exit_code,
            "execution_time": execution_time,
            "stdout_lines": len(stdout_lines),
            "stderr_lines": len(stderr_lines),
            "working_directory": str(working_dir),
            "timestamp": start_timestamp,
        }

        # Log execution details as a result step
        log_results(result_data)

        # Save stdout and stderr as artifacts if non-empty
        if stdout_lines:
            stdout_content = "\n".join(stdout_lines)
            log_text(stdout_content, "script_stdout.txt")

        if stderr_lines:
            stderr_content = "\n".join(stderr_lines)
            log_text(stderr_content, "script_stderr.txt")

        # Prepare return value
        execution_result = {
            "exit_code": exit_code,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "execution_time": execution_time,
            "command": command,
            "working_directory": str(working_dir),
        }

        # Raise exception if requested and command failed
        if raise_on_error and exit_code != 0:
            raise subprocess.CalledProcessError(
                exit_code, command, "\n".join(stdout_lines), "\n".join(stderr_lines)
            )

        return execution_result

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time

        # Log timeout as failed execution
        timeout_result = {
            "command": command,
            "exit_code": -1,
            "execution_time": execution_time,
            "stdout_lines": len(stdout_lines),
            "stderr_lines": len(stderr_lines),
            "working_directory": str(working_dir),
            "timestamp": start_timestamp,
            "error": f"Command timed out after {timeout} seconds",
        }
        log_results(timeout_result)

        # Re-raise the timeout exception
        raise

    except Exception as e:
        execution_time = time.time() - start_time

        # Log execution error
        error_result = {
            "command": command,
            "exit_code": -1,
            "execution_time": execution_time,
            "stdout_lines": len(stdout_lines),
            "stderr_lines": len(stderr_lines),
            "working_directory": str(working_dir),
            "timestamp": start_timestamp,
            "error": str(e),
        }
        log_results(error_result)

        # Re-raise the exception
        raise


def completed() -> None:
    """Manually mark experiment as completed and exit context.

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as completed in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.complete_experiment(experiment_id)

    # Raise special exception to exit context cleanly
    raise _ExperimentCompletedException()


def fail(message: str) -> None:
    """Mark experiment as failed with message and exit context.

    Args:
        message: Error message describing the failure

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as failed in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.fail_experiment(experiment_id, message)

    # Raise special exception to exit context
    raise _ExperimentFailedException(message)


def cancel(message: str) -> None:
    """Mark experiment as cancelled with message and exit context.

    Args:
        message: Cancellation reason

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as cancelled in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.cancel_experiment(experiment_id, message)

    # Raise special exception to exit context
    raise _ExperimentCancelledException(message)


class _ExperimentCompletedException(Exception):
    """Internal exception for manual experiment completion."""

    pass


class _ExperimentFailedException(Exception):
    """Internal exception for manual experiment failure."""

    pass


class _ExperimentCancelledException(Exception):
    """Internal exception for manual experiment cancellation."""

    pass


class ExperimentContext:
    """Context manager for experiment execution."""

    def __init__(self, experiment_id: str):
        """Initialize experiment context.

        Args:
            experiment_id: Experiment ID to manage
        """
        self.experiment_id = experiment_id
        self.manager = _get_experiment_manager()
        self._manual_exit = False

    def __enter__(self):
        """Enter experiment context."""
        # Set thread-local experiment ID
        _set_current_experiment_id(self.experiment_id)

        # Start experiment (update status to 'running')
        self.manager.start_experiment(self.experiment_id)

        # Return self for potential context variable access
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit experiment context."""
        try:
            if exc_type is None:
                # Normal exit - mark as completed
                if not self._manual_exit:
                    self.manager.complete_experiment(self.experiment_id)
                    # Print completion message like CLI mode
                    exp_dir = self.manager.storage.get_experiment_directory(
                        self.experiment_id
                    )
                    print(f"✓ Experiment completed successfully: {self.experiment_id}")
                    print(f"  Directory: {exp_dir}")
            elif exc_type in (
                _ExperimentCompletedException,
                _ExperimentFailedException,
                _ExperimentCancelledException,
            ):
                # Manual exit via completed()/fail()/cancel() - already handled
                self._manual_exit = True
                # Don't propagate these internal exceptions
                return True
            elif exc_type is KeyboardInterrupt:
                # User interruption - mark as cancelled
                self.manager.cancel_experiment(
                    self.experiment_id, "Interrupted by user (Ctrl+C)"
                )
                # Print cancellation message like CLI mode
                exp_dir = self.manager.storage.get_experiment_directory(
                    self.experiment_id
                )
                print(f"✗ Experiment cancelled: {self.experiment_id}")
                print(f"  Directory: {exp_dir}")
                # Re-raise KeyboardInterrupt
                return False
            else:
                # Unhandled exception - mark as failed
                error_message = f"{exc_type.__name__}: {exc_val}"
                self.manager.fail_experiment(self.experiment_id, error_message)
                # Print failure message like CLI mode
                exp_dir = self.manager.storage.get_experiment_directory(
                    self.experiment_id
                )
                print(f"✗ Experiment failed: {self.experiment_id}")
                print(f"  Directory: {exp_dir}")
                # Propagate the original exception
                return False
        finally:
            # Always clear thread-local experiment ID
            _clear_current_experiment_id()


def create_experiment(
    script_path: Path,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
    allow_dirty: bool = False,
) -> ExperimentContext:
    """Create a new experiment.

    Args:
        script_path: Path to the experiment script
        name: Optional experiment name
        config: Optional experiment configuration
        tags: Optional list of tags
        description: Optional experiment description
        allow_dirty: Allow running with uncommitted changes (default: False)

    Returns:
        ExperimentContext for the new experiment

    Raises:
        ExperimentContextError: If experiment creation fails or if called in CLI context
    """
    # Check for CLI context conflict
    if _is_cli_context():
        raise ExperimentContextError(
            "Cannot use yanex.create_experiment() when script is run via 'yanex run'. "
            "Either:\n"
            "  - Run directly: python script.py\n"
            "  - Or remove yanex.create_experiment() and use: yanex run script.py"
        )

    manager = _get_experiment_manager()
    experiment_id = manager.create_experiment(
        script_path=script_path,
        name=name,
        config=config or {},
        tags=tags or [],
        description=description,
        allow_dirty=allow_dirty,
    )
    return ExperimentContext(experiment_id)


def create_context(experiment_id: str) -> ExperimentContext:
    """Create context for an existing experiment.

    Args:
        experiment_id: ID of the existing experiment

    Returns:
        ExperimentContext for the experiment

    Raises:
        ExperimentNotFoundError: If experiment doesn't exist
    """
    manager = _get_experiment_manager()

    # Verify experiment exists
    try:
        manager.get_experiment_metadata(experiment_id)
    except Exception as e:
        raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found") from e

    return ExperimentContext(experiment_id)


def _is_cli_context() -> bool:
    """Check if currently running in a yanex CLI-managed experiment.

    Returns:
        True if running in CLI context, False otherwise
    """
    return bool(os.environ.get("YANEX_CLI_ACTIVE"))
