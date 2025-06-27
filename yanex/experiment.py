"""
Public API for yanex experiment tracking.

This module provides the main interface for experiment tracking using context managers
and thread-local storage for safe concurrent usage.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.manager import ExperimentManager
from .utils.exceptions import ExperimentContextError

# Thread-local storage for current experiment context
_local = threading.local()


def _get_current_experiment_id() -> str:
    """Get current experiment ID from thread-local storage.

    Returns:
        Current experiment ID

    Raises:
        ExperimentContextError: If no active experiment context
    """
    if not hasattr(_local, "experiment_id"):
        raise ExperimentContextError("No active experiment context. Use 'with experiment.run():' to create one.")
    return _local.experiment_id


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


def get_params() -> Dict[str, Any]:
    """Get experiment parameters.

    Returns:
        Dictionary of experiment parameters

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    manager = _get_experiment_manager()
    return manager.storage.load_config(experiment_id)


def get_param(key: str, default: Any = None) -> Any:
    """Get a specific experiment parameter.

    Args:
        key: Parameter key to retrieve
        default: Default value if key not found

    Returns:
        Parameter value or default

    Raises:
        ExperimentContextError: If no active experiment context
    """
    params = get_params()
    return params.get(key, default)


def get_status() -> str:
    """Get current experiment status.

    Returns:
        Current experiment status

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    manager = _get_experiment_manager()
    return manager.get_experiment_status(experiment_id)


def get_experiment_id() -> str:
    """Get current experiment ID.

    Returns:
        Current experiment ID

    Raises:
        ExperimentContextError: If no active experiment context
    """
    return _get_current_experiment_id()


def get_metadata() -> Dict[str, Any]:
    """Get complete experiment metadata.

    Returns:
        Complete experiment metadata

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    manager = _get_experiment_manager()
    return manager.get_experiment_metadata(experiment_id)


def log_results(data: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log experiment results for current step.

    Args:
        data: Results data to log
        step: Optional step number (auto-incremented if None)

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
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

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    manager = _get_experiment_manager()
    manager.storage.save_artifact(experiment_id, name, file_path)


def log_text(content: str, filename: str) -> None:
    """Save text content as an artifact.

    Args:
        content: Text content to save
        filename: Name for the artifact file

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
    manager = _get_experiment_manager()
    manager.storage.save_text_artifact(experiment_id, filename, content)


def log_matplotlib_figure(fig, filename: str, **kwargs) -> None:
    """Save matplotlib figure as artifact.

    Args:
        fig: Matplotlib figure object
        filename: Name for the artifact file
        **kwargs: Additional arguments passed to fig.savefig()

    Raises:
        ExperimentContextError: If no active experiment context
        ImportError: If matplotlib is not available
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for log_matplotlib_figure. "
            "Install it with: pip install matplotlib"
        ) from err
    
    import tempfile
    import os

    experiment_id = _get_current_experiment_id()
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


def completed() -> None:
    """Manually mark experiment as completed and exit context.

    Raises:
        ExperimentContextError: If no active experiment context
    """
    experiment_id = _get_current_experiment_id()
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
            elif exc_type in (_ExperimentCompletedException, _ExperimentFailedException, _ExperimentCancelledException):
                # Manual exit via completed()/fail()/cancel() - already handled
                self._manual_exit = True
                # Don't propagate these internal exceptions
                return True
            elif exc_type is KeyboardInterrupt:
                # User interruption - mark as cancelled
                self.manager.cancel_experiment(self.experiment_id, "Interrupted by user (Ctrl+C)")
                # Re-raise KeyboardInterrupt
                return False
            else:
                # Unhandled exception - mark as failed
                error_message = f"{exc_type.__name__}: {exc_val}"
                self.manager.fail_experiment(self.experiment_id, error_message)
                # Propagate the original exception
                return False
        finally:
            # Always clear thread-local experiment ID
            _clear_current_experiment_id()


def create_context(experiment_id: str) -> ExperimentContext:
    """Create experiment context manager for a specific experiment.
    
    Args:
        experiment_id: ID of existing experiment to create context for
        
    Returns:
        ExperimentContext for use with 'with' statement
        
    Raises:
        ExperimentNotFoundError: If experiment doesn't exist
    """
    # Verify experiment exists
    manager = _get_experiment_manager()
    if not manager.storage.experiment_exists(experiment_id):
        from .utils.exceptions import ExperimentNotFoundError
        raise ExperimentNotFoundError(experiment_id)
    
    return ExperimentContext(experiment_id)


def create_experiment(
    script_path: Path,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
) -> ExperimentContext:
    """Create a new experiment and return its context manager.
    
    Args:
        script_path: Path to the Python script to run
        name: Optional experiment name
        config: Configuration dictionary
        tags: List of tags for the experiment
        description: Optional experiment description
        
    Returns:
        ExperimentContext for use with 'with' statement
        
    Raises:
        DirtyWorkingDirectoryError: If git working directory is not clean
        ValidationError: If input parameters are invalid
        ExperimentAlreadyRunningError: If another experiment is running
        StorageError: If experiment creation fails
    """
    manager = _get_experiment_manager()
    experiment_id = manager.create_experiment(
        script_path=script_path,
        name=name,
        config=config,
        tags=tags,
        description=description,
    )
    return ExperimentContext(experiment_id)


def run() -> ExperimentContext:
    """Create experiment context manager.

    Note: This function is intended to be used by the CLI.
    For programmatic usage, use create_experiment() instead.

    Returns:
        ExperimentContext for use with 'with' statement

    Raises:
        ExperimentContextError: If experiment creation fails
    """
    # This will be called by CLI - experiment should already be created
    # For now, raise an error indicating this needs CLI integration
    raise ExperimentContextError(
        "Direct experiment.run() is not yet implemented. "
        "Use CLI commands like 'yanex run script.py' to run experiments, "
        "or use create_experiment() for programmatic usage."
    )
