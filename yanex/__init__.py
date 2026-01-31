"""
Yanex - Yet Another Experiment Tracker

A lightweight, Git-aware experiment tracking system for Python.
"""

from .api import (
    ExperimentContext,
    _clear_current_experiment_id,
    _ExperimentCancelledException,
    _ExperimentCompletedException,
    _ExperimentFailedException,
    _get_current_experiment_id,
    # Internal functions (for testing)
    _set_current_experiment_id,
    # Artifact management
    artifact_exists,
    # Dependency tracking
    assert_dependency,
    cancel,
    # Manual experiment control
    completed,
    copy_artifact,
    create_context,
    # Experiment creation (advanced)
    create_experiment,
    # Script execution
    execute_bash_script,
    fail,
    # Experiment information
    get_artifacts_dir,
    get_cli_args,
    get_dependencies,
    get_dependency,
    get_experiment_dir,
    get_experiment_id,
    get_metadata,
    get_param,
    # Parameter access
    get_params,
    get_status,
    has_context,
    # Context detection
    is_standalone,
    list_artifacts,
    load_artifact,
    # Result logging
    log_metrics,
    log_results,
    save_artifact,
)

# Custom artifact format registration
from .core.artifact_formats import register_format

# Batch execution API
from .executor import ExperimentResult, ExperimentSpec, run_multiple
from .results.experiment import Experiment

# Exceptions
from .utils.exceptions import ParameterConflictError

__version__ = "0.6.0a3"
__author__ = "Thomas"

__all__ = [
    # Parameter access
    "get_params",
    "get_param",
    "get_cli_args",
    # Context detection
    "is_standalone",
    "has_context",
    # Result logging
    "log_metrics",
    "log_results",
    # Artifact management
    "copy_artifact",
    "save_artifact",
    "load_artifact",
    "artifact_exists",
    "list_artifacts",
    "register_format",
    # Script execution
    "execute_bash_script",
    # Experiment information
    "get_experiment_id",
    "get_experiment_dir",
    "get_artifacts_dir",
    "get_status",
    "get_metadata",
    # Dependency tracking
    "get_dependencies",
    "get_dependency",
    "assert_dependency",
    "Experiment",
    # Experiment creation (advanced)
    "create_experiment",
    "create_context",
    "ExperimentContext",
    # Manual experiment control
    "completed",
    "fail",
    "cancel",
    # Batch execution API
    "ExperimentSpec",
    "ExperimentResult",
    "run_multiple",
    # Exceptions
    "ParameterConflictError",
    # Internal functions (for testing)
    "_clear_current_experiment_id",
    "_ExperimentCancelledException",
    "_ExperimentCompletedException",
    "_ExperimentFailedException",
    "_get_current_experiment_id",
    "_set_current_experiment_id",
]
