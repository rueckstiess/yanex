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
    cancel,
    # Manual experiment control
    completed,
    create_context,
    # Experiment creation (advanced)
    create_experiment,
    fail,
    # Experiment information
    get_experiment_id,
    get_metadata,
    get_param,
    # Parameter access
    get_params,
    get_status,
    has_context,
    # Context detection
    is_standalone,
    log_artifact,
    log_matplotlib_figure,
    # Result logging
    log_results,
    log_text,
)

__version__ = "0.1.0"
__author__ = "Thomas"

__all__ = [
    # Parameter access
    "get_params",
    "get_param",
    # Context detection
    "is_standalone",
    "has_context",
    # Result logging
    "log_results",
    "log_artifact",
    "log_text",
    "log_matplotlib_figure",
    # Experiment information
    "get_experiment_id",
    "get_status",
    "get_metadata",
    # Experiment creation (advanced)
    "create_experiment",
    "create_context",
    "ExperimentContext",
    # Manual experiment control
    "completed",
    "fail",
    "cancel",
    # Internal functions (for testing)
    "_clear_current_experiment_id",
    "_ExperimentCancelledException",
    "_ExperimentCompletedException",
    "_ExperimentFailedException",
    "_get_current_experiment_id",
    "_set_current_experiment_id",
]
