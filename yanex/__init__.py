"""
Yanex - Yet Another Experiment Tracker

A lightweight, Git-aware experiment tracking system for Python.
"""

from .api import (
    # Parameter access
    get_params,
    get_param,
    
    # Context detection
    is_standalone,
    has_context,
    
    # Result logging
    log_results,
    log_artifact,
    log_text,
    log_matplotlib_figure,
    
    # Experiment information
    get_experiment_id,
    get_status,
    get_metadata,
    
    # Experiment creation (advanced)
    create_experiment,
    create_context,
    ExperimentContext,
    
    # Manual experiment control
    completed,
    fail,
    cancel,
    
    # Internal functions (for testing)
    _set_current_experiment_id,
    _clear_current_experiment_id,
    _get_current_experiment_id,
    _get_experiment_manager,
    _is_cli_context,
    _ExperimentCompletedException,
    _ExperimentFailedException,
    _ExperimentCancelledException,
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
]