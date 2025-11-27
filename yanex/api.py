"""
Public API for yanex experiment tracking.

This module provides the main interface for experiment tracking using context managers
and thread-local storage for safe concurrent usage.
"""

import atexit
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .core.manager import ExperimentManager
from .core.param_tracking import save_accessed_params
from .core.tracked_dict import TrackedDict
from .results.experiment import Experiment
from .utils.dict_utils import get_nested_value
from .utils.exceptions import (
    AmbiguousArtifactError,
    ExperimentContextError,
    ExperimentNotFoundError,
)

# Thread-local storage for current experiment context
_local = threading.local()

# Global tracking state for parameters
_tracked_params: TrackedDict | None = None
_atexit_registered = False
_should_save_on_exit = True  # Flag to control atexit handler execution


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


def _atexit_handler_wrapper(experiment_id: str, tracked_dict: TrackedDict) -> None:
    """Wrapper for atexit handler that checks if saving is enabled.

    This wrapper allows tests to disable parameter saving on exit by setting
    _should_save_on_exit = False, preventing errors when storage is cleaned up
    before the atexit handler runs.

    Args:
        experiment_id: ID of the experiment
        tracked_dict: TrackedDict instance with tracked accesses
    """
    global _should_save_on_exit
    if _should_save_on_exit:
        try:
            save_accessed_params(experiment_id, tracked_dict)
        except Exception:
            # Silently ignore errors during exit (storage may be cleaned up)
            pass


def get_params() -> dict[str, Any]:
    """Get experiment parameters with access tracking.

    Returns TrackedDict in experiment mode to monitor which parameters are
    actually used. At script end, only accessed parameters are saved.

    Returns:
        TrackedDict of experiment parameters (empty dict in standalone mode)
    """
    global _tracked_params, _atexit_registered

    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return {}

    # Return cached tracked params if already initialized
    if _tracked_params is not None:
        return _tracked_params

    # Load raw params based on mode
    if hasattr(_local, "experiment_id"):
        # Direct API usage - read from storage
        manager = _get_experiment_manager()
        raw_params = manager.storage.load_config(experiment_id)
    else:
        # CLI subprocess mode - read from environment variables
        raw_params = {}
        for key, value in os.environ.items():
            if key.startswith("YANEX_PARAM_"):
                param_key = key[12:]  # Remove "YANEX_PARAM_" prefix
                # Try to parse as JSON for complex types, fallback to string
                try:
                    import json

                    raw_params[param_key] = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    raw_params[param_key] = value

    # Wrap in TrackedDict for access tracking
    _tracked_params = TrackedDict(raw_params)

    # Register atexit handler to save accessed params (once per process)
    if not _atexit_registered:
        atexit.register(_atexit_handler_wrapper, experiment_id, _tracked_params)
        _atexit_registered = True

    return _tracked_params


def get_param(key: str, default: Any = None) -> Any:
    """Get a specific experiment parameter with support for dot notation.

    Access is tracked for later extraction of only used parameters.

    Args:
        key: Parameter key to retrieve. Supports dot notation (e.g., "model.learning_rate")
        default: Default value if key not found

    Returns:
        Parameter value or default (default is returned in standalone mode)
    """
    params = get_params()

    # Use a sentinel to distinguish "not found" from "found but None"
    _sentinel = object()
    value = get_nested_value(params, key, default=_sentinel)

    if value is _sentinel:
        # Parameter not found, print warning and return default
        print(
            f"Warning: Parameter '{key}' not found in config. Using default value: {default}"
        )
        return default

    return value


def get_cli_args() -> dict[str, Any]:
    """Get parsed CLI arguments used to run the current experiment.

    Returns a dictionary with yanex CLI flags (not script_args - those are
    passed separately to the script).

    Returns:
        Dictionary with parsed CLI flags. Empty dict in standalone mode.
        Keys: script, config, clone_from, param, name, tag, description,
              dry_run, ignore_dirty, stage, staged, parallel

    Example:
        >>> # When run via: yanex run train.py --parallel 3 --tag ml
        >>> cli_args = yanex.get_cli_args()
        >>> cli_args['parallel']  # 3
        >>> cli_args['tag']       # ['ml']
        >>>
        >>> # Clean access with defaults
        >>> parallel = cli_args.get('parallel', 1)
        >>> tags = cli_args.get('tag', [])
        >>>
        >>> # Use in orchestrator scripts that spawn child experiments
        >>> results = yanex.run_multiple(experiments, parallel=parallel)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return {}

    # If experiment ID comes from environment (CLI mode), read from environment
    if not hasattr(_local, "experiment_id"):
        # CLI subprocess mode - read from environment variable
        cli_args_json = os.environ.get("YANEX_CLI_ARGS", "{}")
        try:
            import json

            return json.loads(cli_args_json)
        except (json.JSONDecodeError, ValueError):
            return {}
    else:
        # Direct API usage - read from experiment metadata
        manager = _get_experiment_manager()
        try:
            metadata = manager.get_experiment_metadata(experiment_id)
            return metadata.get("cli_args", {})
        except Exception:
            return {}


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


def get_artifacts_dir() -> Path | None:
    """Get absolute path to current experiment's artifacts directory.

    In standalone mode (no experiment context), returns None.
    Use copy_artifact(), save_artifact(), and load_artifact() which work
    in both modes instead.

    Returns:
        Path to artifacts directory, or None in standalone mode
    """
    exp_dir = get_experiment_dir()
    if exp_dir is None:
        return None

    return exp_dir / "artifacts"


def _get_standalone_artifacts_dir() -> Path:
    """Get artifacts directory for standalone mode.

    Creates ./artifacts/ in current working directory.

    Returns:
        Path to standalone artifacts directory
    """
    artifacts_dir = Path.cwd() / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return artifacts_dir


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


def get_dependency(slot: str) -> Experiment | None:
    """Get dependency experiment for a specific slot.

    Args:
        slot: The slot name (e.g., "data", "model", "dep1")

    Returns:
        Experiment object for the slot, or None if slot not found

    Examples:
        # Get the dependency in the "data" slot
        data_exp = yanex.get_dependency("data")
        if data_exp:
            dataset = data_exp.load_artifact("dataset.pkl")

        # Access default slots
        dep1 = yanex.get_dependency("dep1")

    Note:
        In standalone mode (no experiment context): Returns None
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return None  # Standalone mode - no dependencies

    manager = _get_experiment_manager()

    # Load dependencies dict
    dep_data = manager.storage.dependency_storage.load_dependencies(
        experiment_id, include_archived=True
    )
    dependencies = dep_data.get("dependencies", {})

    # Get the experiment ID for the slot
    dep_id = dependencies.get(slot)
    if dep_id is None:
        return None

    try:
        return Experiment(dep_id, manager)
    except Exception:
        return None


def get_dependencies(
    transitive: bool = False, include_self: bool = False
) -> dict[str, Experiment] | list[Experiment]:
    """Get experiment dependencies.

    Args:
        transitive: If True, return flat list of all transitive dependencies
        include_self: If True, include current experiment in result (only with transitive=True)

    Returns:
        If transitive=False: dict[str, Experiment] - slot name to Experiment
        If transitive=True: list[Experiment] - flat list of all dependencies

    Examples:
        # Get direct dependencies as dict
        deps = yanex.get_dependencies()
        data_exp = deps.get("data")
        if data_exp:
            print(f"Data from: {data_exp.id}")

        # Access by slot name
        deps = yanex.get_dependencies()
        for slot, dep in deps.items():
            print(f"{slot}: {dep.id} ({dep.status})")

        # Get all transitive dependencies as flat list
        all_deps = yanex.get_dependencies(transitive=True)
        for dep in all_deps:
            print(f"{dep.id} artifacts: {dep.list_artifacts()}")

    Note:
        In standalone mode (no experiment context): Returns {} or []
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return [] if transitive else {}  # Standalone mode - no dependencies

    manager = _get_experiment_manager()

    # Get dependency IDs
    if transitive:
        # Get all dependencies using DependencyResolver (flat list)
        from .core.dependencies import DependencyResolver

        resolver = DependencyResolver(manager)
        dependency_ids = resolver.get_transitive_dependencies(
            experiment_id, include_self=include_self, include_archived=True
        )

        # Create Experiment objects for each dependency
        dependencies = []
        for dep_id in dependency_ids:
            try:
                experiment = Experiment(dep_id, manager)
                dependencies.append(experiment)
            except Exception:
                continue

        return dependencies
    else:
        # Get only direct dependencies as dict
        dep_data = manager.storage.dependency_storage.load_dependencies(
            experiment_id, include_archived=True
        )
        deps_dict = dep_data.get("dependencies", {})

        # Create Experiment objects for each dependency
        dependencies = {}
        for slot, dep_id in deps_dict.items():
            try:
                experiment = Experiment(dep_id, manager)
                dependencies[slot] = experiment
            except Exception:
                continue

        return dependencies


def assert_dependency(script_name: str, slot: str | None = None) -> None:
    """Assert that a required dependency exists.

    This is a convenience method for validating dependencies at the start of a script.
    If the dependency check fails, prints an error and fails the experiment.

    Args:
        script_name: Script filename to check for (e.g., "prepare_data.py")
        slot: Optional slot name. If provided, only that slot is checked.

    Examples:
        # Check that any dependency is from prepare_data.py
        >>> import yanex
        >>> yanex.assert_dependency("prepare_data.py")

        # Check that the "data" slot dependency is from prepare_data.py
        >>> yanex.assert_dependency("prepare_data.py", slot="data")

    Note:
        In standalone mode (no experiment context): No-op, allows script to continue
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        # In standalone mode - no-op, allow script to continue
        return

    # Get dependencies as dict
    deps = get_dependencies()

    if not deps:
        # No dependencies at all
        print(f"Error: No dependency from '{script_name}' found")
        print("This experiment has no dependencies.")
        fail(f"Missing required dependency: {script_name}")
        return

    if slot is not None:
        # Check specific slot
        dep = deps.get(slot)
        if dep is None:
            print(f"Error: No dependency in slot '{slot}'")
            print(f"Available slots: {', '.join(deps.keys())}")
            fail(f"Missing required dependency slot: {slot}")
            return

        if dep.script_path and dep.script_path.name == script_name:
            return  # Success - slot matches script

        print(f"Error: Dependency in slot '{slot}' is not from '{script_name}'")
        print(
            f"Slot '{slot}' is from: {dep.script_path.name if dep.script_path else '[unknown]'}"
        )
        fail(f"Slot '{slot}' not from required script: {script_name}")
        return

    # Check if any dependency matches the script name
    for _slot_name, dep in deps.items():
        if dep.script_path and dep.script_path.name == script_name:
            return  # Success - found a match

    # No match found - print error and fail
    dep_scripts = [
        f"{slot_name}={dep.script_path.name if dep.script_path else '[unknown]'}"
        for slot_name, dep in deps.items()
    ]
    print(f"Error: No dependency from '{script_name}' found")
    print(f"Current dependencies: {', '.join(dep_scripts)}")
    fail(f"Missing required dependency: {script_name}")


def log_metrics(data: dict[str, Any], step: int | None = None) -> None:
    """Log experiment metrics for current step.

    Args:
        data: Metrics data to log
        step: Optional step number (auto-incremented if None)

    Raises:
        TypeError: If step is not an int or None

    Note:
        Does nothing in standalone mode (no active experiment context)
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        return  # No-op in standalone mode

    # Validate step parameter type
    if step is not None and not isinstance(step, int):
        raise TypeError(
            f"step parameter must be an int or None, got {type(step).__name__}"
        )

    manager = _get_experiment_manager()

    manager.storage.add_result_step(experiment_id, data, step)


def log_results(data: dict[str, Any], step: int | None = None) -> None:
    """Log experiment results for current step.

    Args:
        data: Results data to log
        step: Optional step number (auto-incremented if None)

    Note:
        Does nothing in standalone mode (no active experiment context)

    Deprecated:
        This function is deprecated. Use log_metrics() instead.
    """
    import warnings

    warnings.warn(
        "log_results() is deprecated and will be removed in a future version. "
        "Use log_metrics() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    log_metrics(data, step)


def copy_artifact(src_path: Path | str, filename: str | None = None) -> None:
    """Copy an existing file to the experiment's artifacts directory.

    Args:
        src_path: Path to source file
        filename: Name to use in artifacts dir (defaults to source filename)

    Raises:
        FileNotFoundError: If source file doesn't exist
        ValueError: If source is not a file

    Examples:
        # Copy with same name
        yanex.copy_artifact("data/results.csv")

        # Copy with different name
        yanex.copy_artifact("output.txt", "final_output.txt")

    Note:
        In standalone mode: Copies to ./artifacts/ directory
        With experiment tracking: Copies to experiment artifacts directory
    """
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode - copy to ./artifacts/
        from .core.artifact_io import copy_artifact_to_path

        artifacts_dir = _get_standalone_artifacts_dir()
        copy_artifact_to_path(src_path, artifacts_dir, filename)
    else:
        # Experiment mode - copy to experiment artifacts
        manager = _get_experiment_manager()
        manager.storage.copy_artifact(experiment_id, src_path, filename)


def save_artifact(
    obj: Any, filename: str, saver: Any | None = None, **kwargs: Any
) -> None:
    """Save a Python object to the experiment's artifacts directory.

    Format is auto-detected from filename extension.

    Args:
        obj: Python object to save
        filename: Name for saved artifact (extension determines format)
        saver: Optional custom saver function (obj, path, **kwargs) -> None
        **kwargs: Additional arguments passed to the underlying save function.
            Common examples by format:
            - Matplotlib (.png): dpi, bbox_inches, facecolor, transparent
            - JSON (.json): indent, ensure_ascii, sort_keys
            - CSV (.csv): index, sep (pandas), delimiter (list of dicts)
            - Pickle (.pkl): protocol
            - NumPy (.npz): compressed
            - PyTorch (.pt, .pth): pickle_protocol

    Supported formats (auto-detected):
        .txt        - Plain text (str.write)
        .csv        - CSV (pandas.DataFrame.to_csv or list of dicts)
        .json       - JSON (json.dump)
        .jsonl      - JSON Lines (one JSON object per line)
        .npy        - NumPy array (numpy.save)
        .npz        - NumPy arrays (numpy.savez)
        .pt, .pth   - PyTorch (torch.save)
        .pkl        - Pickle (pickle.dump)
        .png        - Matplotlib figure (fig.savefig)

    Raises:
        ValueError: If format can't be auto-detected and no custom saver provided
        ImportError: If required library not installed (e.g., torch, pandas)
        TypeError: If object type doesn't match expected type for extension

    Examples:
        # Text
        yanex.save_artifact("Training complete", "status.txt")

        # JSON with custom indent
        yanex.save_artifact({"acc": 0.95}, "metrics.json", indent=4)

        # PyTorch model
        yanex.save_artifact(model.state_dict(), "model.pt")

        # Matplotlib figure with high DPI
        yanex.save_artifact(fig, "plot.png", dpi=300, bbox_inches="tight")

        # Custom format
        def save_custom(obj, path, **kwargs):
            with open(path, 'wb') as f:
                custom_serialize(obj, f)

        yanex.save_artifact(my_obj, "data.custom", saver=save_custom)

    Note:
        In standalone mode: Saves to ./artifacts/ directory
        With experiment tracking: Saves to experiment artifacts directory
    """
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode - save to ./artifacts/
        from .core.artifact_io import _validate_filename, save_artifact_to_path

        # Validate filename to prevent path traversal
        filename = _validate_filename(filename)

        artifacts_dir = _get_standalone_artifacts_dir()
        target_path = artifacts_dir / filename
        save_artifact_to_path(obj, target_path, saver, **kwargs)
    else:
        # Experiment mode - save to experiment artifacts
        manager = _get_experiment_manager()
        manager.storage.save_artifact(experiment_id, obj, filename, saver, **kwargs)


def load_artifact(
    filename: str, loader: Any | None = None, format: str | None = None
) -> Any | None:
    """Load an artifact with automatic format detection.

    Automatically searches dependencies if artifact not found in current experiment.
    Returns None if artifact doesn't exist anywhere (allows optional artifacts).

    Args:
        filename: Name of artifact to load
        loader: Optional custom loader function (path) -> object
        format: Optional format name for explicit format selection

    Supported formats (auto-detected by extension):
        .txt        - Plain text (returns str)
        .csv        - CSV (returns pandas.DataFrame or list[dict])
        .json       - JSON (returns parsed dict/list)
        .jsonl      - JSON Lines (returns list[dict])
        .npy        - NumPy array (returns np.ndarray)
        .npz        - NumPy arrays (returns dict of arrays)
        .pt, .pth   - PyTorch (returns loaded object)
        .pkl        - Pickle (returns unpickled object)
        .png        - Image (returns PIL.Image)

    Returns:
        Loaded object, or None if artifact doesn't exist

    Raises:
        ValueError: If format can't be auto-detected and no custom loader provided
        ImportError: If required library not installed
        AmbiguousArtifactError: If artifact found in multiple dependencies

    Examples:
        # Auto-detect format from extension
        # Load from current experiment or dependencies
        model_state = yanex.load_artifact("model.pt")
        results = yanex.load_artifact("results.json")

        # Explicit format for ambiguous extensions
        workload = yanex.load_artifact("data.jsonl", format="workload")

        # Optional artifact (returns None if missing)
        checkpoint = yanex.load_artifact("checkpoint.pt")
        if checkpoint is not None:
            model.load_state_dict(checkpoint)

        # Custom loader
        def load_custom(path):
            with open(path, 'rb') as f:
                return custom_deserialize(f)

        obj = yanex.load_artifact("data.custom", loader=load_custom)

        # Explicit dependency loading (avoids ambiguity)
        deps = yanex.get_dependencies()
        model = deps[0].load_artifact("model.pt")

    Note:
        In standalone mode: Loads from ./artifacts/ directory (no dependency search)
        With experiment tracking: Searches current experiment + all dependencies
    """
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode - load from ./artifacts/ (no dependency search)
        from .core.artifact_io import load_artifact_from_path

        artifacts_dir = _get_standalone_artifacts_dir()
        artifact_path = artifacts_dir / filename

        if not artifact_path.exists():
            return None

        return load_artifact_from_path(artifact_path, loader, format=format)
    else:
        # Experiment mode - search current experiment + dependencies
        manager = _get_experiment_manager()

        # First try current experiment
        if manager.storage.artifact_exists(experiment_id, filename):
            return manager.storage.load_artifact(experiment_id, filename, loader)

        # Not in current experiment - search dependencies
        from .core.dependencies import DependencyResolver

        resolver = DependencyResolver(manager)
        found_in_id, all_matches = resolver.find_artifact_in_dependencies(
            experiment_id, filename, include_archived=True
        )

        if found_in_id is not None:
            # Found uniquely in one dependency - load it
            return manager.storage.load_artifact(
                found_in_id, filename, loader, include_archived=True, format=format
            )
        elif len(all_matches) > 1:
            # Found in multiple places - raise ambiguity error
            raise AmbiguousArtifactError(filename, all_matches)
        else:
            # Not found anywhere
            return None


def artifact_exists(filename: str) -> bool:
    """Check if an artifact exists without loading it.

    Args:
        filename: Name of artifact

    Returns:
        True if artifact exists, False otherwise

    Examples:
        if yanex.artifact_exists("checkpoint.pt"):
            model.load_state_dict(yanex.load_artifact("checkpoint.pt"))

    Note:
        In standalone mode: Checks ./artifacts/ directory
        With experiment tracking: Checks experiment artifacts directory
    """
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode - check ./artifacts/
        from .core.artifact_io import artifact_exists_at_path

        artifacts_dir = _get_standalone_artifacts_dir()
        return artifact_exists_at_path(artifacts_dir, filename)
    else:
        # Experiment mode - check experiment artifacts
        manager = _get_experiment_manager()
        return manager.storage.artifact_exists(experiment_id, filename)


def list_artifacts(
    transitive: bool = False,
) -> list[str] | dict[str, list[str]]:
    """List all artifacts in the current experiment.

    Args:
        transitive: If True, include artifacts from all dependencies.
                   Returns dict mapping experiment ID to artifact list.

    Returns:
        If transitive=False: List of artifact filenames (sorted)
        If transitive=True: Dict mapping experiment ID to artifact list

    Examples:
        # List current experiment's artifacts
        artifacts = yanex.list_artifacts()
        # Returns: ["model.pt", "metrics.json", "plot.png"]

        # List artifacts from current experiment and all dependencies
        all_artifacts = yanex.list_artifacts(transitive=True)
        # Returns: {"abc123": ["model.pt"], "def456": ["data.csv"]}

    Note:
        In standalone mode: Lists ./artifacts/ directory
        With experiment tracking: Lists experiment artifacts directory
        In standalone mode with transitive=True: Returns {"local": [...]}
    """
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode - list ./artifacts/
        from .core.artifact_io import list_artifacts_at_path

        artifacts_dir = _get_standalone_artifacts_dir()
        artifacts = list_artifacts_at_path(artifacts_dir)

        if transitive:
            return {"local": artifacts}
        return artifacts

    # Experiment mode
    manager = _get_experiment_manager()

    if not transitive:
        return manager.storage.list_artifacts(experiment_id)

    # Transitive: get current + all dependencies
    result = {}
    deps = get_dependencies(transitive=True, include_self=True)
    for dep in deps:
        result[dep.id] = dep.list_artifacts()

    return result


def execute_bash_script(
    command: str,
    timeout: float | None = None,
    raise_on_error: bool = False,
    stream_output: bool = True,
    working_dir: Path | None = None,
    artifact_prefix: str = "script",
) -> dict[str, Any]:
    """Execute bash script/command within experiment context or standalone mode.

    Args:
        command: Shell command to execute
        timeout: Optional timeout in seconds
        raise_on_error: Raise exception on non-zero exit code
        stream_output: Print output in real-time
        working_dir: Working directory (defaults to experiment directory in experiment mode, current directory in standalone mode)
        artifact_prefix: Prefix for artifact filenames (only used in experiment mode)

    Returns:
        Execution result dictionary with exit_code, stdout, stderr, etc.

    Raises:
        subprocess.TimeoutExpired: If command times out
        subprocess.CalledProcessError: If raise_on_error=True and command fails

    Note:
        In standalone mode (no active experiment context):
        - No metrics logging or artifact saving
        - No experiment-specific environment variables
        - Working directory defaults to current directory
        - All other functionality works normally
    """
    experiment_id = _get_current_experiment_id()
    standalone_mode = experiment_id is None

    # Set working directory
    if working_dir is None:
        if standalone_mode:
            working_dir = Path.cwd()
        else:
            manager = _get_experiment_manager()
            working_dir = manager.storage.get_experiment_directory(experiment_id)

    # Prepare environment
    env = os.environ.copy()

    if not standalone_mode:
        # Add experiment context environment variables
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

        # Log execution details to executions file (only in experiment mode)
        if not standalone_mode:
            manager = _get_experiment_manager()
            manager.storage.add_script_run(experiment_id, result_data)

            # Save stdout and stderr as artifacts if non-empty
            if stdout_lines:
                stdout_content = "\n".join(stdout_lines)
                save_artifact(stdout_content, f"{artifact_prefix}_stdout.txt")

            if stderr_lines:
                stderr_content = "\n".join(stderr_lines)
                save_artifact(stderr_content, f"{artifact_prefix}_stderr.txt")

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

        # Log timeout as failed execution (only in experiment mode)
        if not standalone_mode:
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
            manager = _get_experiment_manager()
            manager.storage.add_script_run(experiment_id, timeout_result)

        # Re-raise the timeout exception
        raise

    except Exception as e:
        execution_time = time.time() - start_time

        # Log execution error (only in experiment mode)
        if not standalone_mode:
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
            manager = _get_experiment_manager()
            manager.storage.add_script_run(experiment_id, error_result)

        # Re-raise the exception
        raise


def completed() -> None:
    """Manually mark experiment as completed and exit.

    Marks the current experiment as completed and exits the script gracefully.
    This triggers atexit handlers (e.g., parameter saving) before exiting.

    In CLI subprocess mode: Uses sys.exit(0) for clean exit without stack traces.
    In context manager mode: Raises exception for __exit__ to handle.

    Raises:
        ExperimentContextError: If no active experiment context
        _ExperimentCompletedException: In non-CLI mode for context manager handling
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as completed in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.complete_experiment(experiment_id)

    # Print success message
    exp_dir = manager.storage.get_experiment_directory(experiment_id)
    print(f"✓ Experiment completed successfully: {experiment_id}")
    print(f"  Directory: {exp_dir}")

    # Different behavior based on context
    if _is_cli_context():
        # CLI subprocess mode - exit cleanly without stack trace
        sys.exit(0)
    else:
        # Context manager mode - raise exception for __exit__ to handle
        raise _ExperimentCompletedException()


def fail(message: str) -> None:
    """Mark experiment as failed and exit.

    Marks the current experiment as failed with an error message and exits the script.
    This triggers atexit handlers (e.g., parameter saving) before exiting.

    In CLI subprocess mode: Uses sys.exit(1) for clean exit without stack traces.
    In context manager mode: Raises exception for __exit__ to handle.

    Args:
        message: Error message describing the failure

    Raises:
        ExperimentContextError: If no active experiment context
        _ExperimentFailedException: In non-CLI mode for context manager handling
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as failed in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.fail_experiment(experiment_id, message)

    # Print failure message
    exp_dir = manager.storage.get_experiment_directory(experiment_id)
    print(f"✗ Experiment failed: {experiment_id}")
    print(f"  Directory: {exp_dir}")

    # Different behavior based on context
    if _is_cli_context():
        # CLI subprocess mode - exit cleanly without stack trace
        sys.exit(1)
    else:
        # Context manager mode - raise exception for __exit__ to handle
        raise _ExperimentFailedException(message)


def cancel(message: str) -> None:
    """Mark experiment as cancelled and exit.

    Marks the current experiment as cancelled with a reason and exits the script.
    This triggers atexit handlers (e.g., parameter saving) before exiting.

    In CLI subprocess mode: Uses sys.exit(1) for clean exit without stack traces.
    In context manager mode: Raises exception for __exit__ to handle.

    Args:
        message: Cancellation reason

    Raises:
        ExperimentContextError: If no active experiment context
        _ExperimentCancelledException: In non-CLI mode for context manager handling
    """
    experiment_id = _get_current_experiment_id()
    if experiment_id is None:
        raise ExperimentContextError(
            "No active experiment context. Cannot mark experiment as cancelled in standalone mode."
        )

    manager = _get_experiment_manager()
    manager.cancel_experiment(experiment_id, message)

    # Print cancellation message
    exp_dir = manager.storage.get_experiment_directory(experiment_id)
    print(f"✗ Experiment cancelled: {experiment_id}")
    print(f"  Directory: {exp_dir}")

    # Different behavior based on context
    if _is_cli_context():
        # CLI subprocess mode - exit cleanly without stack trace
        sys.exit(1)
    else:
        # Context manager mode - raise exception for __exit__ to handle
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
) -> ExperimentContext:
    """Create a new experiment.

    Args:
        script_path: Path to the experiment script
        name: Optional experiment name
        config: Optional experiment configuration
        tags: Optional list of tags
        description: Optional experiment description

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
